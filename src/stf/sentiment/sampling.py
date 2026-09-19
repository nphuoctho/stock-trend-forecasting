"""Stratified candidate selection for expanding the in-domain sentiment label set.

The in-domain corpus is naturally dominated by neutral corporate filings, so a
uniform random draw spends most of the annotation budget on the majority class:
the first 306-row batch yielded only 31 ``NEGATIVE`` samples, which is too few to
estimate that class's recall at any useful precision.

This module separates two jobs that must not be conflated:

* an **evaluation** stratum drawn uniformly at random, which preserves the corpus
  label prior and therefore supports an unbiased estimate of real-world
  performance;
* **enrichment** strata drawn with a minority-class prior, which spend the budget
  where the classifier is weakest and are used for training only.

Enrichment uses two cheap signals over text the crawl already stores: a
Vietnamese financial polarity lexicon, and the current checkpoint's own
probabilities (already materialised by ``score-news``). Neither is treated as a
label — they only decide which articles a human is asked to read. The sampling
design, the per-stratum draw sizes and the inclusion probabilities are written to
a manifest so the thesis can state exactly how the set was built.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Vietnamese financial-news polarity cues. These retrieve candidates; they never
# assign a label. Terms are matched on lowercased, accent-preserving text.
NEGATIVE_CUES: tuple[str, ...] = (
    "lỗ", "thua lỗ", "lỗ ròng", "lỗ lũy kế", "âm vốn",
    "giảm lãi", "lãi giảm", "sụt giảm", "lao dốc", "giảm sàn", "bán tháo",
    "doanh thu giảm", "lợi nhuận giảm", "kém khả quan", "suy giảm",
    "nợ xấu", "quá hạn", "chậm thanh toán", "chậm trả", "mất khả năng thanh toán",
    "vỡ nợ", "gia hạn nợ", "cơ cấu nợ",
    "xử phạt", "vi phạm", "phạt hành chính", "truy thu", "cưỡng chế",
    "khởi tố", "bắt tạm giam", "điều tra", "thanh tra", "kiểm toán ngoại trừ",
    "ý kiến ngoại trừ", "hồi tố",
    "đình chỉ", "hủy niêm yết", "hạn chế giao dịch", "cảnh báo", "kiểm soát",
    "rút giấy phép", "thu hồi",
    "giải chấp", "bán giải chấp", "call margin",
    "thoái vốn bắt buộc", "chậm công bố", "chậm nộp",
    "kiện", "tranh chấp", "khiếu nại", "thu hồi sản phẩm",
    "hoãn", "trì hoãn", "chậm tiến độ", "dừng dự án",
    "từ nhiệm", "miễn nhiệm", "bán ra", "đăng ký bán",
)

POSITIVE_CUES: tuple[str, ...] = (
    "lãi", "lợi nhuận tăng", "lãi tăng", "tăng trưởng", "vượt kế hoạch",
    "hoàn thành kế hoạch", "kỷ lục", "cao nhất", "khởi sắc",
    "doanh thu tăng", "trúng thầu", "ký hợp đồng", "mở rộng",
    "chia cổ tức", "cổ tức", "thưởng cổ phiếu", "mua vào", "đăng ký mua",
    "khối ngoại mua", "nâng khuyến nghị", "tăng trần", "bứt phá",
    "khánh thành", "đưa vào hoạt động", "phê duyệt", "chấp thuận",
    "hợp tác chiến lược", "đối tác chiến lược", "rót vốn", "giải ngân",
)

# Draw sizes for the annotation budget. ``eval_random`` is deliberately drawn
# without any model or lexicon input so it can carry the corpus prior.
DEFAULT_QUOTAS: dict[str, int] = {
    "eval_random": 350,
    "train_negative": 400,
    "train_positive": 250,
    "train_active": 0,
}

STRATA: tuple[str, ...] = tuple(DEFAULT_QUOTAS)

# Which strata may be used for evaluation. A stratum selected with help from the
# checkpoint's own probabilities can never be evaluated on: the model would be
# scored on rows it chose, so the number would not describe the corpus.
STRATUM_USAGE: dict[str, str] = {
    "eval_random": "eval_or_train",
    "train_negative": "train_only",
    "train_positive": "train_only",
    "train_active": "train_only",
}

# Whether the checkpoint influenced selection, recorded per row for the report.
STRATUM_MODEL_INFORMED: dict[str, bool] = {
    "eval_random": False,
    "train_negative": False,
    "train_positive": False,
    "train_active": True,
}

# Existing in-domain rows carry a 400-character ``body_preview``. New rows keep the
# same cap so one merged training file has a single input-length regime, and add a
# longer ``body_context`` that only the human annotator reads.
BODY_PREVIEW_CHARS = 400
BODY_CONTEXT_CHARS = 2000

# No single ticker may take more than this share of an enrichment stratum.
MAX_TICKER_SHARE = 0.25

LABEL_TEMPLATE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "ticker",
    "published_at",
    "title",
    "body_preview",
    "url",
    "label",
    "stratum",
    "usage",
    "model_informed",
    "selection_reason",
    "body_context",
    "model_prob_negative",
    "model_prob_neutral",
    "model_prob_positive",
    "annotation_note",
)


@dataclass(frozen=True)
class SamplingPlan:
    """One reproducible candidate draw."""

    quotas: dict[str, int]
    seed: int = 42
    body_preview_chars: int = BODY_PREVIEW_CHARS
    body_context_chars: int = BODY_CONTEXT_CHARS
    max_ticker_share: float = MAX_TICKER_SHARE


def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern[str]:
    ordered = sorted(cues, key=len, reverse=True)
    return re.compile("|".join(re.escape(cue) for cue in ordered))


_NEGATIVE_RE = _cue_pattern(NEGATIVE_CUES)
_POSITIVE_RE = _cue_pattern(POSITIVE_CUES)


def count_cues(text: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    """Number of distinct lexicon hits per row, on lowercased text."""
    lowered = text.fillna("").astype(str).str.lower()
    return lowered.map(lambda value: len(set(pattern.findall(value))))


def prepare_pool(
    articles: pd.DataFrame,
    *,
    scored: pd.DataFrame | None = None,
    exclude_urls: set[str] | None = None,
) -> pd.DataFrame:
    """Build the deduplicated, not-yet-labelled candidate pool.

    ``articles`` is the ticker/article join. ``scored`` is an optional
    ``score-news`` parquet whose probabilities are reused as a retrieval prior, so
    no extra inference is needed. Rows already present in a label file are removed
    via ``exclude_urls`` to guarantee the new batch cannot re-annotate old rows.
    """
    required = {"ticker", "url", "published_at", "title"}
    missing = sorted(required - set(articles.columns))
    if missing:
        raise ValueError(f"articles missing columns {missing}.")

    pool = articles.copy()
    if exclude_urls:
        pool = pool[~pool["url"].isin(exclude_urls)]
    body = (
        pool["body"].fillna("").astype(str)
        if "body" in pool.columns
        else pd.Series("", index=pool.index)
    )
    pool = pool.assign(_body=body)

    # One annotation task per article text, not per ticker mention: the same
    # article linked to several tickers must not be labelled twice.
    pool["_title_key"] = pool["title"].fillna("").astype(str).str.strip().str.lower()
    pool["_body_key"] = pool["_body"].str.strip().str.lower().str.slice(0, 400)
    pool = pool.sort_values(["url", "ticker"]).drop_duplicates(subset=["url"])
    pool = pool[pool["_title_key"] != ""]
    pool = pool.drop_duplicates(subset=["_title_key"])
    pool = pool[pool["_body_key"] != ""].drop_duplicates(subset=["_body_key"])

    searchable = pool["title"].fillna("").astype(str) + " " + pool["_body"]
    pool["negative_cues"] = count_cues(searchable, _NEGATIVE_RE)
    pool["positive_cues"] = count_cues(searchable, _POSITIVE_RE)

    prob_cols = ["prob_negative", "prob_neutral", "prob_positive"]
    if scored is not None and not scored.empty and {"url", *prob_cols} <= set(scored.columns):
        priors = scored.drop_duplicates(subset=["url"])[["url", *prob_cols]]
        pool = pool.merge(priors, on="url", how="left")
    else:
        for col in prob_cols:
            pool[col] = np.nan
    return pool.reset_index(drop=True)


def _capped_weighted_draw(
    candidates: pd.DataFrame,
    weights: pd.Series,
    quota: int,
    rng: np.random.Generator,
    max_ticker_share: float,
) -> pd.DataFrame:
    """Sample ``quota`` rows without replacement, weighted, capped per ticker.

    Taking the top-``quota`` rows of a ranking concentrates the batch on whatever
    template dominates the cue counts -- recurring market wrap-ups all share the
    same vocabulary -- and on the two tickers with the densest news. A weighted
    draw keeps the enrichment while preserving variety, and the per-ticker cap
    stops MBB/TCB from swamping a stratum.
    """
    if candidates.empty or quota < 1:
        return candidates.head(0)
    tickers = candidates["ticker"].astype(str)
    # The share cap must never be tighter than an equal split, or it would be
    # impossible to fill the quota and the draw would silently exceed it anyway.
    n_tickers = max(1, tickers.nunique())
    cap = max(
        1,
        int(np.ceil(quota * max_ticker_share)),
        int(np.ceil(quota / n_tickers)),
    )
    weight = np.clip(weights.to_numpy(dtype=float), 1e-9, None)
    order = rng.choice(
        len(candidates),
        size=len(candidates),
        replace=False,
        p=weight / weight.sum(),
    )
    ticker_values = tickers.to_numpy()
    per_ticker: dict[str, int] = {}
    taken: list[int] = []
    for position in order:
        if len(taken) >= quota:
            break
        ticker = ticker_values[position]
        if per_ticker.get(ticker, 0) >= cap:
            continue
        per_ticker[ticker] = per_ticker.get(ticker, 0) + 1
        taken.append(int(position))
    # Only a ticker running out of eligible rows can leave the quota short; in
    # that case fill the remainder rather than returning an undersized stratum.
    if len(taken) < quota:
        chosen = set(taken)
        for position in order:
            if len(taken) >= quota:
                break
            if int(position) not in chosen:
                taken.append(int(position))
                chosen.add(int(position))
    return candidates.iloc[sorted(taken)]


def draw_candidates(pool: pd.DataFrame, plan: SamplingPlan) -> pd.DataFrame:
    """Draw disjoint strata from ``pool`` in a fixed, documented order.

    ``eval_random`` is drawn first, uniformly, using neither the lexicon nor the
    checkpoint, so it preserves the corpus label prior and stays valid for an
    unbiased estimate. The enrichment strata are then drawn from what remains,
    which keeps every stratum disjoint and gives the manifest an exact inclusion
    probability per stratum.

    Enrichment is ordered by the lexicon, not by the checkpoint's probabilities.
    The lexicon is a fixed rule that never saw a label, so its measured lift
    (NEGATIVE rate 0.101 -> 0.289 across all 306 labelled rows) is an honest
    out-of-sample estimate. The checkpoint's own probabilities looked far better
    but were measured on rows it had trained on; re-measured on fold-01's 62
    unseen rows the advantage disappears, so they are used only to widen
    eligibility and to build the explicitly model-informed ``train_active``
    stratum.
    """
    rng = np.random.default_rng(plan.seed)
    remaining = pool.copy()
    picked: list[pd.DataFrame] = []
    design: dict[str, dict] = {}

    def take(frame: pd.DataFrame, stratum: str, reason: str, eligible: int) -> None:
        nonlocal remaining
        requested = plan.quotas.get(stratum, 0)
        if frame.empty:
            design[stratum] = {"requested": requested, "drawn": 0, "eligible": eligible}
            return
        chosen = frame.assign(
            stratum=stratum,
            usage=STRATUM_USAGE[stratum],
            model_informed=STRATUM_MODEL_INFORMED[stratum],
            selection_reason=reason,
        )
        picked.append(chosen)
        remaining = remaining[~remaining["url"].isin(chosen["url"])]
        design[stratum] = {
            "requested": requested,
            "drawn": int(len(chosen)),
            "eligible": int(eligible),
            "inclusion_probability": round(len(chosen) / eligible, 6) if eligible else None,
            "usage": STRATUM_USAGE[stratum],
            "model_informed": STRATUM_MODEL_INFORMED[stratum],
            "ticker_counts": chosen["ticker"].astype(str).value_counts().to_dict(),
        }

    # 1. Evaluation stratum: uniform, model-free, lexicon-free.
    quota = plan.quotas.get("eval_random", 0)
    if quota > 0:
        eligible = len(remaining)
        size = min(quota, eligible)
        idx = rng.choice(eligible, size=size, replace=False)
        take(
            remaining.iloc[np.sort(idx)],
            "eval_random",
            "uniform random over the deduplicated pool; no model or lexicon input",
            eligible,
        )

    # 2/3. Lexicon-enriched training strata, weighted by cue count.
    for stratum, cue_col, other_col in (
        ("train_negative", "negative_cues", "positive_cues"),
        ("train_positive", "positive_cues", "negative_cues"),
    ):
        quota = plan.quotas.get(stratum, 0)
        if quota < 1:
            continue
        candidates = remaining[remaining[cue_col] > 0].copy()
        eligible = len(candidates)
        weights = candidates[cue_col].astype(float) + 1.0
        # Articles carrying the opposite polarity too are less likely to be clean.
        weights = weights / (1.0 + candidates[other_col].astype(float))
        take(
            _capped_weighted_draw(candidates, weights, quota, rng, plan.max_ticker_share),
            stratum,
            f"{cue_col} > 0; weighted random draw by cue count, capped per ticker",
            eligible,
        )

    # 4. Optional model-informed stratum: lowest checkpoint confidence. Train-only.
    quota = plan.quotas.get("train_active", 0)
    if quota > 0:
        prob_cols = ["prob_negative", "prob_neutral", "prob_positive"]
        candidates = remaining.dropna(subset=prob_cols).copy()
        eligible = len(candidates)
        confidence = candidates[prob_cols].max(axis=1)
        take(
            _capped_weighted_draw(
                candidates, 1.0 - confidence, quota, rng, plan.max_ticker_share
            ),
            "train_active",
            "lowest maximum class probability under the current checkpoint",
            eligible,
        )

    if not picked:
        raise ValueError("No candidates drawn; check the pool and quotas.")
    out = pd.concat(picked, ignore_index=True)
    if out["url"].duplicated().any():
        raise AssertionError("Strata are not disjoint; a url was drawn twice.")
    out.attrs["design"] = design
    return out


def to_label_template(candidates: pd.DataFrame, plan: SamplingPlan) -> pd.DataFrame:
    """Shape a draw into the annotation file, with an empty ``label`` column.

    The first seven columns match ``to_label_r1.csv`` exactly so the existing
    loader and notebook read this file unchanged. ``body_preview`` keeps the same
    400-character cap as the existing rows, giving one merged file a single input
    regime; ``body_context`` carries more text purely for the human to read and is
    never a model input.
    """
    body = candidates["_body"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "sample_id": np.arange(len(candidates), dtype=int),
            "ticker": candidates["ticker"].to_numpy(),
            "published_at": candidates["published_at"].to_numpy(),
            "title": candidates["title"].fillna("").to_numpy(),
            "body_preview": body.str.slice(0, plan.body_preview_chars).to_numpy(),
            "url": candidates["url"].to_numpy(),
            "label": pd.array([pd.NA] * len(candidates), dtype="string"),
            "stratum": candidates["stratum"].to_numpy(),
            "usage": candidates["usage"].to_numpy(),
            "model_informed": candidates["model_informed"].to_numpy(),
            "selection_reason": candidates["selection_reason"].to_numpy(),
            "body_context": body.str.slice(0, plan.body_context_chars).to_numpy(),
            "model_prob_negative": candidates["prob_negative"].to_numpy(),
            "model_prob_neutral": candidates["prob_neutral"].to_numpy(),
            "model_prob_positive": candidates["prob_positive"].to_numpy(),
            "annotation_note": pd.array([pd.NA] * len(candidates), dtype="string"),
        }
    )
    return out[list(LABEL_TEMPLATE_COLUMNS)]


# Provenance written by ``finalize_labels`` once a human has confirmed the rows.
REVIEWED_SOURCE = "human_reviewed"
REVIEWED_STATUS = "REVIEWED"

# Columns a merged training file must carry. The first seven match
# ``to_label_r1.csv`` so the existing loader and notebook read it unchanged.
MERGE_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "ticker",
    "published_at",
    "title",
    "body_preview",
    "url",
    "label",
    "batch",
    "stratum",
    "usage",
    "annotation_source",
    "annotation_status",
)


def finalize_labels(
    frame: pd.DataFrame, *, valid_labels: tuple[str, ...]
) -> tuple[pd.DataFrame, dict]:
    """Mark rows carrying a valid label as human-reviewed.

    ``reject_preliminary_labels`` refuses a row when *either* provenance field
    still marks it preliminary, so both must be rewritten together; flipping only
    the status leaves the file rejected. Rows without a valid label keep their
    preliminary provenance, so a partially reviewed file stays honest.

    Also reports how often the confirmed label differs from the stored
    preliminary one. Plan section 6 permits that figure as a diagnostic of the
    prelabelling step; it is not an inter-rater agreement measure, because there
    is only one human annotator.
    """
    if "label" not in frame.columns:
        raise ValueError("Annotation file has no 'label' column.")
    out = frame.copy()
    label = out["label"].astype("string").str.strip()
    valid = label.isin(list(valid_labels))

    for column, value in (
        ("annotation_source", REVIEWED_SOURCE),
        ("annotation_status", REVIEWED_STATUS),
    ):
        if column not in out.columns:
            out[column] = pd.array([pd.NA] * len(out), dtype="string")
        out[column] = out[column].astype("string").mask(valid, value)

    changed = None
    if "prelabel" in out.columns:
        stored = out["prelabel"].astype("string").str.strip()
        changed = int((valid & stored.notna() & (stored != label)).sum())

    report = {
        "rows": int(len(out)),
        "finalized": int(valid.sum()),
        "left_preliminary": int((~valid).sum()),
        "distribution": {
            name: int((label[valid] == name).sum()) for name in valid_labels
        },
        "changed_from_prelabel": changed,
    }
    return out, report


def merge_label_files(
    sources: dict[str, pd.DataFrame], *, valid_labels: tuple[str, ...]
) -> tuple[pd.DataFrame, dict]:
    """Union several annotation files into one training file.

    ``sources`` maps a batch name to its frame. Every batch numbers its rows from
    zero, so the merged ``sample_id`` is rewritten as ``<batch>-<n>`` to stay a
    unique key; the original number is kept in ``source_sample_id``. Only rows
    with a valid label survive, duplicates are removed by URL with earlier
    batches winning, and files predating the stratified design get
    ``stratum``/``usage`` defaults that keep them evaluable.
    """
    frames = []
    per_batch: dict[str, dict] = {}
    for batch, raw in sources.items():
        if raw is None or raw.empty:
            continue
        frame = raw.copy()
        label = frame["label"].astype("string").str.strip()
        frame = frame[label.isin(list(valid_labels))].copy()
        frame["label"] = frame["label"].astype("string").str.strip()
        frame["source_sample_id"] = frame.get(
            "sample_id", pd.Series(range(len(frame)), index=frame.index)
        )
        frame["batch"] = batch
        frame["sample_id"] = [f"{batch}-{int(i)}" for i in frame["source_sample_id"]]
        # Rows from before the stratified draw carry the corpus prior by construction.
        if "stratum" not in frame.columns:
            frame["stratum"] = "baseline_random"
        if "usage" not in frame.columns:
            frame["usage"] = "eval_or_train"
        frame["stratum"] = frame["stratum"].astype("string").fillna("baseline_random")
        frame["usage"] = frame["usage"].astype("string").fillna("eval_or_train")
        for column in ("annotation_source", "annotation_status"):
            if column not in frame.columns:
                frame[column] = pd.array([pd.NA] * len(frame), dtype="string")
        per_batch[batch] = {
            "rows": int(len(frame)),
            "distribution": {
                name: int((frame["label"] == name).sum()) for name in valid_labels
            },
        }
        frames.append(frame)

    if not frames:
        raise ValueError("No labeled rows found in the given files.")
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    if "url" in merged.columns:
        merged = merged.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)

    missing = [c for c in MERGE_COLUMNS if c not in merged.columns]
    if missing:
        raise ValueError(f"Merged frame is missing {missing}.")
    ordered = [*MERGE_COLUMNS, "source_sample_id"]
    merged = merged[ordered]

    evaluable = merged["usage"] == "eval_or_train"
    report = {
        "per_batch": per_batch,
        "rows": int(len(merged)),
        "duplicate_urls_removed": int(before - len(merged)),
        "distribution": {
            name: int((merged["label"] == name).sum()) for name in valid_labels
        },
        "evaluable_rows": int(evaluable.sum()),
        "evaluable_distribution": {
            name: int((merged.loc[evaluable, "label"] == name).sum())
            for name in valid_labels
        },
        "train_only_rows": int((~evaluable).sum()),
    }
    smallest_eval = min(report["evaluable_distribution"].values())
    report["power_eval_full"] = minority_power(max(1, smallest_eval))
    report["power_eval_per_fold"] = minority_power(max(1, smallest_eval // 5))
    return merged, report


def sampling_manifest(
    pool: pd.DataFrame,
    candidates: pd.DataFrame,
    plan: SamplingPlan,
    *,
    excluded: int,
) -> dict:
    """Record the design so the report can describe the draw exactly."""
    return {
        "plan": {
            "quotas": dict(plan.quotas),
            "seed": plan.seed,
            "body_preview_chars": plan.body_preview_chars,
        },
        "pool": {
            "candidates_after_dedup": int(len(pool)),
            "excluded_already_labelled": int(excluded),
            "with_model_prior": int(pool["prob_negative"].notna().sum()),
            "negative_cue_rows": int((pool["negative_cues"] > 0).sum()),
            "positive_cue_rows": int((pool["positive_cues"] > 0).sum()),
        },
        "strata": candidates.attrs.get("design", {}),
        "drawn_total": int(len(candidates)),
        "lexicon_sizes": {
            "negative_cues": len(NEGATIVE_CUES),
            "positive_cues": len(POSITIVE_CUES),
        },
        "notes": [
            "random_eval preserves the corpus label prior and is the only stratum "
            "usable for an unbiased overall estimate.",
            "Enrichment strata oversample minority classes and must be used for "
            "training, or reported with per-stratum weights.",
            "Lexicon hits and model probabilities select which articles a human "
            "reads; they never assign a label.",
        ],
    }


def minority_power(n_minority: int, *, recall: float = 0.5) -> dict[str, float]:
    """Half-width of the 95% Wald interval for a minority-class recall estimate.

    Answers the question a committee actually asks: with this many samples of the
    rare class in the evaluation set, how precisely is its recall known?
    """
    if n_minority < 1:
        raise ValueError("n_minority must be >= 1.")
    half = 1.96 * float(np.sqrt(recall * (1.0 - recall) / n_minority))
    return {
        "n_minority": int(n_minority),
        "assumed_recall": recall,
        "ci_half_width": round(half, 4),
        "ci_low": round(max(0.0, recall - half), 4),
        "ci_high": round(min(1.0, recall + half), 4),
    }


def audit_labels(frame: pd.DataFrame, *, valid_labels: tuple[str, ...]) -> dict:
    """Check a filled annotation file and report the distribution and power."""
    issues: list[str] = []
    if "label" not in frame.columns:
        raise ValueError("Annotation file has no 'label' column.")

    labelled = frame[frame["label"].notna() & (frame["label"].astype(str).str.strip() != "")]
    bad = sorted(set(labelled["label"].astype(str)) - set(valid_labels))
    if bad:
        issues.append(f"invalid label values: {bad}")
    if "url" in frame.columns and frame["url"].duplicated().any():
        issues.append(f"{int(frame['url'].duplicated().sum())} duplicated urls")
    if "title" in frame.columns:
        dupes = int(labelled["title"].astype(str).str.strip().str.lower().duplicated().sum())
        if dupes:
            issues.append(f"{dupes} duplicated titles among labelled rows")

    counts = labelled["label"].astype(str).value_counts()
    distribution = {label: int(counts.get(label, 0)) for label in valid_labels}
    by_stratum = {}
    if "stratum" in labelled.columns:
        for stratum, block in labelled.groupby("stratum", sort=True):
            block_counts = block["label"].astype(str).value_counts()
            by_stratum[str(stratum)] = {
                label: int(block_counts.get(label, 0)) for label in valid_labels
            }

    smallest = min(distribution.values()) if distribution else 0
    return {
        "rows": int(len(frame)),
        "labelled": int(len(labelled)),
        "pending": int(len(frame) - len(labelled)),
        "distribution": distribution,
        "by_stratum": by_stratum,
        "smallest_class": smallest,
        "power_5fold": minority_power(max(1, smallest // 5)),
        "power_full": minority_power(max(1, smallest)),
        "issues": issues,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
