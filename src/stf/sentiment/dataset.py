"""Load sentiment-labeled data and prepare it for PhoBERT fine-tuning.

Label sources:
  1. Public seed: the CafeF headline corpus (3 classes) for a quick baseline.
  2. In-domain add-on: model prelabels reviewed by one human under a fixed guideline.

Expected label CSV/parquet format: a `text` column (str) and a `label` column

PhoBERT ideally takes word-segmented input (VnCoreNLP). Here we tokenize raw text for
simplicity/reproducibility; to improve quality, add a segmentation step before the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stf import config
from stf.sentiment.labels import LABEL2ID

INPUT_VARIANTS: tuple[str, ...] = ("title", "context", "title_context")


def build_input_text(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Create the selected model input from title/context columns.

    Label files from CafeF contain only ``text`` (headline), while in-domain
    files may retain separate ``title`` and ``body`` columns. Missing columns
    therefore fall back to ``text`` so both sources use one code path.
    """
    if variant not in INPUT_VARIANTS:
        raise ValueError(
            f"Unknown input variant {variant!r}; expected one of {INPUT_VARIANTS}."
        )

    out = df.copy()
    empty = pd.Series("", index=out.index, dtype="string")
    text = (
        out["text"].fillna("").astype("string").str.strip()
        if "text" in out.columns
        else empty
    )
    has_title = "title" in out.columns
    has_context = "body" in out.columns or "body_preview" in out.columns
    title = (
        out["title"].fillna("").astype("string").str.strip()
        if has_title
        else text
    )
    if "body" in out.columns:
        context = out["body"].fillna("").astype("string").str.strip()
    elif "body_preview" in out.columns:
        context = out["body_preview"].fillna("").astype("string").str.strip()
    else:
        context = text

    if variant == "title":
        selected = title.mask(title.eq(""), text)
    elif variant == "context":
        selected = context
    elif not has_title and not has_context:
        selected = text
    elif not has_title:
        selected = context
    elif not has_context:
        selected = title
    else:
        selected = title.where(
            context.eq(""),
            title + "\n\n" + context,
        )
        selected = selected.mask(title.eq(""), context)
        selected = selected.mask(selected.eq(""), text)


    out["text"] = selected.astype("string").str.strip()
    out = out[out["text"].notna() & out["text"].ne("")].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Input variant {variant!r} produced no non-empty text.")
    return out


@dataclass
class Split:
    """One fixed train/val/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def dist(df: pd.DataFrame) -> str:
            vc = df["label_id"].value_counts().sort_index().to_dict()
            return ", ".join(f"{k}:{v}" for k, v in vc.items())

        return (
            f"train={len(self.train)} ({dist(self.train)}) | "
            f"val={len(self.val)} ({dist(self.val)}) | "
            f"test={len(self.test)} ({dist(self.test)})"
        )


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels to integer ids and reject malformed rows."""
    df = df.copy()
    source = df["label_id"] if "label_id" in df.columns else df["label"]

    if pd.api.types.is_string_dtype(source) or source.dtype == object:
        text = source.astype("string").str.upper().str.strip()
        values = text.map(LABEL2ID)
        numeric = pd.to_numeric(text, errors="coerce")
        values = values.fillna(numeric)
    else:
        values = pd.to_numeric(source, errors="coerce")

    invalid = values.isna() | (values % 1 != 0) | ~values.between(0, 2)
    if invalid.any():
        bad = source[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"Invalid labels: {bad}. Expected NEGATIVE/NEUTRAL/POSITIVE or 0/1/2."
        )

    df["label_id"] = values.astype("int64")
    return df


def deduplicate_labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated source items before any split or input variant is built.

    URL is the strongest identity when available. Otherwise the normalized source
    text columns form a stable exact-match key shared by title/context variants.
    """
    out = df.copy()
    if "url" in out.columns:
        key = out["url"].astype("string").str.strip()
    elif "text" in out.columns:
        key = out["text"].astype("string").str.strip()
    else:
        key_parts = []
        for column in ("title", "body", "body_preview"):
            if column in out.columns:
                key_parts.append(out[column].fillna("").astype("string").str.strip())
        if not key_parts:
            return out.reset_index(drop=True)
        key = key_parts[0]
        for part in key_parts[1:]:
            key = key + "\n" + part
    keep = key.notna() & key.ne("") & ~key.duplicated()
    return out.loc[keep].reset_index(drop=True)


def load_labeled(path: str | Path) -> pd.DataFrame:
    """Load, validate, normalize, and deduplicate a label file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Label file must be .csv or .parquet.")
    if "label" not in df.columns:
        raise ValueError("Label file needs a 'label' column.")
    if not (
        {"text"} <= set(df.columns)
        or {"title", "body"} <= set(df.columns)
        or {"title", "body_preview"} <= set(df.columns)
    ):
        raise ValueError(
            "Label file needs 'text' or title plus body/body_preview columns."
        )
    if "text" in df.columns:
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype("string").str.strip()
        df = df[df["text"].ne("")].reset_index(drop=True)
    return deduplicate_labeled(normalize_labels(df))


def resolve_time_column(df: pd.DataFrame) -> pd.Series:
    """Return timezone-aware timestamps to drive a time-aware split."""
    if "date" in df.columns:
        source, name = df["date"], "date"
    elif "published_at" in df.columns:
        source, name = df["published_at"], "published_at"
    else:
        raise ValueError(
            "Time-aware split requires a 'date' or 'published_at' column; "
            "pass time_aware=False to request a random split explicitly."
        )
    dates = pd.to_datetime(source, errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError(
            f"Time-aware split requires a valid timestamp in every '{name}' row."
        )
    return dates.dt.tz_convert(config.TIMEZONE)
def make_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    time_aware: bool = True,
) -> Split:
    """Split into train, validation and test without duplicate or date leakage."""
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < val_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("val_frac and test_frac must be between 0 and 1.")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must leave training rows.")
    if "label_id" not in df.columns:
        raise ValueError("Dataset needs a normalized 'label_id' column.")

    df = deduplicate_labeled(df.reset_index(drop=True))
    if time_aware:
        dates = resolve_time_column(df).dt.normalize()
        df = df.assign(date=dates)
        unique_dates = np.sort(dates.drop_duplicates().to_numpy())
        n_dates = len(unique_dates)
        n_test = max(1, round(n_dates * test_frac))
        n_val = max(1, round(n_dates * val_frac))
        if n_dates - n_val - n_test < 1:
            raise ValueError("Dataset is too small for train, validation and test dates.")
        train_dates = set(unique_dates[: n_dates - n_val - n_test])
        val_dates = set(unique_dates[n_dates - n_val - n_test : n_dates - n_test])
        test_dates = set(unique_dates[n_dates - n_test :])
        return Split(
            df.loc[dates.isin(train_dates)].reset_index(drop=True),
            df.loc[dates.isin(val_dates)].reset_index(drop=True),
            df.loc[dates.isin(test_dates)].reset_index(drop=True),
        )

    n = len(df)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n - n_val - n_test < 1:
        raise ValueError("Dataset is too small for train, validation and test splits.")
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("label_id", sort=True):
        group = group.iloc[rng.permutation(len(group))].reset_index(drop=True)
        n_group_test = min(max(1, round(len(group) * test_frac)), len(group))
        n_group_val = min(
            max(1, round(len(group) * val_frac)),
            max(0, len(group) - n_group_test - 1),
        )
        parts["test"].append(group.iloc[:n_group_test])
        parts["val"].append(group.iloc[n_group_test : n_group_test + n_group_val])
        parts["train"].append(group.iloc[n_group_test + n_group_val :])
    result = {}
    for name, frames in parts.items():
        result[name] = (
            pd.concat(frames)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
    if any(result[name].empty for name in ("train", "val", "test")):
        raise ValueError("Stratified splitting produced an empty partition.")
    return Split(result["train"], result["val"], result["test"])
