"""Walk-forward forecasting experiment over the real point-in-time panel.

This is the only module that turns the forecasting primitives into reportable
numbers. It owns every leakage-sensitive decision in one place so the guarantees
are enforced rather than left to a caller convention:

* trend thresholds are fitted on each window's training returns only;
* feature scalers are fitted on each window's training rows only;
* sequence windows never cross tickers and never read past the observation date;
* validation selects the LSTM checkpoint, the test block is scored once;
* every model in a window is scored on exactly the same test rows.

The model ladder is majority / random / classical logistic regression / LSTM,
each in a price-only and a price-plus-sentiment arm where sentiment applies. The
sentiment contribution is reported as a paired per-window delta with a bootstrap
confidence interval, and additionally split by whether the target row had news,
because sentiment can only act on the sessions that carry news.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stf.forecasting.features import FeatureScaler, price_feature_columns
from stf.forecasting.labels import TREND_LABELS, fit_thresholds, label_panel
from stf.forecasting.sentiment_agg import ROLLING_SENTIMENT_COLUMNS, SENTIMENT_COLUMNS
from stf.forecasting.split import TimeSplit, walk_forward_windows

# Sentiment features handed to the models. ``has_news`` and the trailing columns are
# included so a single-row learner can still see recent news on a quiet session.
SENTIMENT_FEATURES: tuple[str, ...] = (*SENTIMENT_COLUMNS, *ROLLING_SENTIMENT_COLUMNS)

# Arms: (name, model family, uses sentiment features).
LADDER: tuple[tuple[str, str, bool], ...] = (
    ("majority", "majority", False),
    ("random", "random", False),
    ("logreg_price", "classical", False),
    ("logreg_price_sentiment", "classical", True),
    ("lstm_price", "lstm", False),
    ("lstm_price_sentiment", "lstm", True),
)

# Arms compared head-to-head to quantify the sentiment contribution.
ABLATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("logreg_price_sentiment", "logreg_price"),
    ("lstm_price_sentiment", "lstm_price"),
)

DELTA_METRICS: tuple[str, ...] = ("macro_f1", "balanced_accuracy", "accuracy")


@dataclass
class ForecastConfig:
    """Everything that changes a number, in one hashable place."""

    seq_len: int = 5
    n_windows: int = 5
    test_size: int = 60
    val_size: int = 60
    expanding: bool = True
    hidden: int = 32
    num_layers: int = 1
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 5
    seeds: tuple[int, ...] = (42, 43, 44)
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 7

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["seeds"] = list(self.seeds)
        return data


@dataclass
class WindowArtifacts:
    """Per-window record kept for the report and for error analysis."""

    window: int
    thresholds: tuple[float, float]
    train_dates: tuple[str, str]
    val_dates: tuple[str, str]
    test_dates: tuple[str, str]
    sizes: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    metrics: dict[str, dict] = field(default_factory=dict)


def _date_bounds(frame: pd.DataFrame, col: str = "target_date") -> tuple[str, str]:
    dates = pd.to_datetime(frame[col]).dropna()
    if dates.empty:
        return ("", "")
    return (str(dates.min().date()), str(dates.max().date()))


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target_label"].value_counts()
    return {label: int(counts.get(label, 0)) for label in TREND_LABELS}


def _mean_std(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.size),
    }


def _bootstrap_ci(
    deltas: list[float], *, samples: int, seed: int, alpha: float = 0.05
) -> dict[str, float | None]:
    """Percentile bootstrap over per-window deltas.

    Windows are the resampling unit because observations inside a window are
    serially dependent; resampling rows would understate the interval.
    """
    arr = np.asarray([d for d in deltas if d is not None], dtype=float)
    if arr.size < 2:
        return {"mean": float(arr.mean()) if arr.size else None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(samples, arr.size), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "low": float(np.quantile(draws, alpha / 2)),
        "high": float(np.quantile(draws, 1 - alpha / 2)),
    }


def _metrics_from_confusion(counts: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized macro-F1 / balanced accuracy / accuracy from confusion matrices.

    ``counts`` has shape ``(..., n_classes, n_classes)`` with true labels on rows.
    Working from counts lets the bootstrap resample whole dates by summing their
    matrices instead of rescoring predictions on every draw.
    """
    counts = np.asarray(counts, dtype=float)
    tp = np.einsum("...ii->...i", counts)
    true_total = counts.sum(axis=-1)
    pred_total = counts.sum(axis=-2)
    denom = true_total + pred_total
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    recall = np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0)
    total = counts.sum(axis=(-2, -1))
    accuracy = np.divide(
        tp.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0
    )
    # Balanced accuracy averages recall only over classes present in the block.
    present = true_total > 0
    n_present = present.sum(axis=-1)
    balanced = np.divide(
        (recall * present).sum(axis=-1),
        n_present,
        out=np.zeros_like(total),
        where=n_present > 0,
    )
    return {
        "macro_f1": f1.mean(axis=-1),
        "balanced_accuracy": balanced,
        "accuracy": accuracy,
    }


def _date_confusions(block: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    """Per-date confusion matrices for one arm/seed, aligned to ``dates``."""
    n_classes = len(TREND_LABELS)
    mats = np.zeros((len(dates), n_classes, n_classes), dtype=float)
    index = {date: i for i, date in enumerate(dates)}
    rows = block["_date"].map(index).to_numpy()
    np.add.at(mats, (rows, block["_true"].to_numpy(), block["_pred"].to_numpy()), 1.0)
    return mats


def _date_block_bootstrap(
    predictions: pd.DataFrame,
    treated: str,
    control: str,
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | None]]:
    """Bootstrap the paired metric delta by resampling whole test dates.

    Test dates are the resampling unit: every ticker sharing a date moves
    together, which preserves the cross-sectional dependence inside a session
    while giving ~300 blocks instead of the 5 blocks a per-window bootstrap has.
    Seeds are averaged inside each draw so the interval reflects date sampling,
    not initialization noise.
    """
    wanted = predictions[predictions["arm"].isin((treated, control))]
    if wanted.empty:
        return {metric: {"mean": None, "low": None, "high": None} for metric in DELTA_METRICS}

    frame = wanted.copy()
    frame["_date"] = pd.to_datetime(frame["target_date"]).to_numpy("datetime64[ns]")
    label_ids = {label: i for i, label in enumerate(TREND_LABELS)}
    frame["_true"] = frame["y_true"].map(label_ids).to_numpy()
    frame["_pred"] = frame["y_pred"].map(label_ids).to_numpy()
    dates = np.sort(frame["_date"].unique())

    # (arm, seed) -> per-date confusion matrices, flattened for a matmul draw.
    seeds = sorted(frame["seed"].unique())
    stacks: dict[str, list[np.ndarray]] = {treated: [], control: []}
    for arm in (treated, control):
        for seed_value in seeds:
            block = frame[(frame["arm"] == arm) & (frame["seed"] == seed_value)]
            stacks[arm].append(_date_confusions(block, dates).reshape(len(dates), -1))

    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(dates), np.full(len(dates), 1.0 / len(dates)), size=samples)
    weights = weights.astype(float)

    def draw_metrics(arm: str) -> dict[str, np.ndarray]:
        per_seed = []
        for flat in stacks[arm]:
            counts = (weights @ flat).reshape(samples, len(TREND_LABELS), len(TREND_LABELS))
            per_seed.append(_metrics_from_confusion(counts))
        return {
            metric: np.mean([m[metric] for m in per_seed], axis=0) for metric in DELTA_METRICS
        }

    treated_draws = draw_metrics(treated)
    control_draws = draw_metrics(control)

    full = np.ones((1, len(dates)), dtype=float)
    point = {}
    for arm in (treated, control):
        per_seed = []
        for flat in stacks[arm]:
            counts = (full @ flat).reshape(1, len(TREND_LABELS), len(TREND_LABELS))
            per_seed.append(_metrics_from_confusion(counts))
        point[arm] = {
            metric: float(np.mean([m[metric][0] for m in per_seed])) for metric in DELTA_METRICS
        }

    out = {}
    for metric in DELTA_METRICS:
        deltas = treated_draws[metric] - control_draws[metric]
        out[metric] = {
            "mean": point[treated][metric] - point[control][metric],
            "low": float(np.quantile(deltas, alpha / 2)),
            "high": float(np.quantile(deltas, 1 - alpha / 2)),
            "n_blocks": int(len(dates)),
        }
    return out


def _fit_arm(
    family: str,
    use_sentiment: bool,
    *,
    seed: int,
    cfg: ForecastConfig,
    train: dict,
    val: dict,
    test: dict,
    n_price: int,
    n_sent: int,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Train one ladder arm and return ``(test_pred, test_proba, train_info)``."""
    from stf.forecasting.models import (
        ClassicalBaseline,
        MajorityBaseline,
        PriceLSTM,
        PriceSentimentLSTM,
        RandomBaseline,
        fit_lstm,
        predict_lstm,
        set_seed,
    )

    n_test = len(test["y"])
    if family == "majority":
        model = MajorityBaseline().fit(train["y"])
        return model.predict(n_test), None, {}
    if family == "random":
        model = RandomBaseline(seed=seed).fit(train["y"])
        return model.predict(n_test), None, {}
    if family == "classical":
        key = "X_both" if use_sentiment else "X_price"
        model = ClassicalBaseline(seed=seed).fit(train[key], train["y"])
        proba = model.predict_proba(test[key])
        return proba.argmax(axis=1), proba, {}
    if family != "lstm":
        raise ValueError(f"Unknown model family {family!r}.")

    # Seed before constructing the net: weight initialisation draws from the global
    # torch RNG, so without this the initial weights depend on how many random
    # numbers earlier arms and earlier windows happened to consume.
    set_seed(seed)

    if use_sentiment:
        net = PriceSentimentLSTM(
            n_price,
            n_sent,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            num_classes=len(TREND_LABELS),
        )
        info = fit_lstm(
            net,
            train["X_price"],
            train["y"],
            X_sent=train["X_sent"],
            X_val=val["X_price"],
            y_val=val["y"],
            X_sent_val=val["X_sent"],
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            patience=cfg.patience,
            seed=seed,
        )
        proba = predict_lstm(net, test["X_price"], test["X_sent"])
    else:
        net = PriceLSTM(
            n_price,
            hidden=cfg.hidden,
            num_layers=cfg.num_layers,
            num_classes=len(TREND_LABELS),
        )
        info = fit_lstm(
            net,
            train["X_price"],
            train["y"],
            X_val=val["X_price"],
            y_val=val["y"],
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            patience=cfg.patience,
            seed=seed,
        )
        proba = predict_lstm(net, test["X_price"])
    return proba.argmax(axis=1), proba, info


def _partition(
    X_price: np.ndarray,
    X_sent: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    dates: set,
) -> dict:
    mask = meta["target_date"].isin(dates).to_numpy()
    both = np.concatenate([X_price[mask], X_sent[mask]], axis=2) if mask.any() else X_price[mask]
    return {
        "X_price": X_price[mask],
        "X_sent": X_sent[mask],
        "X_both": both,
        "y": y[mask],
        "meta": meta.loc[mask].reset_index(drop=True),
        "mask": mask,
    }


def run_experiment(
    panel: pd.DataFrame,
    *,
    cfg: ForecastConfig | None = None,
    output_dir: Path | None = None,
    provenance: dict | None = None,
) -> dict:
    """Run the full walk-forward ladder and return the aggregated result record."""
    from stf.forecasting.models import evaluate_predictions, make_two_branch_sequences

    cfg = cfg or ForecastConfig()
    if panel.empty:
        raise ValueError("Cannot run a forecasting experiment on an empty panel.")

    price_cols = price_feature_columns()
    sent_cols = list(SENTIMENT_FEATURES)
    windows = walk_forward_windows(
        panel,
        n_windows=cfg.n_windows,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        expanding=cfg.expanding,
    )

    per_window: list[WindowArtifacts] = []
    # arm -> metric -> per-(window, seed) values
    collected: dict[str, dict[str, list[float]]] = {
        name: {m: [] for m in DELTA_METRICS} for name, _, _ in LADDER
    }
    collected_auc: dict[str, list[float]] = {name: [] for name, _, _ in LADDER}
    # (arm, window) -> metric -> seed-averaged value, for paired deltas.
    paired: dict[tuple[str, int], dict[str, float]] = {}
    strat_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for w_index, split in enumerate(windows, start=1):
        train_rows = panel.iloc[split.train]
        thresholds = fit_thresholds(train_rows["target_return"].dropna().to_numpy())
        labeled = label_panel(panel, thresholds)

        price_scaler = FeatureScaler.fit(labeled.iloc[split.train], price_cols)
        sent_scaler = FeatureScaler.fit(labeled.iloc[split.train], sent_cols)
        scaled = sent_scaler.transform(price_scaler.transform(labeled))

        X_price, X_sent, y, meta = make_two_branch_sequences(
            scaled, price_cols, sent_cols, seq_len=cfg.seq_len
        )
        if len(y) == 0:
            raise RuntimeError(f"Window {w_index} produced no usable sequences.")

        date_sets = {
            part: set(pd.to_datetime(panel.iloc[idx]["target_date"]).dropna())
            for part, idx in (("train", split.train), ("val", split.val), ("test", split.test))
        }
        parts = {
            part: _partition(X_price, X_sent, y, meta, dates)
            for part, dates in date_sets.items()
        }
        for part in ("train", "val", "test"):
            if len(parts[part]["y"]) == 0:
                raise RuntimeError(f"Window {w_index} has no {part} sequences.")

        # has_news of the observation row behind each test sequence, for stratification.
        test_meta = parts["test"]["meta"]
        news_flag = (
            labeled[["ticker", "observation_date", "has_news"]]
            .merge(test_meta, on=["ticker", "observation_date"], how="right")["has_news"]
            .fillna(0)
            .to_numpy()
            .astype(int)
        )

        artifacts = WindowArtifacts(
            window=w_index,
            thresholds=(float(thresholds[0]), float(thresholds[1])),
            train_dates=_date_bounds(panel.iloc[split.train]),
            val_dates=_date_bounds(panel.iloc[split.val]),
            test_dates=_date_bounds(panel.iloc[split.test]),
            sizes={part: int(len(parts[part]["y"])) for part in ("train", "val", "test")},
            class_distribution={
                part: _label_counts(labeled.iloc[idx])
                for part, idx in (
                    ("train", split.train),
                    ("val", split.val),
                    ("test", split.test),
                )
            },
        )

        for name, family, use_sentiment in LADDER:
            seed_metrics: list[dict] = []
            for seed in cfg.seeds:
                pred, proba, info = _fit_arm(
                    family,
                    use_sentiment,
                    seed=seed,
                    cfg=cfg,
                    train=parts["train"],
                    val=parts["val"],
                    test=parts["test"],
                    n_price=len(price_cols),
                    n_sent=len(sent_cols),
                )
                scored = evaluate_predictions(parts["test"]["y"], pred, proba)
                scored["seed"] = seed
                if info:
                    scored["best_epoch"] = info["best_epoch"]
                    scored["best_val_macro_f1"] = info["best_val_macro_f1"]
                seed_metrics.append(scored)
                for metric in DELTA_METRICS:
                    collected[name][metric].append(scored[metric])
                if scored["macro_ovr_auc"] is not None:
                    collected_auc[name].append(scored["macro_ovr_auc"])

                for label, keep in (("has_news", news_flag == 1), ("no_news", news_flag == 0)):
                    if not keep.any():
                        continue
                    sub = evaluate_predictions(
                        parts["test"]["y"][keep],
                        np.asarray(pred)[keep],
                        None if proba is None else proba[keep],
                    )
                    strat_rows.append(
                        {
                            "window": w_index,
                            "seed": seed,
                            "arm": name,
                            "stratum": label,
                            "n": sub["n"],
                            "macro_f1": sub["macro_f1"],
                            "balanced_accuracy": sub["balanced_accuracy"],
                            "accuracy": sub["accuracy"],
                        }
                    )

                prediction_rows.append(
                    test_meta.assign(
                        window=w_index,
                        seed=seed,
                        arm=name,
                        y_true=[TREND_LABELS[i] for i in parts["test"]["y"]],
                        y_pred=[TREND_LABELS[int(i)] for i in pred],
                        has_news=news_flag,
                    )
                )

            # Seed average is the window's value; paired deltas use these.
            window_value = {
                metric: float(np.mean([m[metric] for m in seed_metrics]))
                for metric in DELTA_METRICS
            }
            paired[(name, w_index)] = window_value
            artifacts.metrics[name] = {
                "seed_runs": seed_metrics,
                "window_mean": window_value,
            }

        per_window.append(artifacts)

    summary = {}
    for name, _, _ in LADDER:
        entry = {metric: _mean_std(collected[name][metric]) for metric in DELTA_METRICS}
        entry["macro_ovr_auc"] = _mean_std(collected_auc[name])
        summary[name] = entry

    predictions = (
        pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    )

    ablation = {}
    for treated, control in ABLATION_PAIRS:
        deltas = {metric: [] for metric in DELTA_METRICS}
        for w_index in range(1, len(windows) + 1):
            for metric in DELTA_METRICS:
                deltas[metric].append(
                    paired[(treated, w_index)][metric] - paired[(control, w_index)][metric]
                )
        by_date = (
            {}
            if predictions.empty
            else _date_block_bootstrap(
                predictions,
                treated,
                control,
                samples=cfg.bootstrap_samples,
                seed=cfg.bootstrap_seed,
            )
        )
        ablation[f"{treated}__minus__{control}"] = {
            metric: {
                "per_window": deltas[metric],
                "window_bootstrap": _bootstrap_ci(
                    deltas[metric],
                    samples=cfg.bootstrap_samples,
                    seed=cfg.bootstrap_seed,
                ),
                "date_block_bootstrap": by_date.get(metric, {}),
            }
            for metric in DELTA_METRICS
        }

    strat = pd.DataFrame(strat_rows)
    stratified = {}
    if not strat.empty:
        grouped = strat.groupby(["arm", "stratum"], sort=True)
        for (arm, stratum), block in grouped:
            stratified.setdefault(arm, {})[stratum] = {
                "n_mean": float(block["n"].mean()),
                "macro_f1_mean": float(block["macro_f1"].mean()),
                "balanced_accuracy_mean": float(block["balanced_accuracy"].mean()),
                "accuracy_mean": float(block["accuracy"].mean()),
            }

    record = {
        "config": cfg.as_dict(),
        "price_features": price_cols,
        "sentiment_features": sent_cols,
        "class_labels": list(TREND_LABELS),
        "chance_level": round(1.0 / len(TREND_LABELS), 6),
        "panel_rows": int(len(panel)),
        "panel_tickers": sorted(panel["ticker"].unique().tolist()),
        "windows": [
            {
                "window": a.window,
                "thresholds": list(a.thresholds),
                "train_dates": list(a.train_dates),
                "val_dates": list(a.val_dates),
                "test_dates": list(a.test_dates),
                "sizes": a.sizes,
                "class_distribution": a.class_distribution,
                "metrics": a.metrics,
            }
            for a in per_window
        ],
        "summary": summary,
        "ablation": ablation,
        "stratified_by_news": stratified,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "provenance": provenance or {},
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "forecast_results.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        _summary_table(summary, ablation).to_csv(
            output_dir / "forecast_metrics.csv", index=False
        )
        if not strat.empty:
            strat.to_csv(output_dir / "forecast_stratified.csv", index=False)
        if not predictions.empty:
            predictions.to_csv(output_dir / "forecast_predictions.csv", index=False)
    return record


def _summary_table(summary: dict, ablation: dict) -> pd.DataFrame:
    """Flatten the ladder summary into the table shape the report needs."""
    rows = []
    for arm, metrics in summary.items():
        row = {"arm": arm}
        for metric in (*DELTA_METRICS, "macro_ovr_auc"):
            row[f"{metric}_mean"] = metrics[metric]["mean"]
            row[f"{metric}_std"] = metrics[metric]["std"]
        rows.append(row)
    for pair, metrics in ablation.items():
        row = {"arm": f"DELTA {pair}"}
        for metric in DELTA_METRICS:
            window_ci = metrics[metric]["window_bootstrap"]
            date_ci = metrics[metric]["date_block_bootstrap"]
            row[f"{metric}_mean"] = window_ci["mean"]
            row[f"{metric}_std"] = None
            row[f"{metric}_window_ci_low"] = window_ci["low"]
            row[f"{metric}_window_ci_high"] = window_ci["high"]
            row[f"{metric}_date_ci_low"] = date_ci.get("low")
            row[f"{metric}_date_ci_high"] = date_ci.get("high")
        rows.append(row)
    return pd.DataFrame(rows)




def _paired_arm_predictions(
    real_dir: Path,
    control_dir: Path,
    *,
    arm: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one arm from paired runs and require identical test observations."""
    key_columns = ["window", "seed", "ticker", "target_date", "y_true"]
    real_preds = pd.read_csv(Path(real_dir) / "forecast_predictions.csv")
    control_preds = pd.read_csv(Path(control_dir) / "forecast_predictions.csv")
    required = set(key_columns + ["arm"])
    for name, predictions in (("real", real_preds), ("control", control_preds)):
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(
                f"{name} predictions lack required pairing columns: {sorted(missing)}."
            )

    real_arm = real_preds[real_preds["arm"] == arm].copy()
    control_arm = control_preds[control_preds["arm"] == arm].copy()
    for name, predictions in (("real", real_arm), ("control", control_arm)):
        if predictions.empty:
            raise ValueError(f"{name} predictions contain no rows for arm {arm!r}.")
        if predictions.duplicated(key_columns).any():
            raise ValueError(f"{name} predictions duplicate a pairing key.")

    real_keys = real_arm[key_columns].sort_values(key_columns).reset_index(drop=True)
    control_keys = control_arm[key_columns].sort_values(key_columns).reset_index(drop=True)
    if not real_keys.equals(control_keys):
        raise ValueError(
            "Runs have different prediction keys; the information gain is not attributable."
        )
    return real_arm, control_arm


def compare_information_gain(
    real_dir: Path,
    control_dir: Path,
    *,
    arm: str = "lstm_price_sentiment",
    price_arm: str = "lstm_price",
    samples: int = 2000,
    seed: int = 7,
) -> dict:
    """Split a two-branch arm's apparent gain into architecture and information.

    ``control_dir`` must come from a run with no sentiment parquet, where the
    sentiment branch receives the constant neutral prior. Comparing the two-branch
    arm against the single-branch price model conflates two changes: the extra
    branch and parameters, and the information the branch carries. The control run
    holds the architecture fixed and removes only the information, so

        architecture effect = control(two-branch) - control(price-only)
        information gain    = real(two-branch)    - control(two-branch)

    The runs must share their configuration, otherwise the difference is not
    attributable and this raises.
    """
    real = json.loads((Path(real_dir) / "forecast_results.json").read_text(encoding="utf-8"))
    control = json.loads(
        (Path(control_dir) / "forecast_results.json").read_text(encoding="utf-8")
    )
    if real["config"] != control["config"]:
        raise ValueError("Runs have different configs; the difference is not attributable.")
    if control["provenance"].get("news_sentiment") is not None:
        raise ValueError("control_dir must be a run without --news-sentiment.")
    real_price_hash = real.get("provenance", {}).get("prices_hash")
    control_price_hash = control.get("provenance", {}).get("prices_hash")
    if not real_price_hash or not control_price_hash:
        raise ValueError("Runs lack prices_hash provenance; the difference is not attributable.")
    if real_price_hash != control_price_hash:
        raise ValueError("Runs have different price provenance; the difference is not attributable.")
    real_test_dates = {window["window"]: window["test_dates"] for window in real["windows"]}
    control_test_dates = {
        window["window"]: window["test_dates"] for window in control["windows"]
    }
    if (
        len(real_test_dates) != len(real["windows"])
        or len(control_test_dates) != len(control["windows"])
        or real_test_dates != control_test_dates
    ):
        raise ValueError("Runs have different test windows; the difference is not attributable.")
    real_arm, control_arm = _paired_arm_predictions(real_dir, control_dir, arm=arm)

    levels = {
        "price_only": real["summary"][price_arm],
        "two_branch_neutral_prior": control["summary"][arm],
        "two_branch_real_sentiment": real["summary"][arm],
    }
    effects = {}
    for metric in DELTA_METRICS:
        price = levels["price_only"][metric]["mean"]
        neutral = levels["two_branch_neutral_prior"][metric]["mean"]
        actual = levels["two_branch_real_sentiment"][metric]["mean"]
        per_window = [
            real["windows"][index]["metrics"][arm]["window_mean"][metric]
            - control["windows"][index]["metrics"][arm]["window_mean"][metric]
            for index in range(len(real["windows"]))
        ]
        effects[metric] = {
            "price_only": price,
            "two_branch_neutral_prior": neutral,
            "two_branch_real_sentiment": actual,
            "architecture_effect": neutral - price,
            "information_gain": actual - neutral,
            "naive_delta": actual - price,
            "information_gain_per_window": per_window,
            "information_gain_window_bootstrap": _bootstrap_ci(
                per_window, samples=samples, seed=seed
            ),
        }

    paired = pd.concat(
        [
            real_arm.assign(arm="real"),
            control_arm.assign(arm="neutral"),
        ],
        ignore_index=True,
    )
    by_date = _date_block_bootstrap(paired, "real", "neutral", samples=samples, seed=seed)
    for metric in DELTA_METRICS:
        effects[metric]["information_gain_date_bootstrap"] = by_date[metric]

    return {
        "arm": arm,
        "price_arm": price_arm,
        "config": real["config"],
        "effects": effects,
    }


def frame_hash(frame: pd.DataFrame) -> str:
    """Stable content hash of a frame, for the reproducibility manifest."""
    digest = hashlib.sha256()
    for col in sorted(frame.columns):
        digest.update(col.encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame[col], index=False).values.tobytes())
    return digest.hexdigest()
