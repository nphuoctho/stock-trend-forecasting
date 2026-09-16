"""Reproducible PhoBERT ablations and stratified cross-validation."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from stf.sentiment import dataset, model
from stf.sentiment.dataset import INPUT_VARIANTS, Split

TRUNCATION_STRATEGIES = model.TRUNCATION_STRATEGIES


def make_stratified_folds(
    df: pd.DataFrame, *, n_splits: int = 5, seed: int = 42
) -> list[tuple[list[int], list[int]]]:
    """Return disjoint train/holdout indices for reproducible stratified folds."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if "label_id" not in df.columns:
        raise ValueError("Dataset needs a normalized 'label_id' column.")
    counts = df["label_id"].value_counts()
    if len(counts) < 3 or counts.min() < n_splits:
        raise ValueError(
            f"Each of the three classes needs at least {n_splits} samples for CV."
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train_idx.tolist(), holdout_idx.tolist())
        for train_idx, holdout_idx in splitter.split(df, df["label_id"])
    ]


def _prepare_frame(
    df: pd.DataFrame, input_variant: str, *, allow_preliminary: bool = False
) -> pd.DataFrame:
    frame = dataset.reject_preliminary_labels(
        dataset.normalize_labels(df), allow_preliminary=allow_preliminary
    )
    return dataset.prepare_model_input(frame, input_variant)


def _outer_split(
    frame: pd.DataFrame, train_idx: list[int], holdout_idx: list[int], seed: int
) -> Split:
    """Create inner train/validation data and keep the outer holdout untouched."""
    outer_train = frame.iloc[train_idx].reset_index(drop=True)
    outer_test = frame.iloc[holdout_idx].reset_index(drop=True)
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    train_rows, val_rows = next(inner.split(outer_train, outer_train["label_id"]))
    return Split(
        outer_train.iloc[train_rows].reset_index(drop=True),
        outer_train.iloc[val_rows].reset_index(drop=True),
        outer_test,
    )


def run_cross_validation(
    df: pd.DataFrame,
    *,
    input_variant: str = "title",
    truncation_strategy: str = "head",
    cfg: model.TrainConfig | None = None,
    folds: int = 5,
    seed: int = 42,
    out_dir: Path,
    allow_preliminary: bool = False,
    source_path: str | Path | None = None,
) -> dict:
    """Train/evaluate one input configuration with outer stratified K-fold CV.

    The full labeled frame is used exactly once as an outer holdout per fold.
    A separate 10% split inside each training fold selects the best checkpoint,
    so the outer metric remains unseen during model selection. Unreviewed
    preliminary rows are rejected unless ``allow_preliminary=True`` (diagnostics
    only). ``source_path``, when given, is fingerprinted into each fold's manifest
    and the top-level result for provenance.
    """
    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"Unknown input variant {input_variant!r}.")
    if truncation_strategy not in TRUNCATION_STRATEGIES:
        raise ValueError(f"Unknown truncation strategy {truncation_strategy!r}.")
    frame = _prepare_frame(df, input_variant, allow_preliminary=allow_preliminary)
    fold_indices = make_stratified_folds(frame, n_splits=folds, seed=seed)
    cfg = cfg or model.TrainConfig()
    cfg = replace(cfg, seed=seed, truncation_strategy=truncation_strategy)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fold_number, (train_idx, holdout_idx) in enumerate(fold_indices, start=1):
        split = _outer_split(frame, train_idx, holdout_idx, seed + fold_number)
        fold_cfg = replace(cfg, seed=seed + fold_number)
        fold_dir = out_dir / f"fold-{fold_number:02d}"
        manifest = model.fine_tune(
            split, fold_cfg, out_dir=fold_dir, source_path=source_path
        )
        fold_results.append(
            {
                "fold": fold_number,
                "train_size": len(split.train),
                "validation_size": len(split.val),
                "holdout_size": len(split.test),
                **manifest["test_metrics"],
            }
        )

    metrics = pd.DataFrame(fold_results)
    aggregate = {
        metric: {
            "mean": float(metrics[metric].mean()),
            "std": float(metrics[metric].std(ddof=1)) if len(metrics) > 1 else 0.0,
        }
        for metric in ("macro_f1", "accuracy", "balanced_accuracy")
    }
    result = {
        "input_variant": input_variant,
        "truncation_strategy": truncation_strategy,
        "folds": folds,
        "seed": seed,
        "data_size": len(frame),
        "class_distribution": {
            str(k): int(v) for k, v in frame["label_id"].value_counts().sort_index().items()
        },
        "train_config": asdict(cfg),
        "fold_results": fold_results,
        "aggregate": aggregate,
        "reproducibility": model.reproducibility_metadata(),
        "source_file_sha256": (
            dataset.file_fingerprint(source_path) if source_path is not None else None
        ),
    }
    metrics.to_csv(out_dir / "cv_results.csv", index=False)
    (out_dir / "cv_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _remove_model_artifacts(root: Path) -> None:
    """Remove persisted model directories owned by an ablation run."""
    if not root.exists():
        return
    for artifact_name in ("best", "checkpoints"):
        for artifact_dir in root.rglob(artifact_name):
            if artifact_dir.is_dir():
                shutil.rmtree(artifact_dir, ignore_errors=True)


def run_ablation(
    df: pd.DataFrame,
    *,
    cfg: model.TrainConfig | None = None,
    folds: int = 5,
    seed: int = 42,
    out_dir: Path,
    input_variants: tuple[str, ...] = INPUT_VARIANTS,
    truncation_strategies: tuple[str, ...] = TRUNCATION_STRATEGIES,
    allow_preliminary: bool = False,
    source_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run the full matrix while retaining metrics, not 45 model copies."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_model_artifacts(out_dir)
    rows = []
    for input_variant in input_variants:
        for truncation_strategy in truncation_strategies:
            combo_dir = out_dir / f"{input_variant}__{truncation_strategy}"
            try:
                result = run_cross_validation(
                    df,
                    input_variant=input_variant,
                    truncation_strategy=truncation_strategy,
                    cfg=cfg,
                    folds=folds,
                    seed=seed,
                    out_dir=combo_dir,
                    allow_preliminary=allow_preliminary,
                    source_path=source_path,
                )
            finally:
                _remove_model_artifacts(combo_dir)
            row = {
                "input_variant": input_variant,
                "truncation_strategy": truncation_strategy,
                "data_size": result["data_size"],
            }
            for metric, values in result["aggregate"].items():
                row[f"{metric}_mean"] = values["mean"]
                row[f"{metric}_std"] = values["std"]
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    return summary
