"""Validation and agreement checks for human sentiment annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

ALLOWED_LABELS = frozenset(("NEGATIVE", "NEUTRAL", "POSITIVE"))


def load_annotation_file(path: str | Path) -> pd.DataFrame:
    """Load one completed annotation file and reject incomplete rows."""
    path = Path(path)
    frame = pd.read_csv(path)
    required = {"sample_id", "label"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing annotation columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{path}: annotation file is empty.")
    if frame["sample_id"].isna().any():
        raise ValueError(f"{path}: sample_id values cannot be empty.")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id values in {path}.")

    labels = frame["label"].fillna("").astype("string").str.strip().str.upper()
    missing = labels.eq("")
    invalid = ~labels.isin(ALLOWED_LABELS | {""})
    if missing.any():
        raise ValueError(f"{path}: {int(missing.sum())} rows have no label.")
    if invalid.any():
        values = sorted(set(labels[invalid].tolist()))
        raise ValueError(f"{path}: invalid labels {values}.")
    frame = frame.copy()
    frame["label"] = labels
    return frame


def compare_raters(paths: Sequence[str | Path]) -> dict:
    """Compare completed rater files and return agreement plus disagreements."""
    if len(paths) != 2:
        raise ValueError("Exactly two annotation files are required for Cohen's kappa.")
    frames = [load_annotation_file(path) for path in paths]
    base_ids = set(frames[0]["sample_id"])
    if any(set(frame["sample_id"]) != base_ids for frame in frames[1:]):
        raise ValueError("All annotation files must contain the same sample_id values.")

    aligned = [frame.set_index("sample_id")["label"] for frame in frames]
    labels = pd.concat(aligned, axis=1)
    labels.columns = [f"rater_{i}" for i in range(1, len(frames) + 1)]
    disagreements = labels[labels.nunique(axis=1) > 1].reset_index()

    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(labels.iloc[:, 0], labels.iloc[:, 1])
    return {
        "n_items": len(labels),
        "n_disagreements": len(disagreements),
        "agreement_rate": float(1 - len(disagreements) / len(labels)),
        "cohen_kappa": float(kappa),
        "disagreements": disagreements,
        "label_distribution": {
            column: labels[column].value_counts().sort_index().to_dict()
            for column in labels.columns
        },
    }
