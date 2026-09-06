"""Sentiment evaluation metrics and annotation agreement.

macro-F1 is the PRIMARY metric (class-imbalanced corpus). Plus accuracy, balanced
accuracy, per-class precision/recall/F1. Cohen's/Fleiss' κ for reporting annotation agreement.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict:
    """Return classification metrics for the fixed three-class label set."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
    if len(y_true) == 0:
        raise ValueError("Cannot calculate metrics for an empty set.")
    if not np.isin(y_true, [0, 1, 2]).all() or not np.isin(y_pred, [0, 1, 2]).all():
        raise ValueError("Labels must use the fixed ids 0, 1 and 2.")

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {i: float(v) for i, v in enumerate(f1)},
        "per_class_precision": {i: float(v) for i, v in enumerate(p)},
        "per_class_recall": {i: float(v) for i, v in enumerate(r)},
        "support": {i: int(v) for i, v in enumerate(support)},
    }


def fleiss_kappa(table: np.ndarray) -> float:
    """Calculate Fleiss' kappa from an item-by-class count table."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] < 2:
        raise ValueError("table must be a non-empty two-dimensional count matrix.")
    if not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("table counts must be finite and non-negative.")

    rater_counts = table.sum(axis=1)
    if not np.allclose(rater_counts, rater_counts[0]):
        raise ValueError("Every item must have the same number of ratings.")
    n_raters = rater_counts[0]
    if n_raters < 2 or not n_raters.is_integer():
        raise ValueError("Each item needs at least two ratings.")

    n_items = table.shape[0]
    p_j = table.sum(axis=0) / (n_items * n_raters)
    P_i = (np.square(table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    if np.isclose(1 - P_e, 0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))
