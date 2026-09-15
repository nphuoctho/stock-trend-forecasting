"""Sentiment evaluation metrics for the fixed three-class protocol.

Macro-F1 is the primary fixed-three-class metric. Balanced accuracy keeps its
standard definition over classes present in the evaluation targets.
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
    fixed_labels = [0, 1, 2]
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=fixed_labels,
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_f1": {i: float(v) for i, v in enumerate(f1)},
        "per_class_precision": {i: float(v) for i, v in enumerate(p)},
        "per_class_recall": {i: float(v) for i, v in enumerate(r)},
        "support": {i: int(v) for i, v in enumerate(support)},
    }
