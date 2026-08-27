"""Sentiment evaluation metrics and annotation agreement.

macro-F1 is the PRIMARY metric (class-imbalanced corpus). Plus accuracy, balanced
accuracy, per-class precision/recall/F1. Cohen's/Fleiss' κ for reporting annotation agreement.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict:
    """Return a dict of classification metrics. macro-F1 is under the 'macro_f1' key."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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


def cohen_kappa(rater_a, rater_b) -> float:
    """Cohen's κ for TWO annotators (acceptance report #6)."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(rater_a, rater_b))


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' κ for THREE OR MORE annotators.

    table: an (n_items, n_categories) matrix; each cell = how many raters put the item in that class.
    Implemented here (sklearn has none) from the Fleiss 1971 formula.
    """
    table = np.asarray(table, dtype=float)
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)[0]  # assumes every item has the same rater count
    p_j = table.sum(axis=0) / (n_items * n_raters)  # proportion per class
    P_i = (np.square(table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    if np.isclose(1 - P_e, 0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))
