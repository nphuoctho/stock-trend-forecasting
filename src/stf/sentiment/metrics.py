"""Chỉ số đánh giá cảm xúc và độ đồng thuận gán nhãn.

macro-F1 là chỉ số CHÍNH (corpus lệch lớp). Kèm accuracy, balanced accuracy,
per-class precision/recall/F1. Cohen's/Fleiss' κ cho báo cáo độ đồng thuận annotation.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict:
    """Trả dict các chỉ số phân loại. macro-F1 là khóa 'macro_f1'."""
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
    """Cohen's κ cho HAI người gán nhãn (báo cáo acceptance #6)."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(rater_a, rater_b))


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' κ cho TỪ BA người gán nhãn trở lên.

    table: ma trận (n_items, n_categories), mỗi ô = số người gán item vào lớp đó.
    Tự cài (sklearn không có) theo công thức Fleiss 1971.
    """
    table = np.asarray(table, dtype=float)
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)[0]  # giả định mọi item cùng số người gán
    p_j = table.sum(axis=0) / (n_items * n_raters)  # tỷ lệ mỗi lớp
    P_i = (np.square(table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    if np.isclose(1 - P_e, 0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))
