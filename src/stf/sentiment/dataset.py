"""Nạp dữ liệu nhãn cảm xúc và chuẩn bị cho fine-tune PhoBERT.

Nguồn nhãn (theo plan.md Phase 2):
  1. Seed công khai: corpus tiêu đề CafeF (3 lớp) để tái lập nhanh baseline.
  2. Bổ sung in-domain: mẫu tự gán nhãn theo hướng dẫn cố định (báo cáo Cohen/Fleiss κ).

Định dạng CSV/parquet nhãn kỳ vọng: cột `text` (str) và `label` (NEGATIVE/NEUTRAL/POSITIVE
hoặc 0/1/2). Nếu có cột `date` (ISO), split được thực hiện TÁCH THỜI GIAN để tránh rò rỉ.

PhoBERT lý tưởng dùng đầu vào đã word-segment (VnCoreNLP). Ở đây tokenize text thô cho
đơn giản/tái lập; nếu cần nâng chất lượng, thêm bước segment trước khi gọi tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stf.sentiment.labels import LABEL2ID


@dataclass
class Split:
    """Một lần chia dữ liệu train/val/test đã cố định."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def describe(self) -> str:
        def dist(df: pd.DataFrame) -> str:
            vc = df["label_id"].value_counts().sort_index().to_dict()
            return ", ".join(f"{k}:{v}" for k, v in vc.items())
        return (f"train={len(self.train)} ({dist(self.train)}) | "
                f"val={len(self.val)} ({dist(self.val)}) | "
                f"test={len(self.test)} ({dist(self.test)})")


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa cột `label` về `label_id` (int 0/1/2). Chấp nhận tên lớp hoặc số."""
    df = df.copy()
    if "label_id" in df.columns:
        return df
    raw = df["label"]
    if raw.dtype == object:
        df["label_id"] = raw.str.upper().map(LABEL2ID)
    else:
        df["label_id"] = raw.astype(int)
    if df["label_id"].isna().any():
        bad = df.loc[df["label_id"].isna(), "label"].unique()[:5]
        raise ValueError(f"Nhãn không hợp lệ: {bad}. Cần NEGATIVE/NEUTRAL/POSITIVE hoặc 0/1/2.")
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_labeled(path: str | Path) -> pd.DataFrame:
    """Nạp file nhãn (.csv/.parquet), trả DataFrame có cột text, label_id[, date]."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("File nhãn cần cột 'text' và 'label'.")
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    return normalize_labels(df)


def make_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    time_aware: bool = True,
) -> Split:
    """Chia train/val/test.

    time_aware=True và có cột `date`: chia theo THỜI GIAN (train = cũ nhất,
    test = mới nhất) để mô phỏng đúng ràng buộc chống rò rỉ khi đánh giá dự báo.
    Ngược lại: chia ngẫu nhiên phân tầng theo lớp (seed cố định).
    """
    df = df.reset_index(drop=True)
    if time_aware and "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        train = df.iloc[: n - n_val - n_test]
        val = df.iloc[n - n_val - n_test : n - n_test]
        test = df.iloc[n - n_test :]
        return Split(train.reset_index(drop=True),
                     val.reset_index(drop=True),
                     test.reset_index(drop=True))

    # Ngẫu nhiên phân tầng theo lớp.
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, grp in df.groupby("label_id"):
        idx = rng.permutation(len(grp))
        grp = grp.iloc[idx].reset_index(drop=True)
        n = len(grp)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        parts["test"].append(grp.iloc[:n_test])
        parts["val"].append(grp.iloc[n_test : n_test + n_val])
        parts["train"].append(grp.iloc[n_test + n_val :])
    return Split(
        pd.concat(parts["train"]).sample(frac=1, random_state=seed).reset_index(drop=True),
        pd.concat(parts["val"]).sample(frac=1, random_state=seed).reset_index(drop=True),
        pd.concat(parts["test"]).sample(frac=1, random_state=seed).reset_index(drop=True),
    )


def synthetic_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Sinh dataset giả cân bằng 3 lớp để smoke-test pipeline khi CHƯA có nhãn thật.

    KHÔNG dùng để báo cáo kết quả — chỉ để kiểm tra code train/eval chạy thông.
    """
    rng = np.random.default_rng(seed)
    pos = ["cổ phiếu tăng mạnh", "lợi nhuận vượt kỳ vọng", "doanh thu kỷ lục",
           "khối ngoại mua ròng", "triển vọng tích cực"]
    neg = ["cổ phiếu lao dốc", "thua lỗ nặng", "khối ngoại bán tháo",
           "nợ xấu tăng cao", "triển vọng ảm đạm"]
    neu = ["công bố thông tin định kỳ", "họp đại hội cổ đông", "thay đổi nhân sự",
           "phát hành trái phiếu", "cập nhật giao dịch"]
    rows = []
    pools = {0: neg, 1: neu, 2: pos}
    for i in range(n):
        lab = i % 3
        text = rng.choice(pools[lab]) + f" phiên {i}"
        rows.append({"text": text, "label_id": lab,
                     "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)})
    return pd.DataFrame(rows)
