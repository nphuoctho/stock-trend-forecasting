"""Load sentiment-labeled data and prepare it for PhoBERT fine-tuning.

Label sources (per plan.md Phase 2):
  1. Public seed: the CafeF headline corpus (3 classes) for a quick baseline.
  2. In-domain add-on: self-labeled samples following a fixed guideline (report Cohen/Fleiss κ).

Expected label CSV/parquet format: a `text` column (str) and a `label` column
(NEGATIVE/NEUTRAL/POSITIVE or 0/1/2). If a `date` column (ISO) exists, the split is done
BY TIME to avoid leakage.

PhoBERT ideally takes word-segmented input (VnCoreNLP). Here we tokenize raw text for
simplicity/reproducibility; to improve quality, add a segmentation step before the tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stf.sentiment.labels import LABEL2ID


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
        return (f"train={len(self.train)} ({dist(self.train)}) | "
                f"val={len(self.val)} ({dist(self.val)}) | "
                f"test={len(self.test)} ({dist(self.test)})")


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the `label` column to `label_id` (int 0/1/2). Accepts class names or numbers."""
    df = df.copy()
    if "label_id" in df.columns:
        return df
    raw = df["label"]
    # Tell text labels (object or pandas' newer string dtype) apart from numeric ones.
    is_text = raw.dtype == object or pd.api.types.is_string_dtype(raw)
    if is_text:
        df["label_id"] = raw.astype("string").str.upper().str.strip().map(LABEL2ID)
    else:
        df["label_id"] = raw.astype(int)
    if df["label_id"].isna().any():
        bad = df.loc[df["label_id"].isna(), "label"].unique()[:5]
        raise ValueError(f"Invalid labels: {bad}. Expected NEGATIVE/NEUTRAL/POSITIVE or 0/1/2.")
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_labeled(path: str | Path) -> pd.DataFrame:
    """Load a label file (.csv/.parquet), return a DataFrame with text, label_id[, date]."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Label file needs 'text' and 'label' columns.")
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
    """Split into train/val/test.

    time_aware=True with a `date` column: split BY TIME (train = oldest,
    test = newest) to match the no-leakage constraint of forecast evaluation.
    Otherwise: random split stratified by class (fixed seed).
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

    # Random split stratified by class.
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
    """Build a balanced 3-class fake dataset to smoke-test the pipeline before real labels exist.

    Not for reporting results — only to check the train/eval code runs.
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
