"""Load sentiment-labeled data and prepare it for PhoBERT fine-tuning.

Label sources:
  1. Public seed: the CafeF headline corpus (3 classes) for a quick baseline.
  2. In-domain add-on: self-labeled samples following a fixed guideline (report Cohen/Fleiss kappa).

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


def load_labeled(path: str | Path) -> pd.DataFrame:
    """Load a label file and return text plus normalized integer labels."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Label file must be .csv or .parquet.")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Label file needs 'text' and 'label' columns.")
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype("string").str.strip()
    df = df[df["text"].ne("")].reset_index(drop=True)
    return normalize_labels(df)


def make_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    time_aware: bool = True,
) -> Split:
    """Split into train, validation and test sets without empty partitions."""
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < val_frac < 1 or not 0 < test_frac < 1:
        raise ValueError("val_frac and test_frac must be between 0 and 1.")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must leave training rows.")
    if "label_id" not in df.columns:
        raise ValueError("Dataset needs a normalized 'label_id' column.")

    df = df.reset_index(drop=True)
    n = len(df)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n - n_val - n_test < 1:
        raise ValueError("Dataset is too small for train, validation and test splits.")

    if time_aware and "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("Time-aware splits require valid dates in every row.")
        df = df.assign(_split_date=dates).sort_values("_split_date").drop(
            columns="_split_date"
        )
        train_end = n - n_val - n_test
        val_end = n - n_test
        return Split(
            df.iloc[:train_end].reset_index(drop=True),
            df.iloc[train_end:val_end].reset_index(drop=True),
            df.iloc[val_end:].reset_index(drop=True),
        )

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


def synthetic_dataset(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Build a balanced 3-class fake dataset to smoke-test the pipeline before real labels exist.

    Not for reporting results - only to check the train/eval code runs.
    """
    rng = np.random.default_rng(seed)
    # Vietnamese phrases on purpose: the real model reads Vietnamese financial news.
    pos = [
        "cổ phiếu tăng mạnh",
        "lợi nhuận vượt kỳ vọng",
        "doanh thu kỷ lục",
        "khối ngoại mua ròng",
        "triển vọng tích cực",
    ]
    neg = [
        "cổ phiếu lao dốc",
        "thua lỗ nặng",
        "khối ngoại bán tháo",
        "nợ xấu tăng cao",
        "triển vọng ảm đạm",
    ]
    neu = [
        "công bố thông tin định kỳ",
        "họp đại hội cổ đông",
        "thay đổi nhân sự",
        "phát hành trái phiếu",
        "cập nhật giao dịch",
    ]
    rows = []
    pools = {0: neg, 1: neu, 2: pos}
    for i in range(n):
        lab = i % 3
        text = rng.choice(pools[lab]) + f" phiên {i}"
        rows.append(
            {
                "text": text,
                "label_id": lab,
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            }
        )
    return pd.DataFrame(rows)
