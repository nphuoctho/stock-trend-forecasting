"""Normalize the public CafeF seed (209sontung) into the pipeline's label format.

Source: https://github.com/209sontung/Vietnamese-stock-article-classification
  Dataset/raw_data.xlsx - 1005 expert-labeled CafeF headlines, 3 classes.
  Original label -> class mapping: 1=NEGATIVE (187), 2=NEUTRAL (249), 3=POSITIVE (569).

Output: data/labeled/cafef_seed/cafef_seed.parquet with text, label (class name) columns.
This is training data to TEACH PhoBERT, distinct from the crawled Vietstock news (for inference).

Run: uv run --with openpyxl python -m stf.sentiment.prepare_cafef_seed
"""

from __future__ import annotations

import pandas as pd

from stf import config
from stf.sentiment.labels import LABELS

# Maps the original numeric labels in raw_data.xlsx -> the pipeline's class names.
_RAW2LABEL = {1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE"}

SEED_DIR = config.DATA / "labeled" / "cafef_seed"
RAW_XLSX = SEED_DIR / "raw_data.xlsx"
OUT_PARQUET = SEED_DIR / "cafef_seed.parquet"
OUT_CSV = SEED_DIR / "cafef_seed.csv"


def prepare() -> pd.DataFrame:
    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"{RAW_XLSX} not found. Download it first:\n"
            "  curl -sL -o data/labeled/cafef_seed/raw_data.xlsx "
            "https://raw.githubusercontent.com/209sontung/"
            "Vietnamese-stock-article-classification/main/Dataset/raw_data.xlsx"
        )
    df = pd.read_excel(RAW_XLSX)
    df = df.rename(columns={"title": "text"})
    df["label"] = df["label"].map(_RAW2LABEL)
    if df["label"].isna().any():
        raise ValueError("Found source labels outside {1,2,3}; check raw_data.xlsx")
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[["text", "label"]]

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"[cafef-seed] {len(df)} titles normalized -> {OUT_PARQUET.name}, {OUT_CSV.name}")
    print("[cafef-seed] class distribution:")
    for name in LABELS:
        print(f"    {name}: {(df['label'] == name).sum()}")
    return df


if __name__ == "__main__":
    prepare()
