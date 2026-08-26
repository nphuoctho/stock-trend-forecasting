"""Chuẩn hóa seed CafeF công khai (209sontung) về định dạng nhãn của pipeline.

Nguồn: https://github.com/209sontung/Vietnamese-stock-article-classification
  Dataset/raw_data.xlsx - 1005 tiêu đề tin CafeF, gán nhãn bởi chuyên gia, 3 lớp.
  Mapping label gốc -> lớp: 1=NEGATIVE (187), 2=NEUTRAL (249), 3=POSITIVE (569).

Đầu ra: data/labeled/cafef_seed/cafef_seed.parquet với cột text, label (tên lớp).
Đây là dữ liệu để DẠY PhoBERT (train), khác với tin Vietstock crawl (để inference).

Chạy: uv run --with openpyxl python -m stf.sentiment.prepare_cafef_seed
"""

from __future__ import annotations

import pandas as pd

from stf import config
from stf.sentiment.labels import LABELS

# Mapping số nhãn gốc trong raw_data.xlsx -> tên lớp chuẩn của pipeline.
_RAW2LABEL = {1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE"}

SEED_DIR = config.DATA / "labeled" / "cafef_seed"
RAW_XLSX = SEED_DIR / "raw_data.xlsx"
OUT_PARQUET = SEED_DIR / "cafef_seed.parquet"
OUT_CSV = SEED_DIR / "cafef_seed.csv"


def prepare() -> pd.DataFrame:
    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"Chưa có {RAW_XLSX}. Tải trước:\n"
            "  curl -sL -o data/labeled/cafef_seed/raw_data.xlsx "
            "https://raw.githubusercontent.com/209sontung/"
            "Vietnamese-stock-article-classification/main/Dataset/raw_data.xlsx"
        )
    df = pd.read_excel(RAW_XLSX)
    df = df.rename(columns={"title": "text"})
    df["label"] = df["label"].map(_RAW2LABEL)
    if df["label"].isna().any():
        raise ValueError("Có nhãn gốc ngoài {1,2,3}; kiểm tra lại raw_data.xlsx")
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"]).reset_index(drop=True)
    df = df[["text", "label"]]

    SEED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)

    print(f"[cafef-seed] {len(df)} tiêu đề đã chuẩn hóa -> {OUT_PARQUET.name}, {OUT_CSV.name}")
    print("[cafef-seed] phân bố lớp:")
    for name in LABELS:
        print(f"    {name}: {(df['label'] == name).sum()}")
    return df


if __name__ == "__main__":
    prepare()
