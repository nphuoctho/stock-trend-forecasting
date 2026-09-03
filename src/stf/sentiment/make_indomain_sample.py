"""Build an in-domain sample from crawled Vietstock news for manual sentiment labeling.

Purpose: add a small IN-DOMAIN label set (in the actual "voice" of Vietstock news for the
10 tickers) alongside the public CafeF seed. Used to (a) fine-tune PhoBERT closer to the real
data, and (b) report Cohen's/Fleiss' kappa agreement.

Sampling strategy:
  - Only take articles that HAVE a body (full content makes labeling more accurate).
  - Stratify by ticker and by year so the sample represents the whole corpus without bias.
  - Export a CSV with an EMPTY `label` column for annotators to fill NEGATIVE/NEUTRAL/POSITIVE.
  - Ship a labeling guideline file to keep annotators consistent.

Run:
    uv run python -m stf.sentiment.make_indomain_sample --n 300
    uv run python -m stf.sentiment.make_indomain_sample --n 300 --raters 2
"""

from __future__ import annotations

import argparse

import pandas as pd

from stf import config

OUT_DIR = config.DATA / "labeled" / "indomain"
GUIDELINE = OUT_DIR / "labeling-guide.md"

# Guideline stays in Vietnamese on purpose: annotators are Vietnamese and the news is too.
_GUIDELINE_TEXT = """# Hướng dẫn gán nhãn cảm xúc tin tài chính (in-domain)

Gán mỗi bài vào ĐÚNG MỘT trong 3 lớp, dựa trên TÁC ĐỘNG KỲ VỌNG lên giá cổ phiếu
của mã liên quan, theo góc nhìn nhà đầu tư ngắn hạn.

## Ba lớp

- **POSITIVE** - tin có khả năng đẩy giá TĂNG: lợi nhuận vượt kỳ vọng, doanh thu kỷ lục,
  ký hợp đồng lớn, khối ngoại mua ròng, mở rộng kinh doanh thuận lợi, được nâng hạng.
- **NEGATIVE** - tin có khả năng đẩy giá GIẢM: thua lỗ, doanh thu sụt, khối ngoại bán ròng,
  bị xử phạt, nợ xấu tăng, lãnh đạo bị điều tra, triển vọng xấu.
- **NEUTRAL** - tin không rõ hướng tác động hoặc chỉ mang tính thông báo: lịch sự kiện,
  họp ĐHĐCĐ định kỳ, thay đổi nhân sự thường lệ, công bố thông tin theo quy định,
  bản tin tổng hợp thị trường không nghiêng về một mã.

## Nguyên tắc

1. Gán theo TÁC ĐỘNG LÊN GIÁ, không theo cảm xúc câu chữ. Tin "công ty cắt giảm chi phí
   mạnh" nghe tiêu cực nhưng có thể là POSITIVE nếu giúp cải thiện lợi nhuận.
2. Nếu phân vân giữa hai lớp, ưu tiên NEUTRAL (thận trọng).
3. Đọc cả tiêu đề và một phần nội dung (cột body) trước khi quyết.
4. Không đoán theo mã hay theo diễn biến giá đã biết; chỉ dựa vào NỘI DUNG TIN.

## Cách điền

- Điền cột `label` bằng đúng một trong: NEGATIVE / NEUTRAL / POSITIVE (viết hoa).
- Không sửa các cột khác. Không xóa dòng.
- Nếu có từ 2 người gán, mỗi người điền vào file riêng (label_r1.csv, label_r2.csv)
  để tính Cohen's kappa; bất đồng được thảo luận và chốt lại theo guideline.
"""


def make_sample(n: int, seed: int, raters: int) -> pd.DataFrame:
    if not config.ARTICLES_PQ.exists():
        raise FileNotFoundError(
            f"{config.ARTICLES_PQ} not found. Run the news crawl first."
        )

    articles = pd.read_parquet(config.ARTICLES_PQ)
    listings = pd.read_parquet(config.LISTINGS_PQ)

    # Only take articles that already have a body (full content for accurate labeling).
    has_body = articles["body"].notna() & (
        articles["body"].astype("string").str.len() > 0
    )
    pool = articles[has_body].copy()
    if pool.empty:
        raise RuntimeError(
            "No articles with a body yet. Wait for the body-crawl cron to run more."
        )

    # Attach ticker (an article may map to several; take the first for simplicity).
    url2ticker = listings.drop_duplicates("url").set_index("url")["ticker"].to_dict()
    pool["ticker"] = pool["url"].map(url2ticker)
    pool = pool.dropna(subset=["ticker"])
    pool["year"] = pd.to_datetime(pool["published_at"], errors="coerce").dt.year
    pool = pool.dropna(subset=["year"])

    # Stratify by (ticker, year): sample proportionally so it represents the whole corpus.
    frac = min(1.0, n / len(pool))
    picked_idx: list = []
    for _, g in pool.groupby(["ticker", "year"]):
        k = max(1, round(len(g) * frac))
        picked_idx.extend(g.sample(min(k, len(g)), random_state=seed).index.tolist())
    sample = pool.loc[picked_idx]
    # Trim back to exactly n (rounding may overshoot), then shuffle.
    sample = sample.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GUIDELINE.write_text(_GUIDELINE_TEXT, encoding="utf-8")

    # Labeling file: keep the needed info + an EMPTY label column.
    body_preview = sample["body"].astype("string").str.slice(0, 400)
    out = pd.DataFrame(
        {
            "sample_id": range(len(sample)),
            "ticker": sample["ticker"],
            "published_at": sample["published_at"],
            "title": sample["title"],
            "body_preview": body_preview,
            "url": sample["url"],
            "label": "",  # annotator fills NEGATIVE/NEUTRAL/POSITIVE
        }
    )

    files = []
    if raters <= 1:
        f = OUT_DIR / "to_label.csv"
        out.to_csv(f, index=False)
        files.append(f)
    else:
        for r in range(1, raters + 1):
            f = OUT_DIR / f"to_label_r{r}.csv"
            out.to_csv(f, index=False)
            files.append(f)

    print(
        f"[in-domain] built sample of {len(out)} articles (stratified by ticker x year, body only)"
    )
    print(
        f"[in-domain] distribution by ticker:\n{sample['ticker'].value_counts().to_string()}"
    )
    print(f"[in-domain] guideline: {GUIDELINE}")
    for f in files:
        print(f"[in-domain] labeling file (empty label column): {f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build an in-domain sample for sentiment labeling"
    )
    ap.add_argument(
        "--n", type=int, default=300, help="number of articles in the sample"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--raters",
        type=int,
        default=1,
        help="number of annotators (>=2 makes multiple files for kappa)",
    )
    args = ap.parse_args()
    make_sample(args.n, args.seed, args.raters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
