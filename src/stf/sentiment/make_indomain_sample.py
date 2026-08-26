"""Tạo mẫu in-domain từ tin Vietstock đã crawl để gán nhãn cảm xúc thủ công.

Mục đích: bổ sung một tập nhỏ nhãn IN-DOMAIN (đúng "giọng" tin Vietstock của 10 mã)
cạnh seed CafeF công khai. Dùng để (a) tinh chỉnh PhoBERT sát dữ liệu thật, và
(b) báo cáo độ đồng thuận Cohen's/Fleiss' kappa.

Chiến lược lấy mẫu:
  - Chỉ lấy bài ĐÃ CÓ body (nội dung đầy đủ giúp gán nhãn chính xác hơn).
  - Phân tầng theo mã và theo năm để mẫu đại diện toàn corpus, không lệch.
  - Xuất CSV có cột `label` TRỐNG để người gán điền NEGATIVE/NEUTRAL/POSITIVE.
  - Kèm file guideline gán nhãn để đảm bảo nhất quán giữa những người gán.

Chạy:
    uv run python -m stf.sentiment.make_indomain_sample --n 300
    uv run python -m stf.sentiment.make_indomain_sample --n 300 --raters 2
"""

from __future__ import annotations

import argparse

import pandas as pd

from stf import config

OUT_DIR = config.DATA / "labeled" / "indomain"
GUIDELINE = OUT_DIR / "HUONG_DAN_GAN_NHAN.md"

_GUIDELINE_TEXT = """# Hướng dẫn gán nhãn cảm xúc tin tài chính (in-domain)

Gán mỗi bài vào ĐÚNG MỘT trong 3 lớp, dựa trên TÁC ĐỘNG KỲ VỌNG lên giá cổ phiếu
của mã liên quan, theo góc nhìn nhà đầu tư ngắn hạn.

## Ba lớp

- **POSITIVE** — tin có khả năng đẩy giá TĂNG: lợi nhuận vượt kỳ vọng, doanh thu kỷ lục,
  ký hợp đồng lớn, khối ngoại mua ròng, mở rộng kinh doanh thuận lợi, được nâng hạng.
- **NEGATIVE** — tin có khả năng đẩy giá GIẢM: thua lỗ, doanh thu sụt, khối ngoại bán ròng,
  bị xử phạt, nợ xấu tăng, lãnh đạo bị điều tra, triển vọng xấu.
- **NEUTRAL** — tin không rõ hướng tác động hoặc chỉ mang tính thông báo: lịch sự kiện,
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
        raise FileNotFoundError(f"Chưa có {config.ARTICLES_PQ}. Chạy crawl tin trước.")

    articles = pd.read_parquet(config.ARTICLES_PQ)
    listings = pd.read_parquet(config.LISTINGS_PQ)

    # Chỉ lấy bài đã có body (nội dung đầy đủ để gán nhãn chính xác).
    has_body = articles["body"].notna() & (articles["body"].astype("string").str.len() > 0)
    pool = articles[has_body].copy()
    if pool.empty:
        raise RuntimeError("Chưa có bài nào có body. Đợi cron crawl body chạy thêm.")

    # Ghép mã (một bài có thể gắn nhiều mã; lấy mã đầu tiên cho đơn giản).
    url2ticker = listings.drop_duplicates("url").set_index("url")["ticker"].to_dict()
    pool["ticker"] = pool["url"].map(url2ticker)
    pool = pool.dropna(subset=["ticker"])
    pool["year"] = pd.to_datetime(pool["published_at"], errors="coerce").dt.year
    pool = pool.dropna(subset=["year"])

    # Phân tầng theo (mã, năm): rút tỷ lệ đều để mẫu đại diện toàn corpus.
    frac = min(1.0, n / len(pool))
    picked_idx: list = []
    for _, g in pool.groupby(["ticker", "year"]):
        k = max(1, round(len(g) * frac))
        picked_idx.extend(g.sample(min(k, len(g)), random_state=seed).index.tolist())
    sample = pool.loc[picked_idx]
    # Cắt về đúng n (nếu dư do làm tròn), xáo trộn.
    sample = sample.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GUIDELINE.write_text(_GUIDELINE_TEXT, encoding="utf-8")

    # File để gán nhãn: giữ thông tin cần thiết + cột label TRỐNG.
    body_preview = sample["body"].astype("string").str.slice(0, 400)
    out = pd.DataFrame({
        "sample_id": range(len(sample)),
        "ticker": sample["ticker"],
        "published_at": sample["published_at"],
        "title": sample["title"],
        "body_preview": body_preview,
        "url": sample["url"],
        "label": "",  # người gán điền NEGATIVE/NEUTRAL/POSITIVE
    })

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

    print(f"[in-domain] tạo mẫu {len(out)} bài (phân tầng theo mã x năm, chỉ bài có body)")
    print(f"[in-domain] phân bố theo mã:\n{sample['ticker'].value_counts().to_string()}")
    print(f"[in-domain] guideline: {GUIDELINE}")
    for f in files:
        print(f"[in-domain] file gán nhãn (cột label trống): {f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tạo mẫu in-domain để gán nhãn cảm xúc")
    ap.add_argument("--n", type=int, default=300, help="số bài trong mẫu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raters", type=int, default=1,
                    help="số người gán nhãn (>=2 sinh nhiều file để tính kappa)")
    args = ap.parse_args()
    make_sample(args.n, args.seed, args.raters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
