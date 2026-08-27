# Dự báo xu hướng giá cổ phiếu từ cảm xúc tin tài chính - Stock Trend Forecasting

**TÊN ĐỀ TÀI:** Xây dựng hệ thống dự báo xu hướng biến động giá cổ phiếu dựa trên phân tích cảm xúc tin tức tài chính bằng mô hình Transformer và chuỗi thời gian

**TÊN ĐỀ TÀI (tiếng Anh):** Building a stock price movement forecasting system based on financial news sentiment analysis using Transformer and time-series models

**Cán bộ hướng dẫn:** TS. Đặng Văn Thìn

**Thời gian thực hiện:** Từ ngày 15/07/2026 đến ngày 23/09/2026

**Sinh viên thực hiện:** Nguyễn Phước Thọ – 25410139 – Lớp LT.K2025.2.TTNT

## Môi trường

- Python 3.12, quản lý bằng `uv`. Tạo/cập nhật môi trường bằng `uv sync`.
- Chạy lệnh bằng `uv run`.

## Cấu trúc

```
src/stf/                 package chính
  config.py              cấu hình tập trung: mã, cửa sổ thời gian, đường dẫn
  cli.py                 CLI: prices | news | verify | sentiment-smoke | sentiment-train
  data/
    prices.py            loader giá OHLCV điều chỉnh (vnstock/VCI)
    news.py              scraper tin Vietstock: timestamp + tiêu đề + nội dung
  sentiment/
    labels.py            định nghĩa 3 lớp NEGATIVE/NEUTRAL/POSITIVE
    dataset.py           nạp nhãn, split tách thời gian (chống rò rỉ)
    metrics.py           macro-F1, per-class, Cohen/Fleiss kappa
    model.py             fine-tune PhoBERT + sinh xác suất 3 lớp
tests/                   test pipeline (offline)
data/                    dữ liệu thô/xử lý (gitignore; tái tạo bằng script)
models/                  checkpoint mô hình (gitignore)
```

## Chạy pipeline dữ liệu

```bash
# Kéo giá 10 mã VN30 (01/2020 -> 31/03/2026)
uv run python -m stf.cli prices              # toàn bộ
uv run python -m stf.cli prices --limit 1    # thử nhanh 1 mã

# Crawl tin (timestamp phút + tiêu đề + nội dung bài)
uv run python -m stf.cli news                     # toàn bộ (chạy nền dài)
uv run python -m stf.cli news --limit-urls 20     # thử nhanh 20 bài

# Báo cáo coverage dữ liệu đã có (offline, không tải mạng)
uv run python -m stf.cli verify
```

## Phase 2 - PhoBERT sentiment

```bash
# Smoke-test pipeline bằng dữ liệu giả (verify code chạy thông, KHÔNG phải kết quả thật)
uv run python -m stf.cli sentiment-smoke --n 60

# Fine-tune trên file nhãn thật (.csv/.parquet có cột text, label)
uv run python -m stf.cli sentiment-train --data data/labeled/seed.csv --epochs 3
```

> Máy phát triển hiện tại không có GPU/CUDA. Fine-tune thật nên chạy trên GPU
> (Google Colab/Kaggle). Xem `docs/phase2-phobert-guide.md` (trong thư mục docs của
> project research) để biết quy trình training trên GPU và ràng buộc chống rò rỉ.

## Kiểm thử

```bash
uv run pytest tests/ -q
```
