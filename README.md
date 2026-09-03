# Stock Trend Forecasting

Stock price movement forecasting from Vietnamese financial-news sentiment, combining a
Transformer (PhoBERT) sentiment branch with a time-series price branch. Market: HOSE/VN30.

**Đề tài:** Xây dựng hệ thống dự báo xu hướng biến động giá cổ phiếu dựa trên phân tích cảm xúc tin tức tài chính bằng mô hình Transformer và chuỗi thời gian.

**GVHD:** TS. Đặng Văn Thìn · **SV:** Nguyễn Phước Thọ (25410139) · Lớp LT.K2025.2.TTNT.

## Environment

- Python 3.12, managed with `uv`. Create/update the env with `uv sync`.
- Run commands with `uv run`.

## Layout

```
src/stf/                 main package
  config.py              central config: tickers, date window, paths
  cli.py                 CLI: prices | news | verify | sentiment-smoke | sentiment-train
  data/
    prices.py            adjusted OHLCV loader (vnstock/VCI)
    news.py              Vietstock scraper: timestamp + title + body
  sentiment/
    labels.py            the 3 classes NEGATIVE/NEUTRAL/POSITIVE
    dataset.py           load labels, time-based split (leakage-safe)
    metrics.py           macro-F1, per-class, Cohen/Fleiss kappa
    model.py             fine-tune PhoBERT + 3-class probabilities
tests/                   offline pipeline tests
data/                    raw/processed data (gitignored; rebuilt by scripts)
models/                  model checkpoints (gitignored)
```

## Data pipeline

```bash
# Fetch prices for the 10 VN30 tickers (2020-01 -> 2026-03-31)
uv run python -m stf.cli prices              # full run
uv run python -m stf.cli prices --limit 1    # quick 1-symbol test

# Crawl news (minute timestamp + title + body)
uv run python -m stf.cli news                     # full run (long)
uv run python -m stf.cli news --limit-urls 20     # quick 20-article test

# Coverage report for existing data (offline, no network)
uv run python -m stf.cli verify
```

## Phase 2: PhoBERT sentiment

```bash
# Smoke-test the pipeline on fake data (checks the code runs, not real results)
uv run python -m stf.cli sentiment-smoke --n 60

# Fine-tune on a real label file (.csv/.parquet with text, label columns)
uv run python -m stf.cli sentiment-train --data data/labeled/seed.csv --epochs 3
```

> This machine has no GPU/CUDA. Real fine-tuning should run on a GPU (Google Colab/Kaggle).
> See `notebooks/training-guide.md`.

## Tests

```bash
uv run pytest tests/ -q
```
