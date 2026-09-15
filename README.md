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
  cli.py                 CLI: prices | news | verify | sentiment-* | forecast-smoke
  data/
    prices.py            adjusted OHLCV loader (vnstock/VCI)
    news.py              Vietstock scraper: timestamp + title + body
  sentiment/
    dataset.py            labels, deduplication and time-aware split
    labels.py             the 3 classes NEGATIVE/NEUTRAL/POSITIVE
    metrics.py            macro-F1 fixed over NEG/NEU/POS
    model.py              fine-tune PhoBERT + 3-class probabilities
  forecasting/
    calendar.py          cutoff-safe news/session alignment
    features.py          causal price features and train-only scaler
    panel.py             price-sentiment panel and next-session target
    models.py             baselines and LSTM harness
tests/                   offline pipeline tests
data/                    raw/processed data (gitignored; rebuilt by scripts)
models/                  model checkpoints (gitignored)
```

## Data pipeline

```bash
# Fetch prices for the 10 VN30 tickers (2020-01 -> 2025-12-31)
uv run python -m stf.cli prices              # full run
uv run python -m stf.cli prices --limit 1    # quick 1-symbol test

# Crawl news (minute timestamp + title + body)
uv run python -m stf.cli news                     # full run (long)
uv run python -m stf.cli news --limit-urls 20     # quick 20-article test

# Coverage report for existing data (offline, no network)
uv run python -m stf.cli verify
```

```bash
# Offline smoke-test for point-in-time panel, labels and price LSTM
uv run python -m stf.cli forecast-smoke --n 40 --epochs 2
```

## Phase 2: PhoBERT sentiment

```bash
# Smoke-test the pipeline on fake data (checks the code runs, not real results)
uv run python -m stf.cli sentiment-smoke --n 60

# Fine-tune on a real label file (.csv/.parquet with text, label columns)
uv run python -m stf.cli sentiment-train --data data/labeled/seed.csv --epochs 3
```

### So sánh đầu vào và cắt độ dài

Khi tệp nhãn có các cột `title`, `body` (hoặc `body_preview`) và `label`, có thể
chạy một cấu hình với 5-fold cross-validation:

```bash
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/labeled.csv \
  --input-variant title_context \
  --truncation-strategy head_tail \
  --epochs 3 --folds 5 \
  --output models/experiments/title_context__head_tail
```

Mỗi fold giữ lại 20% dữ liệu làm outer holdout và dùng 80% còn lại để huấn luyện;
10% của phần huấn luyện được giữ lại để chọn checkpoint tốt nhất. Chỉ số chính là
`macro_f1` trên đủ ba lớp; kết quả từng fold được lưu trong `cv_results.csv` và
tổng hợp trong `cv_results.json`.

Khi dữ liệu huấn luyện bị lệch lớp, có thể dùng trọng số nghịch đảo tần suất.
Trọng số được tính riêng từ phần huấn luyện của từng fold; tập đánh giá không
bị lấy mẫu lại:

```bash
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/labeled.csv \
  --input-variant title_context \
  --truncation-strategy head_tail \
  --class-weighting inverse_frequency \
  --epochs 3 --folds 5 \
  --output models/experiments/title_context__head_tail__weighted
```

Chỉ bật tùy chọn này sau khi nhãn đã được con người rà soát. Nhãn sơ bộ do hệ
thống tạo chỉ dùng để chẩn đoán, không dùng làm số liệu chính.

### Kiểm tra chất lượng gán nhãn

Tập in-domain phải có hơn 300 mẫu. Mô hình ngôn ngữ lớn chỉ được dùng để tạo
nhãn sơ bộ và gợi ý mức độ không chắc chắn; một người duy nhất rà soát toàn bộ
nhãn trước khi huấn luyện chính thức. Không dùng nhãn sơ bộ làm kết quả chính và
không báo cáo Cohen's kappa vì quy trình không có người gán nhãn thứ hai.

Chạy toàn bộ ma trận 3 phương án đầu vào (`title`, `context`, `title_context`)
nhân 3 cách cắt (`head`, `tail`, `head_tail`) tuần tự:

```bash
uv run python -m stf.cli sentiment-ablation \
  --data data/labeled/indomain/labeled.csv \
  --epochs 3 --folds 5 \
  --output models/experiments/ablation
```

Không dùng nhãn hoặc giá tương lai để tạo đầu vào cảm xúc. Tập CafeF chỉ có
tiêu đề nên không đủ để kết luận riêng về `context`; cần dùng tệp Vietstock đã
gán nhãn có nội dung bài viết.

> This machine has no GPU/CUDA. Real fine-tuning should run on a GPU (Google Colab/Kaggle).
> See `notebooks/training-guide.md`.

## Tests

```bash
uv run pytest tests/ -q
```
