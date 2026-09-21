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
  cli.py                 CLI: prices | news | verify | sentiment-* | forecast-smoke | score-news
  data/
    prices.py            adjusted OHLCV loader (vnstock/VCI)
    news.py              Vietstock scraper: timestamp + title + body
  sentiment/
    dataset.py            labels, deduplication and time-aware split
    experiments.py        leakage-safe CV and input/truncation ablations
    make_indomain_sample.py  one-reviewer Vietstock sample generator
    labels.py             the 3 classes NEGATIVE/NEUTRAL/POSITIVE
    metrics.py            macro-F1 fixed over NEG/NEU/POS
    model.py              fine-tune PhoBERT + 3-class probabilities
  forecasting/
    calendar.py           cutoff-safe news/session alignment
    features.py           causal price features and train-only scaler
    labels.py             train-only trend thresholds and target labels
    panel.py              price-sentiment panel and next-session target
    sentiment_agg.py      daily probability aggregation
    split.py              chronological and walk-forward splits
    models.py             baselines and LSTM harness
tests/                   offline pipeline tests
data/                    raw/processed data (gitignored; rebuilt by scripts)
models/                  model checkpoints (gitignored)
```

## Data pipeline

# Fetch prices for the 10 VN30 tickers (2020-01 -> 2025-12-31)
uv run python -m stf.cli prices              # full run
uv run python -m stf.cli prices --limit 1    # quick 1-symbol test

# Crawl news (minute timestamp + title + body)
uv run python -m stf.cli news                 # full run (long)
# `--refresh` re-crawls listings and repairs cached listing dates after parser changes.
uv run python -m stf.cli news --refresh       # full crawl/refresh
uv run python -m stf.cli news --limit-urls 20 # quick 20-article test

# Coverage report for existing data (offline, no network)
uv run python -m stf.cli verify
```

```bash
# Offline smoke-test for point-in-time panel, labels, and both LSTM variants
# (price-only PriceLSTM and the two-branch PriceSentimentLSTM)
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
Kết quả lần chạy Kaggle và quyết định nghiên cứu được ghi tại
[`docs/sentiment-experiment-results.md`](docs/sentiment-experiment-results.md).


Khi tệp nhãn có các cột `title`, `body` (hoặc `body_preview`) và `label`, có thể
chạy một cấu hình với 5-fold cross-validation:

```bash
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/to_label_r1.csv \
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
  --data data/labeled/indomain/to_label_r1.csv \
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

Theo mặc định, `sentiment-train`/`sentiment-cv`/`sentiment-ablation` từ chối các
dòng còn gắn cờ `annotation_status=PRELIMINARY_REVIEW_REQUIRED` hoặc
`annotation_source=assistant_prelabel` (ví dụ `labeled.csv`). Thêm
`--allow-preliminary` chỉ để chẩn đoán nhanh; không dùng cờ này khi báo cáo kết
quả chính thức. Tệp người dùng rà soát phải chỉ chứa nhãn cuối hợp lệ và được
kiểm tra trước khi chạy.

Chạy toàn bộ ma trận 3 phương án đầu vào (`title`, `context`, `title_context`)
nhân 3 cách cắt (`head`, `tail`, `head_tail`) tuần tự:

```bash
uv run python -m stf.cli sentiment-ablation \
  --data data/labeled/indomain/to_label_r1.csv \
  --epochs 3 --folds 5 \
  --output models/experiments/ablation
```

Lệnh `sentiment-ablation` chỉ giữ các tệp số liệu và bản kê nguồn của từng cấu hình;
để tránh đầy đĩa, không lưu các thư mục mô hình `best/` của từng fold. Sau khi chọn
cấu hình, chạy `sentiment-cv` để tạo artifact xác thực chéo. Checkpoint dùng suy luận
phải được tạo bằng `sentiment-refit`, lệnh này khóa cấu hình và tệp nhãn theo
`cv_results.json`, rồi học lại trên toàn bộ nhãn đã duyệt mà không dùng outer holdout.

Không dùng nhãn hoặc giá tương lai để tạo đầu vào cảm xúc. Tập CafeF chỉ có
tiêu đề nên không đủ để kết luận riêng về `context`; cần dùng tệp Vietstock đã
gán nhãn có nội dung bài viết.

### Gán nhãn cảm xúc cho tin đã crawl

`sentiment-refit` chỉ nhận tệp nhãn đã duyệt và từ chối cấu hình không khớp artifact
xác thực chéo. Lần chạy thực cần GPU (Kaggle/Colab); xem `notebooks/training-guide.md`.

```bash
uv run python -m stf.cli sentiment-refit \
  --data data/labeled/indomain/labeled_merged.csv \
  --cv-results outputs/sentiment-cv-merged/cv_results.json \
  --input-variant title_context \
  --truncation-strategy head_tail \
  --class-weighting inverse_frequency \
  --epochs 5 --batch-size 16 --seed 42 \
  --output models/sentiment/merged-refit
```

`--context-chars` giữ độ dài ngữ cảnh khớp với tệp nhãn đã huấn luyện. Tệp nhãn chặn
`body_preview` ở 400 ký tự, còn `articles.parquet` giữ nội dung đầy đủ; bỏ cờ này sẽ
suy luận trên đầu vào dài hơn miền huấn luyện.

```bash
uv run python -m stf.cli score-news \
  --model-dir models/sentiment/merged-refit/best \
  --input-variant title_context \
  --context-chars 400 \
  --batch-size 64 \
  --output data/processed/news_sentiment_merged.parquet
```

## Phase 4: forecasting experiment

`forecast-smoke` only proves the code runs. `forecast` is the command that produces
reportable numbers: it loads the real price parquets, optionally a `score-news` parquet,
and runs the full ladder on walk-forward windows.

```bash
# Đối chứng chỉ giá: mọi phiên nhận prior trung tính. Giữ cùng cấu hình phía dưới
# để hai lần chạy chỉ khác thông tin đi vào nhánh cảm xúc.
uv run python -m stf.cli forecast --windows 5 --test-size 60 --val-size 60 \
  --epochs 40 --patience 6 --seeds 42 43 44 \
  --output outputs/forecast_merged_control

# Thang đầy đủ với cảm xúc; hai nhánh dùng cùng cửa sổ, hạt giống và hàng kiểm thử.
uv run python -m stf.cli forecast \
  --news-sentiment data/processed/news_sentiment_merged.parquet \
  --windows 5 --test-size 60 --val-size 60 --epochs 40 --patience 6 \
  --seeds 42 43 44 --output outputs/forecast_merged_sentiment

# Tách ảnh hưởng kiến trúc và giá trị thông tin.
uv run python -m stf.cli forecast-compare \
  --real outputs/forecast_merged_sentiment \
  --control outputs/forecast_merged_control \
  --output outputs/forecast_merged_sentiment/information_gain.json
```

The ladder is `majority`, `random`, `logreg_price`, `logreg_price_sentiment`,
`lstm_price`, `lstm_price_sentiment`. Per window the command refits the trend thresholds
and both feature scalers on training rows only, selects each LSTM checkpoint on that
window's validation macro-F1, and scores every arm once on the same test rows. Results
are averaged over `--seeds`. The sentiment contribution is reported twice: as a paired
per-window delta bootstrapped over the 5 windows, and as a bootstrap over the ~300 test
**dates** (all tickers of a date resample together). Prefer the date-block interval; the
window interval has only 5 blocks and is coarse enough to exclude zero by accident.

`forecast-compare` exists because comparing the two-branch arm against the single-branch
price model conflates two changes: the extra branch, and the information it carries. The

control run keeps the architecture and removes only the information, so
`architecture_effect + information_gain = naive_delta` exactly.

Trước khi tính chênh lệch, `forecast-compare` bắt buộc hai lần chạy có cùng cấu hình,
mã băm dữ liệu giá, các ngày kiểm thử và khóa dự đoán `(window, seed, ticker,
target_date, y_true)`. Vì vậy không thể ghép một artifact cũ hoặc tập kiểm thử khác
vào phép đo giá trị thông tin.

Artifacts written to `--output`:

| File | Content |
| --- | --- |
| `forecast_results.json` | full record: per-window thresholds, dates, class distributions, per-seed metrics, ablation deltas with CI, news stratification, data hashes |
| `forecast_metrics.csv` | the summary table for the report |
| `forecast_stratified.csv` | per-window/seed metrics split by whether the row had news |
| `forecast_predictions.csv` | test predictions for every arm and seed, for error analysis |

The three trend classes are cut at the training window's return terciles, so the classes
are balanced by construction and **the chance level is 0.333, not 0.5**. Report `macro_f1`,
`balanced_accuracy` and macro OvR-AUC; accuracy alone is not interpretable here.

## Expanding the sentiment label set

`label-candidates` draws a stratified annotation batch. A uniform draw spends the budget
on the neutral majority — the first 306-row batch yielded only 31 `NEGATIVE` — so the
command splits the draw into one stratum that preserves the corpus prior and several that
oversample the minority classes.

```bash
uv run python -m stf.cli label-candidates \
  --exclude data/labeled/indomain/to_label_r1.csv \
  --news-sentiment data/processed/news_sentiment.parquet \
  --eval-random 350 --negative 400 --positive 250 --active 0 \
  --output data/labeled/indomain/to_label_batch2.csv

# After annotating: validate structure and report minority-class precision
uv run python -m stf.cli label-audit --data data/labeled/indomain/to_label_batch2_ai.csv

# Record the completed human review of the legacy batch as well.
uv run python -m stf.cli label-finalize --data data/labeled/indomain/to_label_r1.csv

# Attest that a human confirmed the labels; rewrites BOTH provenance fields
uv run python -m stf.cli label-finalize --data data/labeled/indomain/to_label_batch2_ai.csv

# Union with the earlier batch; ids are re-keyed so they cannot collide
uv run python -m stf.cli label-merge \
  r1=data/labeled/indomain/to_label_r1.csv \
  b2=data/labeled/indomain/to_label_batch2_ai.csv \
  --output data/labeled/indomain/labeled_merged.csv

# Train on everything, score only the prior-preserving strata
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/labeled_merged.csv \
  --input-variant title_context --truncation-strategy head_tail \
  --class-weighting inverse_frequency --epochs 5 --folds 5 \
  --eval-strata eval_random baseline_random \
  --output models/experiments/merged__title_context__head_tail
```

| Stratum | Selection | Use |
| --- | --- | --- |
| `eval_random` | uniform over the deduplicated pool; no model or lexicon input | the only stratum valid for an unbiased overall estimate |
| `train_negative` | negative lexicon hit, weighted random draw by cue count, capped per ticker | training only |
| `train_positive` | same, positive side | training only |
| `train_active` | lowest checkpoint confidence | training only; the one model-informed stratum |

Every row records its own `usage` and `model_informed`, so the provenance cannot drift from
how the row was chosen. Lexicon hits only decide which articles a human reads; they never
assign a label.

Enrichment is ordered by the **lexicon**, not by the checkpoint's probabilities. Measured on
all 306 already-labelled rows, a negative cue raises the `NEGATIVE` rate from the 10.1%
corpus base rate to 28.9% (2.9x). The checkpoint's probabilities looked much stronger, but
that was measured on rows it had trained on; re-measured on fold-01's 62 unseen rows the
advantage could not be confirmed (only 16 eligible rows, 7 `NEGATIVE`), so it is not used for
ranking. The lexicon is a fixed rule that never saw a label, so its 2.9x figure is honest.
Expect roughly 25–30% `NEGATIVE` in `train_negative`.

`--eval-strata` keeps the enriched rows in every training fold and folds the holdout only
over the strata that preserve the corpus prior; files predating this design default to
`baseline_random` and stay evaluable. This costs minority-class precision — 59 evaluable
`NEGATIVE` rows give a recall interval of about ±0.128, against ±0.070 if all 194 were
scored — but a wider unbiased interval beats a narrow one computed on rows selected for
being rare. `label-candidates` writes a `*_manifest.json` with the quotas, seed,
eligible-set sizes and inclusion probability of every stratum.

`body_preview` is capped at 400 characters to match the existing rows, so a merged file has a
single model-input length; `body_context` carries 2000 characters purely for the human
annotator and is never a model input.

## Tests

```bash
uv run pytest tests/ -q
```
