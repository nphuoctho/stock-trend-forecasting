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
uv run python -m stf.cli prices --end 2026-09-22  # extend past the study window (live mode)

# Crawl news (minute timestamp + title + body)
uv run python -m stf.cli news                 # full run (long)
# `--refresh` re-crawls listings and repairs cached listing dates after parser changes.
# Without `--refresh`, cached years are reused and only the final year is re-walked,
# so a daily run picks up fresh articles without re-crawling the archive.
uv run python -m stf.cli news --end 2026-09-22  # include the current year
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

Không dùng nhãn hoặc giá tương lai để tạo đầu vào cảm xúc. Khi đánh giá cuốn chiếu, phải
tinh chỉnh lại trong từng cửa sổ chỉ bằng nhãn quá khứ hoặc dùng một điểm kiểm đóng băng
được huấn luyện trước ngày kiểm thử đầu tiên. Tập CafeF chỉ có tiêu đề nên không đủ để kết
luận riêng về `context`; cần dùng tệp Vietstock đã gán nhãn có nội dung bài viết.

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

`score-news` đọc dạng đầu vào đã khóa trong manifest của điểm kiểm; cờ
`--input-variant` phải khớp chính xác để không chấm tiêu đề bằng điểm kiểm đã huấn luyện
trên tiêu đề--ngữ cảnh.

```bash
uv run python -m stf.cli score-news \
  --model-dir models/sentiment/merged-refit/best \
  --input-variant title_context \
  --context-chars 400 \
  --batch-size 64 \
  --output data/processed/news_sentiment_merged.parquet
```

`score-news` cũng ghi `data/processed/news_sentiment_merged.manifest.json` theo lược đồ
phiên bản 2. Sidecar ghi số dòng, tổng và kiểm tra xác suất, mã băm parquet/điểm kiểm,
fingerprint bất biến theo thứ tự của nội dung đã chấm sau `--limit`, cấu hình suy luận hiệu
lực và phiên bản môi trường. `forecast` chỉ nhận parquet có sidecar khớp mã băm, rồi chép
mã băm sidecar, điểm kiểm và cấu hình suy luận vào `forecast_results.json`.

## Phase 4: forecasting experiment

`forecast-smoke` chỉ chứng minh mã chạy được. `forecast` tải parquet giá thực tế, tùy chọn
một parquet từ `score-news`, rồi chạy thang đầy đủ trên các cửa sổ cuốn chiếu.

Để báo cáo kết quả ngoài mẫu, parquet cảm xúc phải được tạo bởi điểm kiểm phù hợp thời
điểm của từng cửa sổ. Một điểm kiểm tinh chỉnh trên toàn bộ tệp nhãn chỉ phù hợp cho phân
tích hồi cứu; nó không được dùng để kết luận hiệu quả dự báo trên các ngày có nhãn tương lai.
`sentiment-refit --before-date` tạo checkpoint đóng băng trước ngày kiểm thử đầu tiên và
`forecast` ghi `point_in_time` vào `forecast_results.json` (xem `notebooks/training-guide.md`).

```bash
# Đối chứng trung tính ghép cặp: giữ nguyên bài tin, thời điểm và khối lượng tin,
# chỉ thay ba xác suất thành prior trung tính. Hai lần chạy vì thế chỉ khác
# thông tin phân cực cảm xúc.
# `--panel-end` khóa cửa sổ đánh giá khi dữ liệu giá đã kéo dài qua cửa sổ nghiên cứu.
uv run python -m stf.cli forecast \
  --neutral-news-sentiment data/processed/news_sentiment_merged.parquet \
  --panel-end 2025-12-31 \
  --windows 5 --test-size 60 --val-size 60 --epochs 40 --patience 6 \
  --seeds 42 43 44 --output outputs/forecast_merged_control

# Thang đầy đủ với cảm xúc; hai nhánh dùng cùng cửa sổ, hạt giống và hàng kiểm thử.
uv run python -m stf.cli forecast \
  --news-sentiment data/processed/news_sentiment_merged.parquet \
  --panel-end 2025-12-31 \
  --windows 5 --test-size 60 --val-size 60 --epochs 40 --patience 6 \
  --seeds 42 43 44 --output outputs/forecast_merged_sentiment

# Tách ảnh hưởng kiến trúc và giá trị thông tin.
uv run python -m stf.cli forecast-compare \
  --real outputs/forecast_merged_sentiment \
  --control outputs/forecast_merged_control \
  --output outputs/forecast_merged_sentiment/information_gain.json
```

The ladder is `majority`, `random`, `logreg_price`, `logreg_price_sentiment`,
`lstm_price`, `lstm_price_sentiment`, `tft_price`, `tft_price_sentiment`. The TFT
arms use `TemporalFusionClassifier`, a minimal TFT-style encoder (per-feature
embeddings, variable selection, LSTM encoder, multi-head self-attention, gated
residuals) trained for 3-class classification on the same windows. Per window the
command refits the trend thresholds
and both feature scalers on training rows only, selects each LSTM checkpoint on that
window's validation macro-F1, and scores every arm once on the same test rows. Results
are averaged over `--seeds`. The sentiment contribution is reported twice: as a paired
per-window delta bootstrapped over the 5 windows, and as a bootstrap over the ~300 test
**dates**. In the date-block interval all tickers of a date resample together and dates
are resampled *inside their own window*, so the interval brackets the same
window-averaged quantity the point estimate reports. Prefer the date-block interval; the
window interval has only 5 blocks and is coarse enough to exclude zero by accident.

`date_block_bootstrap` resamples dates inside their own window and averages the
per-window metrics, so it brackets the same window-averaged quantity
`information_gain` reports. An earlier version pooled every window into one confusion
matrix. That pooled delta is a legitimate estimate in its own right and it came with
its own interval, but it is a *different* estimand: macro-F1 is non-linear in the
confusion matrix, so the pooled delta does not equal the mean of the per-window
deltas, and the two had to be read as a pair of separate results. Reporting one
estimand with one matching interval is simpler to state and to defend; it is not
evidence that the pooled figure was wrong.

Two limits apply to both intervals. Dates are drawn independently, so neither is a
serial block bootstrap and neither models day-to-day dependence. And
`one_sided_p_le_zero` is reported for convenience only: the direction was not fixed
before the results were seen, so it cannot be used to claim significance. Treat the
two-sided interval as the reportable quantity and correct for the number of arms and
runs compared.

### Lựa chọn nhãn: lợi suất thô hay lợi suất vượt trội

`--target raw` (mặc định) gán nhãn theo lợi suất phiên kế tiếp của chính mã đó.
`--target excess` gán nhãn theo phần lợi suất vượt trên trung bình đồng hạng của cùng
phiên. Trên 2020--2025, mười mã nghiên cứu có tương quan lợi suất ngày trung bình theo
cặp là 0,42 và $R^2$ trung bình 0,48 so với trung bình đồng hạng: gần một nửa biến động
hằng ngày là nhịp chung của thị trường, thứ mà tin riêng của doanh nghiệp không giải
thích được. Mục tiêu vượt trội loại bỏ thành phần chung nên đo đúng câu hỏi "tin của mã
này có báo trước việc nó chạy nhanh hơn rổ hay không".

Đây là đổi nhãn chứ không phải đổi đặc trưng: trung bình đồng hạng được trừ đi cũng chỉ
biết được vào đúng ngày mục tiêu, nên không có thông tin nào xuất hiện sớm hơn trước.
Ngưỡng tam phân vẫn khớp riêng trên phần huấn luyện của từng cửa sổ.

### Chân trời dự báo

`--horizon 1` (mặc định) là trường hợp khó nhất: điều mà bài tin hàm ý phải hiện ra trong
đúng một nhịp đóng cửa sang đóng cửa. `--horizon 3` hoặc `--horizon 5` kiểm tra xem kết
quả rỗng ở `h=1` là đặc thù của nhịp thời gian đó hay là tính chất của tín hiệu.

```bash
uv run python -m stf.cli forecast --news-sentiment data/processed/news_sentiment_pit.parquet \
  --panel-end 2025-12-31 --horizon 5 --seeds 42 43 44 --output outputs/forecast_h5_pit_sentiment
```

Cái giá phải trả là **nhãn chồng lấn**: ở `h=5`, hai dòng liên tiếp dùng chung bốn trên năm
phiên, nên số quan sát độc lập hữu hiệu chỉ còn khoảng `n/h`. Ước lượng điểm vẫn dùng được,
nhưng khoảng tin cậy tính như thể các dòng độc lập sẽ **hẹp hơn mức bằng chứng cho phép**.
Luôn ghi chân trời cạnh mọi con số lấy từ lượt chạy kiểu này, và đọc khoảng tin cậy của
`h>1` như một chỉ báo, không phải một phép kiểm.

`forecast-compare` exists because comparing the two-branch arm against the single-branch
price model conflates the added branch with news presence and volume. The neutral control
keeps article timing, news volume, and probability-independent features fixed, replacing
only the probability vector. Consequently,
`architecture_and_news_presence_volume_effect + information_gain = naive_delta` exactly;
only `information_gain` isolates polarity information.

Trước khi tính chênh lệch, `forecast-compare` bắt buộc hai lần chạy có cùng nguồn
tin đã chấm, mã băm dữ liệu giá, cấu hình, ngày kiểm thử và khóa dự đoán
`(window, seed, ticker, target_date, y_true)`. Nó cũng yêu cầu prediction và metric của
nhánh chỉ giá trùng khớp, nên không thể ghép một artifact cũ, nguồn tin khác hoặc lượt
chạy bị lệch vào phép đo giá trị thông tin.

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

### Tương quan hạng của tín hiệu, không qua mô hình

Một phép cắt bỏ cho kết quả rỗng không phân biệt được "bộ ước lượng yếu" với "tín hiệu
yếu". `forecast-ic` bổ sung một mảnh bằng chứng: tương quan hạng Spearman giữa điểm cảm
xúc theo ngày và lợi suất thực hiện, **không khớp bất kỳ mô hình nào**, nên nó không bị
lựa chọn kiến trúc làm nhiễu.

```bash
uv run python -m stf.cli forecast-ic \
  --news-sentiment data/processed/news_sentiment_pit.parquet \
  --output outputs/signal_ic_pit.json
```

**Nó không chứng minh điều gì.** Hệ số Spearman gần 0 chỉ bác bỏ liên hệ **đơn điệu** của
**đúng điểm số vô hướng này**. Nó không phải cận trên của khả năng dự báo và không phải
thước đo lượng thông tin: quan hệ $y = x^2$ với $x$ đối xứng dự báo được hoàn hảo nhưng
tương quan hạng bằng 0. Nó cũng không nói gì về một cách tổng hợp khác, một hiệu ứng có
điều kiện hay tương tác, một chân trời dài hơn, hay một cách đo cảm xúc tốt hơn. Đọc kết
quả rỗng ở đây đúng như nó là: *không phát hiện được liên hệ đơn điệu cho điểm số này ở
chân trời này*. Các phép đối chiếu chưa được đăng ký trước, nên chúng mang tính thăm dò.

Báo cáo đối chiếu hai cặp. `same` so với `next`: điểm số đồng biến với chính phiên của nó
nhưng không với phiên sau là **phù hợp với** giả thuyết tin đã phản ánh vào giá lúc đóng
cửa --- phù hợp với, chứ không phải chứng minh: cùng một hình mẫu cũng xuất hiện khi giá
chi phối giọng điệu bài viết (nhân quả ngược), hoặc khi điểm số đơn giản là quá nhiễu để
còn sót lại sau một ngày pha loãng nữa. `raw` so với `excess` tách nhịp chung của thị
trường khỏi phần riêng của mã. Khoảng tin cậy lấy mẫu lặp theo trọn phiên và coi các phiên
là hoán vị được, nên **không** mô hình hóa phụ thuộc chuỗi.

### Chẩn đoán kinh tế (không phải backtest giao dịch được)

`macro_f1` không cho biết biên lợi thế lớn hay nhỏ tính bằng điểm cơ bản.
`forecast-backtest` quy đổi dự đoán đã lưu thành một sổ mua/bán khống để đọc con số đó.

```bash
uv run python -m stf.cli forecast-backtest \
  --run outputs/forecast_pit_sentiment \
  --arm lstm_price_sentiment \
  --cost-bps 20
```

> **Cảnh báo ràng buộc: kết quả này không giao dịch được.** Đặc trưng của ngày `t` gồm
> `close_t` và mọi bài tin tới mốc 15:00 của `t`. Sổ lệnh vào lệnh tại `close_t` --- đúng
> cái giá mà tín hiệu vừa dùng, và chỉ biết được sau khi phiên đã đóng. Đây là cận trên
> dưới giả định khớp lệnh hoàn hảo, tức thời, không trượt giá. Một phương án giao dịch
> được phải vào lệnh từ phiên mở cửa kế tiếp trở đi và sẽ mất phần biến động qua đêm.

Sổ lệnh **không trung hòa thị trường**. Khi cả hai vế cùng có lệnh thì trọng số triệt tiêu;
khi chỉ một vế có lệnh thì sổ mang trạng thái một chiều --- vế sống bị giảm một nửa nhưng
rủi ro thị trường vẫn còn. `mean_abs_net_exposure` và `one_sided_fraction` đo đúng mức vi
phạm đó; trên lượt point-in-time có 23% số phiên một chiều.

Lợi suất thực hiện tính lại từ tệp giá, không suy ngược từ nhãn. Chi phí tính trên vòng
quay so với **vị thế đã trôi giá**, không phải so với trọng số mục tiêu hôm trước: giữ
nguyên mục tiêu vẫn phải cân bằng lại khi giá đã chạy. Báo cáo kèm hai mốc so sánh khác
nhau: `equal_weight_rebalanced` (đặt lại tỷ trọng đều mỗi phiên) và `buy_and_hold` (mua một
lần rồi để trôi) --- chúng là hai sản phẩm khác nhau.

Đọc kết quả phải kèm ba cảnh báo. 300 phiên là quá ít để tách các mức Sharpe gần nhau:
chạy `--arm random` và một null hoán vị nhãn trong từng phiên trước khi diễn giải. Trước
phí và sau phí là hai kết luận khác nhau. Và một phần lợi suất gộp có thể đến từ trạng thái
ròng một chiều, nên hãy hồi quy chuỗi lợi suất sổ lệnh lên lợi suất rổ để tách alpha.

## Phase 5: results dashboard

`webapp` serves a read-only dashboard over the run directories under `outputs/`
(any directory containing `forecast_results.json`):

```bash
uv run python -m stf.cli webapp --port 8000
```

The API lives under `/api` (`/api/runs`, per-run `summary`, `metrics`,
`predictions`, `stratified`, `information-gain`; interactive docs at
`/api/docs`). The React frontend in `web/` is built once with
`cd web && bun install && bun run build` and then served by the same process;
during frontend development run `bun run dev` in `web/` for the Vite dev server
with an `/api` proxy.

## Phase 6: daily prediction (live serving)

The walk-forward experiment discards every trained model; `forecast-refit` freezes one
arm — label thresholds, both feature scalers, and per-seed weights — into
`models/forecast/<arm>/` so a daily job can score the latest session without refitting:

```bash
# one-time: freeze the serving arms (sentiment arms need the scored-news parquet)
uv run python -m stf.cli forecast-refit --arm lstm_price_sentiment \
  --news-sentiment data/processed/news_sentiment_merged.parquet \
  --model-dir models/forecast/lstm_price_sentiment
uv run python -m stf.cli forecast-refit --arm lstm_price \
  --model-dir models/forecast/lstm_price
```

`forecast-predict` scores the last `seq_len` sessions per ticker and writes
`outputs/live/<arm>/predictions_<date>.parquet` plus `latest.parquet`. Every dated
file is stamped with `issued_at` and the checkpoint manifest hash, and a re-run
that would change an issued file is refused — issued predictions are an audit
trail, not a cache. `forecast-resolve` joins stored predictions with realized
next-session labels into `resolved.parquet` and marks each row `prospective`
only when it was issued **strictly before the target session's 09:00 opening
auction**; rows issued once that session could already move are kept but excluded
from the live track record. The stricter `issued_same_session` flag is reported
alongside it:

```bash
uv run python -m stf.cli forecast-predict \
  --model-dir models/forecast/lstm_price_sentiment \
  --news-sentiment data/processed/news_sentiment_merged.parquet \
  --output-dir outputs/live/lstm_price_sentiment
uv run python -m stf.cli forecast-resolve \
  --model-dir models/forecast/lstm_price_sentiment \
  --predictions-dir outputs/live/lstm_price_sentiment \
  --news-sentiment data/processed/news_sentiment_merged.parquet
```

`score-news --incremental` appends only articles not already present in the output
parquet, so the daily job scores just the new crawl instead of the full archive.
`scripts/daily-forecast.sh` chains prices → news → score-news → predict → resolve
for all five arms (`$ARMS` overridable). Each run tees to
`logs/daily-<date>.log` and writes `outputs/live/last_run.json` (finish time,
exit code, failed step).

**Schedule it in the morning, not after the close.** The price provider publishes
a session's close only on the following day, so at 15:30 on day D the freshest
close available is still D-1 and the job would be "predicting" a session that has
already traded — every row lands as a replay and the live track record stays
empty. Running before the opening auction uses the overnight publication of D-1
and targets D, which has not moved yet:

```cron
0 8 * * 1-5  /path/to/stock-trend-forecasting/scripts/daily-forecast.sh
```

A crontab line only fires when a cron daemon is running, which is not the default
on every distribution. `scripts/systemd/` carries an equivalent user timer that
needs no root and, with lingering enabled, runs while logged out:

```bash
cp scripts/systemd/stf-daily-forecast.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now stf-daily-forecast.timer
systemctl --user list-timers stf-daily-forecast.timer   # confirm the next run
loginctl enable-linger "$USER"                          # if Linger=no
```

The timer is deliberately not `Persistent`: catching up a missed run in the
afternoon would issue a prediction for a session already in progress, adding a
replay to the audit trail and nothing to the track record.

`forecast-predict` prints a warning whenever an issuance cannot count as
prospective, and `forecast-resolve` reports the prospective/replayed split of
every run, so an empty track record is visible the day it happens rather than
weeks later.

The dashboard exposes the results at `/api/live/latest` (per-arm signals for the
newest session), `/api/live/history` (prospective vs replayed resolved rows,
pending count, per-date accuracy over prospective rows only), and
`/api/live/status` (last job outcome plus each arm's data-through date).


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
