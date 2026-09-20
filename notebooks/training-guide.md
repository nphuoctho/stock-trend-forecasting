# Chạy notebook huấn luyện và đánh giá PhoBERT trên Colab / Kaggle

Có ba notebook cho ba mục đích:

- `phobert_finetune.ipynb`: huấn luyện PhoBERT cơ bản và xuất checkpoint.
- `sentiment_cv.ipynb`: notebook GPU chính thức cho xác thực chéo 5 lượt trên
  1.306 nhãn trong miền dữ liệu đã rà soát. Nó huấn luyện trên mọi tầng, nhưng
  chỉ đánh giá holdout ngoài trên các tầng giữ phân phối gốc.
- `sentiment_refit.ipynb`: tinh chỉnh checkpoint triển khai trên toàn bộ 1.306 nhãn
  bằng cấu hình đã khóa từ artifact xác thực chéo; không tạo chỉ số kiểm thử mới.

Khi cần chạy lại thực nghiệm luận văn, dùng `sentiment_cv.ipynb`, bật Internet/GPU,
và attach một phiên bản Kaggle Dataset có tệp `labeled_merged.csv` đã được rà soát.
Sau khi artifact CV được xác nhận, dùng `sentiment_refit.ipynb` để tạo checkpoint
cho bước chấm toàn bộ tin.

---

## Part A: Google Colab

### Step 1: Open the notebook
1. Go to https://colab.research.google.com.
2. Upload `sentiment_cv.ipynb` cho thực nghiệm xác thực chéo 5 lượt,
   `sentiment_refit.ipynb` cho checkpoint triển khai đã khóa cấu hình, hoặc
   `phobert_finetune.ipynb` cho lần huấn luyện cơ bản một split.
### Step 2: Enable the GPU (required)
1. Menu **Runtime > Change runtime type**.
2. Hardware accelerator > **T4 GPU** > Save.
3. Without it, section 3 fails on `assert torch.cuda.is_available()`.

### Step 3: Run cells in order
Press Play (or Shift+Enter) on each cell top to bottom:

| Section | Does | Expected |
|---|---|---|
| 1 | Check GPU | Prints "GPU: Tesla T4, 15360 MiB" |
| 2 | Install libraries | ~1-2 min. May show a "RESTART SESSION" button |
| - | **If it asks to RESTART:** restart, then continue FROM SECTION 3 (skip 1-2) | |
| 3 | Confirm versions | transformers 4.46.3, CUDA: True |
| 4 | Download CafeF seed | "CafeF seed downloaded" |
| 5 | Prepare labels | "CafeF seed only: 999 samples", distribution {0:186,1:249,2:564} |
| 6 | Split train/val/test | train~799 val~100 test~100 |
| 7 | Tokenize | Loads the PhoBERT tokenizer |
| 8 | Load model + create RUN_DIR | Downloads phobert-base (~500MB); the "newly initialized" warning is NORMAL |
| 9 | **Train** | ~5-10 min on T4, shows loss/macro_f1 per epoch; writes train_log_history.csv |
| 10 | Evaluate + save report | macro-F1 + per-class + confusion matrix (png/csv) into RUN_DIR |
| 11 | Save model + manifest | model + manifest.json (versions, seed, hyperparameters, metrics) |
| 12 | Package zip | Zips the whole RUN for download |
| 13 | Inference check | Prints a 2x3 probability matrix |

### Step 4: Save the results (important: don't lose your numbers with the session)
The notebook writes every artifact to `runs/phobert-sentiment-<timestamp>/`:
- `train_log.jsonl`: appended per step during training, so a mid-run crash still leaves numbers.
- `train_log_history.csv`: per-epoch loss/metric (debugging + plotting).
- `learning_curve.png`: train loss + val macro-F1 curve, to see convergence/overfitting.
- `test_classification_report.txt` / `test_report.json`: metrics to cite in the report.
- `confusion_matrix.png` / `.csv`: ready to drop into the report.
- `misclassified.csv`: wrong predictions (text/true/pred/probabilities/confidence), sorted by
  confidence, for the qualitative error analysis.
- `test_predictions.csv`: all test predictions, to re-analyze later without rerunning the model.
- `manifest.json`: versions, seed, hyperparameters, metrics (reproducibility evidence).
- `best/`: model + tokenizer.
- `checkpoints/`: per-epoch checkpoints (last 3 kept) for resuming.
- `logs/`: TensorBoard logs.

**To keep results across sessions, pick one:**
- **Best (Colab):** uncomment the 3 Drive-mount lines at the top of section 8 and point
  `BASE_OUT` at a Drive path. Everything saves straight to Drive.
- **Quick:** in section 12, uncomment `files.download(zip_path)` (Colab) to pull the zip down.
  On Kaggle the zip shows up in the Output tab.

### Step 4b: Resume after Colab/Kaggle drops mid-run
If the session dies during training (free-tier timeout, lost network), you don't restart from zero:
1. Make sure the old RUN dir still exists (Drive-mounted, or the Kaggle session is still open).
2. Reopen the notebook, run sections 1-7 as usual (reload libs + data, keep the SAME seed/hyperparameters).
3. In **section 8**, set `RESUME_RUN_DIR = "runs/phobert-sentiment-<old timestamp>"`.
4. Continue with sections 8-9. The Trainer loads the latest checkpoint and continues; logs append
   to the same RUN_DIR.

---

## Part B: Kaggle

### Step 1: Create or import the notebook
1. Go to https://www.kaggle.com/code > New Notebook.
2. Import `sentiment_cv.ipynb` cho thực nghiệm xác thực chéo, `sentiment_refit.ipynb`
   cho checkpoint triển khai, hoặc `phobert_finetune.ipynb` cho lần huấn luyện cơ bản.
3. Đính kèm phiên bản dataset `phuocthoai/stock-trend-forecasting` có tệp
   `labeled_merged.csv`. Notebook kiểm tra đủ 1.306 nhãn đã rà soát trước khi
   bắt đầu huấn luyện.

### Step 2: Enable GPU + Internet
1. Open the right panel **Settings**.
2. Set **Accelerator** to **GPU T4 x2** (or P100).
3. Set **Internet** to **On** so the notebook can clone the repository and
   download PhoBERT when needed.

`sentiment_cv.ipynb` ghi artifact xác thực chéo vào `/kaggle/working`.
`sentiment_refit.ipynb` ghi checkpoint cuối và ZIP vào `/kaggle/working`.
Tải output hoặc lưu phiên bản notebook mới sau khi chạy.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `assert torch.cuda.is_available()` fails in section 3 | GPU not on | Enable GPU (Step 2), rerun from section 1 |
| Section 2 asks to RESTART | Colab updated numpy/pandas | Restart, continue FROM section 3 (don't rerun 2) |
| `NameError: torch not defined` | Ran a cell out of order | Rerun from section 3 |
| Section 8 "newly initialized weights" warning | Normal (fresh classification head) | Ignore, that's expected when fine-tuning |
| Training very slow (>30 min) | Running on CPU, not GPU | Section 3 must print "CUDA: True" |
| `wget` fails in section 4 | Kaggle Internet off | Turn Internet on in Settings |
| `operator torchvision::nms does not exist` hoặc import `Trainer` thất bại | `torchvision` không tương thích với `torch` đã cài | Khởi tạo phiên mới và chạy bootstrap của `sentiment_cv.ipynb` hoặc `sentiment_refit.ipynb`; hai notebook giữ nguyên stack huấn luyện văn bản và gỡ `torchvision` lỗi |
| Hết dung lượng khi lưu model | Lượt CV giữ nhiều checkpoint hoặc thư mục cũ còn tồn tại | Khởi tạo phiên Kaggle mới; `sentiment_refit.ipynb` chỉ lưu một checkpoint cuối và một ZIP |
| Low macro-F1 (<0.5) | Normal for a small, imbalanced seed | Add in-domain labels, or accept it and report honestly |

---

## After training

1. **Record the macro-F1** (section 10) in your journal/plan.
2. **Save the model** to Drive or download the zip (section 11).
3. **Next step:** use the model to generate sentiment probabilities for the whole news corpus
   (section 12 extended): upload `articles.parquet`, run `predict_proba` on the title (or
   title+body), and save the output for Phase 3 (ticker-day features).

## Thực nghiệm luận văn hiện tại

Chạy đúng `sentiment_cv.ipynb`, không chạy lại ma trận 306 mẫu cũ. Cấu hình đã
khóa là:

```bash
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/labeled_merged.csv \
  --input-variant title_context \
  --truncation-strategy head_tail \
  --class-weighting inverse_frequency \
  --folds 5 --epochs 5 --batch-size 16 --seed 42 \
  --eval-strata eval_random baseline_random \
  --output models/experiments/merged__title_context__head_tail
```

Năm outer holdout cộng lại gồm 656 dòng thuộc các tầng bảo toàn phân phối gốc,
tương đương khoảng 131 dòng mỗi fold. Các dòng làm giàu lớp thiểu số luôn ở
train; validation được rút từ tầng đánh giá, nhưng có kích thước bằng 10\% của
toàn bộ outer train. Không dùng holdout hoặc các tầng làm giàu để chọn checkpoint.
Báo cáo `macro_f1`, `balanced_accuracy`, F1 và recall từng lớp; 59 mẫu NEGATIVE
ở toàn bộ holdout đòi hỏi diễn giải khoảng tin cậy thận trọng.

Sau khi artifact `cv_results.json` được kiểm tra, tạo checkpoint triển khai bằng:

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

Lệnh bắt buộc mã băm dữ liệu, cấu hình và kích thước mẫu phải khớp artifact CV. Nó
huấn luyện trên toàn bộ nhãn đã duyệt với số epoch cố định, không rút validation hoặc
outer holdout nên manifest không chứa chỉ số kiểm thử. Sau khi tải checkpoint về, chạy
`score-news`, hai lệnh `forecast` và `forecast-compare` theo `README.md`.

## Cải thiện tùy chọn
- Nếu mở rộng nhãn, giữ một tầng lấy mẫu ngẫu nhiên độc lập cho đánh giá và đưa
  các tầng làm giàu chỉ vào huấn luyện; sau rà soát, hợp nhất vào
  `labeled_merged.csv`.
- Có thể cài `py_vncorenlp` để tách từ trước khi mã hóa, vì PhoBERT được huấn
  luyện trên văn bản đã tách từ.
