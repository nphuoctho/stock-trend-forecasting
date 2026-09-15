# Chạy notebook huấn luyện và đánh giá PhoBERT trên Colab / Kaggle

Có hai notebook cho hai mục đích:

- `phobert_finetune.ipynb`: huấn luyện PhoBERT cơ bản và xuất checkpoint.
- `sentiment_cv.ipynb`: chạy so sánh biến thể đầu vào/cắt token với 5-fold
  cross-validation và lưu kết quả vào Google Drive.

Phần dưới đây hướng dẫn notebook huấn luyện cơ bản; khi cần lấy kết quả so sánh
theo góp ý GVHD, dùng `sentiment_cv.ipynb`.

---

## Part A: Google Colab

### Step 1: Open the notebook
1. Go to https://colab.research.google.com.
2. Upload `sentiment_cv.ipynb` for the 5-fold input/truncation experiments,
   or `phobert_finetune.ipynb` for the basic single-split training run.
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
2. Import `sentiment_cv.ipynb` for the cross-validation experiment, or
   `phobert_finetune.ipynb` for the basic training run.
3. In the notebook, attach the dataset
   `phuocthoai/stock-trend-forecasting`; the code finds `labeled.csv` and
   `cafef_seed.csv` under `/kaggle/input/stock-trend-forecasting`.

### Step 2: Enable GPU + Internet
1. Open the right panel **Settings**.
2. Set **Accelerator** to **GPU T4 x2** (or P100).
3. Set **Internet** to **On** so the notebook can clone the repository and
   download PhoBERT when needed.

### Step 3: Run cells in order
The CV notebook writes results to `/kaggle/working`; download the output or
save a new notebook version after the run.

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
| Low macro-F1 (<0.5) | Normal for a small, imbalanced seed | Add in-domain labels, or accept it and report honestly |

---

## After training

1. **Record the macro-F1** (section 10) in your journal/plan.
2. **Save the model** to Drive or download the zip (section 11).
3. **Next step:** use the model to generate sentiment probabilities for the whole news corpus
   (section 12 extended): upload `articles.parquet`, run `predict_proba` on the title (or
   title+body), and save the output for Phase 3 (ticker-day features).

## Thực nghiệm bắt buộc theo góp ý GVHD

Không chọn cấu hình đầu vào dựa trên một lần chia ngẫu nhiên. Với tệp
Vietstock đã gán nhãn, chạy `sentiment-cv` cho từng cấu hình:

```bash
uv run python -m stf.cli sentiment-cv \
  --data data/labeled/indomain/labeled.csv \
  --input-variant title \
  --truncation-strategy head \
  --folds 5 --epochs 3 \
  --output models/experiments/title__head
```

Lặp lại với ba biến thể `title`, `context`, `title_context` và ba cách cắt
`head`, `tail`, `head_tail`. PhoBERT có tối đa 256 token; phần cắt được thực
hiện sau khi mã hóa token và trước khi thêm token đặc biệt. `head_tail` giữ
hai vùng đầu và cuối, không phải cắt chuỗi theo số ký tự.

Trong mỗi fold, outer holdout chỉ dùng để đánh giá. Một phần 10% của outer
train được dùng làm validation để chọn checkpoint. Không gộp các kết quả
holdout vào quá trình chọn mô hình. So sánh `macro-F1` trung bình và độ lệch
chuẩn qua năm fold; báo cáo thêm accuracy, balanced accuracy và F1 từng lớp.

Tập CafeF hiện chỉ có tiêu đề. Vì vậy, các cấu hình `context` và
`title_context` trên CafeF sẽ không phải phép so sánh nội miền hợp lệ. Cần
hoàn tất gán nhãn tệp Vietstock có `title`, `body` và `label` trước khi chạy
ma trận đầy đủ. Không dùng bài viết hoặc nhãn phát sinh từ giai đoạn dự báo
để huấn luyện mô hình cảm xúc.

## Optional quality improvements
- **In-domain labels:** run `stf.sentiment.make_indomain_sample` locally, label ~300 articles
  by the guideline, save `indomain_labeled.csv`, and upload it (section 5 merges it).
- **Word segmentation:** install `py_vncorenlp` to segment words before tokenizing (PhoBERT was
  trained on segmented text), which usually improves macro-F1.
