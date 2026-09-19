# PhoBERT fine-tuning notebook (Colab / Kaggle)

Fine-tune `vinai/phobert-base` into a 3-class sentiment classifier
(NEGATIVE / NEUTRAL / POSITIVE) for Vietnamese financial news.

## Files
- `phobert_finetune.ipynb` - the basic single-split fine-tuning notebook.
- `sentiment_cv.ipynb` - the canonical GPU notebook for the reviewed 1,306-row,
  stratum-aware 5-fold thesis run.
- `training-guide.md` - step-by-step run instructions and troubleshooting.

## Quick start

### Google Colab
1. Upload `phobert_finetune.ipynb` for the basic run, or `sentiment_cv.ipynb` for
   the reviewed 1,306-row 5-fold experiment.
2. Runtime > Change runtime type > **T4 GPU**.
3. Before running `sentiment_cv.ipynb`, make the Kaggle Dataset contain the
   reviewed `labeled_merged.csv`; then run cells in order.

### Kaggle
1. New Notebook > Upload `sentiment_cv.ipynb` and attach
   `phuocthoai/stock-trend-forecasting`.
2. Settings > Accelerator > **GPU** and turn **Internet** on.
3. Ensure the attached dataset has `labeled_merged.csv`, then run cells in order.

See `training-guide.md` for the detailed walkthrough.

## Data

- **CafeF seed** (999 pre-labeled headlines): the notebook downloads it from the public
  repo `209sontung/Vietnamese-stock-article-classification`.
- **In-domain thesis run:** attach a Dataset version containing the reviewed
  `labeled_merged.csv`. It contains 1,306 rows, records `stratum`, `usage` and
  final human-review provenance, and is deliberately not committed to Git.

## Version notes (avoid breakage)

The basic fine-tuning notebook pins `transformers==4.46.3`, `numpy<2`, and
`pandas<2.3`. The canonical `sentiment_cv.ipynb` instead pins the exact stack
recorded by the successful Kaggle CV manifest: platform `torch==2.10.0+cu128`
and `transformers==5.15.1`. It never reinstalls torch; it checks the version
before importing `Trainer` and stops with a fresh-runtime message on mismatch.
If Kaggle's `torchvision` import is broken, the notebook removes `torchvision`
and `timm`, which are not needed for text-only PhoBERT.

The `sentiment-cv` runner keeps only the final `best/` model and manifest for
each fold; transient per-epoch checkpoints are removed after evaluation to fit
Kaggle's working-disk limit. The notebook runs the selected
`title_context + head_tail`, inverse-frequency, five-epoch configuration and
uses `eval_random` plus `baseline_random` only for outer holdouts. Keep its
branch set to the repaired source.

## Output

Every run writes to `runs/phobert-sentiment-<timestamp>/`:
- `best/` - model + tokenizer.
- `manifest.json` - versions, seed, hyperparameters, test metrics (for reproducibility).
- `test_classification_report.txt` / `test_report.json` - metrics to cite in the report.
- `confusion_matrix.png` / `.csv` - ready to drop into the report.
- `train_log_history.csv` - per-epoch loss/metric for debugging.

The `predict_proba` helper then generates 3-class probabilities for the news corpus (next step).
