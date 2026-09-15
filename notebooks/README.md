# PhoBERT fine-tuning notebook (Colab / Kaggle)

Fine-tune `vinai/phobert-base` into a 3-class sentiment classifier
(NEGATIVE / NEUTRAL / POSITIVE) for Vietnamese financial news.

## Files
- `phobert_finetune.ipynb` - the fine-tuning notebook, runs on Colab/Kaggle.
- `sentiment_cv.ipynb` - the 5-fold input/truncation comparison notebook.
- `training-guide.md` - step-by-step run instructions and troubleshooting.

## Quick start

### Google Colab
1. Upload `phobert_finetune.ipynb` for the basic run, or `sentiment_cv.ipynb` for
   the 5-fold input/truncation experiments.
2. Runtime > Change runtime type > **T4 GPU**.
3. Run cells in order. If Colab asks to restart after the install cell,
   restart and continue FROM section 3 (don't re-run section 2).

### Kaggle
1. New Notebook > Upload `phobert_finetune.ipynb` or `sentiment_cv.ipynb`.
2. Settings > Accelerator > **GPU** (T4 x2 or P100), and turn **Internet** on.
3. Run cells in order.

See `training-guide.md` for the detailed walkthrough.

## Data

- **CafeF seed** (999 pre-labeled headlines): the notebook downloads it from the public
  repo `209sontung/Vietnamese-stock-article-classification`.
- **In-domain (required for the thesis matrix):** run
  `stf.sentiment.make_indomain_sample`, have one human reviewer label at least 301
  articles, and provide `to_label_r1.csv` with `title`, `body_preview`, `published_at`,
  `url`, and final `label` columns. Do not upload preliminary labels.

## Version notes (avoid breakage)

The fine-tuning notebook pins `transformers==4.46.3`, `numpy<2`, and `pandas<2.3`
for stable compatibility with the torch+CUDA already on Colab/Kaggle. The
`sentiment_cv.ipynb` bootstrap uses the project-compatible text-training stack
(`transformers==5.15.1`, `tokenizers>=0.22,<=0.23.0`) and does not reinstall
torch. If Kaggle's `torchvision` import is broken, it removes `torchvision` and
`timm`, which are not needed for text-only PhoBERT training. If Kaggle reports a
dependency conflict, turn Internet on and run the bootstrap cell in a fresh
session.
The CV runner keeps only the final `best/` model and manifest for each fold;
transient per-epoch checkpoints are removed after evaluation to fit Kaggle's
working-disk limit. The notebook includes the epoch count in each output
directory name, so smoke-test and final runs do not share artifacts.
The bootstrap also verifies that the cloned `develop` branch contains the
disk-safe checkpoint settings before starting CV; update that branch before
running the uploaded notebook.

## Output

Every run writes to `runs/phobert-sentiment-<timestamp>/`:
- `best/` - model + tokenizer.
- `manifest.json` - versions, seed, hyperparameters, test metrics (for reproducibility).
- `test_classification_report.txt` / `test_report.json` - metrics to cite in the report.
- `confusion_matrix.png` / `.csv` - ready to drop into the report.
- `train_log_history.csv` - per-epoch loss/metric for debugging.

The `predict_proba` helper then generates 3-class probabilities for the news corpus (next step).
