# PhoBERT fine-tuning notebook (Colab / Kaggle)

Fine-tune `vinai/phobert-base` into a 3-class sentiment classifier
(NEGATIVE / NEUTRAL / POSITIVE) for Vietnamese financial news.

## Files
- `phobert_finetune_colab.ipynb` - the fine-tuning notebook, runs on Colab/Kaggle.
- `phobert_finetune_colab.py` - the same notebook as a jupytext script (easier to diff/review).
- `training-guide.md` - step-by-step run instructions and troubleshooting.

## Quick start

### Google Colab
1. Upload `phobert_finetune_colab.ipynb`.
2. Runtime > Change runtime type > **T4 GPU**.
3. Run cells in order. If Colab asks to restart after the install cell,
   restart and continue FROM section 3 (don't re-run section 2).

### Kaggle
1. New Notebook > Upload `phobert_finetune_colab.ipynb`.
2. Settings > Accelerator > **GPU** (T4 x2 or P100), and turn **Internet** on.
3. Run cells in order.

See `training-guide.md` for the detailed walkthrough.

## Data

- **CafeF seed** (999 pre-labeled headlines): the notebook downloads it from the public
  repo `209sontung/Vietnamese-stock-article-classification`.
- **In-domain (optional):** if you have labeled the in-domain sample (from
  `stf.sentiment.make_indomain_sample`), save it as `indomain_labeled.csv`
  (`text`, `label` columns) and upload it. The notebook merges it in automatically.

## Version notes (avoid breakage)

The notebook pins `transformers==4.46.3`, `numpy<2`, `pandas<2.3` for stable compatibility
with the torch+CUDA already on Colab/Kaggle. It does NOT reinstall torch. If Kaggle reports
a dependency conflict, turn Internet on and re-run the install cell, or use `--no-deps` for
transformers.

## Output

Every run writes to `runs/phobert-sentiment-<timestamp>/`:
- `best/` - model + tokenizer.
- `manifest.json` - versions, seed, hyperparameters, test metrics (for reproducibility).
- `test_classification_report.txt` / `test_report.json` - metrics to cite in the report.
- `confusion_matrix.png` / `.csv` - ready to drop into the report.
- `train_log_history.csv` - per-epoch loss/metric for debugging.

The `predict_proba` helper then generates 3-class probabilities for the news corpus (next step).
