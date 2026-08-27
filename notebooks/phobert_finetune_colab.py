# %% [markdown]
# # Fine-tune PhoBERT for financial-news sentiment classification (Colab / Kaggle)
#
# This notebook fine-tunes `vinai/phobert-base` into a 3-class sentiment classifier
# (NEGATIVE / NEUTRAL / POSITIVE) for Vietnamese financial news, for the project
# "Forecasting stock price trends from news sentiment".
#
# **Run on GPU** (Colab: Runtime > Change runtime type > T4 GPU; Kaggle: Settings > Accelerator > GPU).
#
# It installs the exact tested library versions to avoid API-compatibility breakage.

# %% [markdown]
# ## 1. Check environment and GPU

# %%
import sys
import platform
print("Python:", sys.version.split()[0], "|", platform.platform())

# Check the GPU before installing, so we know whether training is worth it.
import subprocess
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    print("GPU:", out.decode().strip())
except Exception:
    print("WARNING: no GPU detected. Fine-tuning on CPU will be very slow.")
    print("Colab: Runtime > Change runtime type > T4 GPU. Kaggle: Settings > Accelerator > GPU.")

# %% [markdown]
# ## 2. Install libraries (pin versions to avoid API breakage)
#
# IMPORTANT: this notebook is SELF-CONTAINED and does not depend on the `stf` code.
# We pin `transformers` 4.46.3 because it's stable on Colab/Kaggle and exposes every
# argument used below (`eval_strategy`, `warmup_ratio`, `fp16`). We do NOT reinstall
# `torch`, to keep the torch+CUDA build that Colab/Kaggle ships (overwriting it easily
# breaks CUDA).
#
# Kaggle note: if you hit dependency conflicts on overwrite, add `--no-deps` for
# `transformers`/`accelerate`, or enable Internet in Settings.

# %%
import subprocess, sys

# Pin numpy < 2 for a reliable ABI match with transformers 4.46 and Colab/Kaggle's torch.
PKGS = [
    "transformers==4.46.3",   # stable; has eval_strategy/warmup_ratio/fp16
    "tokenizers>=0.20,<0.22",
    "accelerate==1.2.1",
    "sentencepiece==0.2.0",
    "protobuf<5",             # some environments need it for tokenizer/sentencepiece
    "scikit-learn>=1.3",
    "pandas>=2.0,<2.3",
    "numpy<2",                # avoid ABI clash with this torch/transformers build
    "matplotlib>=3.5",        # confusion matrix plots
    "openpyxl>=3.1",          # read raw_data.xlsx
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], check=True)
print("Install done. If Colab/Kaggle asks to RESTART the runtime, restart it")
print("then continue FROM SECTION 3 (do not rerun section 2).")

# %% [markdown]
# ## 3. Confirm versions after install

# %%
import torch, transformers, sklearn, pandas, numpy
print("torch        :", torch.__version__, "| CUDA:", torch.cuda.is_available())
print("transformers :", transformers.__version__)
print("sklearn      :", sklearn.__version__)
print("pandas       :", pandas.__version__)
print("numpy        :", numpy.__version__)
assert torch.cuda.is_available(), "GPU required. Enable the accelerator and rerun."

# %% [markdown]
# ## 4. Get the `stf` code and label data
#
# Two options:
# - **A. Clone the repo** (if it's public or you've set up access).
# - **B. Download manually**: `cafef_seed.csv` and (optionally) in-domain labels, then upload.
#
# Default is option B (no repo permissions needed). Uncomment option A to clone.

# %%
# --- Option A: clone the repo (uncomment to use) ---
# !git clone -b feature/data-pipeline-and-phobert https://github.com/nphuoctho/stock-trend-forecasting.git
# %cd stock-trend-forecasting
# import sys; sys.path.insert(0, "src")

# --- Option B: download the public CafeF seed directly (no repo needed) ---
import subprocess, sys
subprocess.run([
    "wget", "-q", "-O", "cafef_seed_raw.xlsx",
    "https://raw.githubusercontent.com/209sontung/Vietnamese-stock-article-classification/main/Dataset/raw_data.xlsx"
], check=True)
print("Downloaded CafeF seed (raw_data.xlsx).")

# %% [markdown]
# ## 5. Prepare label data
#
# Raw label mapping: 1=NEGATIVE, 2=NEUTRAL, 3=POSITIVE. If you have a labeled in-domain
# set (columns `text`, `label`), merge it here to improve in-domain quality.

# %%
import pandas as pd

LABELS = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL2ID = {n: i for i, n in enumerate(LABELS)}
RAW2LABEL = {1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE"}

seed = pd.read_excel("cafef_seed_raw.xlsx").rename(columns={"title": "text"})
seed["label"] = seed["label"].map(RAW2LABEL)
seed = seed.dropna(subset=["text", "label"]).drop_duplicates("text")[["text", "label"]]

# (Optional) merge in-domain labels if you've uploaded to_label_done.csv
import os
if os.path.exists("indomain_labeled.csv"):
    ind = pd.read_csv("indomain_labeled.csv")
    ind = ind.dropna(subset=["text", "label"])[["text", "label"]]
    df = pd.concat([seed, ind], ignore_index=True).drop_duplicates("text")
    print(f"Merged seed ({len(seed)}) + in-domain ({len(ind)}) = {len(df)}")
else:
    df = seed
    print(f"Using CafeF seed only: {len(df)} samples")

df["label_id"] = df["label"].astype(str).str.upper().str.strip().map(LABEL2ID)
assert df["label_id"].notna().all(), "Found unknown labels outside NEGATIVE/NEUTRAL/POSITIVE"
df["label_id"] = df["label_id"].astype(int)
print("Class distribution:", df["label_id"].value_counts().sort_index().to_dict())

# %% [markdown]
# ## 6. Split train / val / test (stratified by class, fixed seed)

# %%
from sklearn.model_selection import train_test_split

SEED = 42
train_df, tmp = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=SEED)
val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label_id"], random_state=SEED)
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# %% [markdown]
# ## 7. Tokenize with PhoBERT (max 256 tokens)
#
# Note: PhoBERT was trained on word-segmented text (VnCoreNLP). Here we tokenize raw
# text for simplicity and reproducibility. To improve quality, add a word-segmentation
# step (py_vncorenlp) before the tokenizer.

# %%
import torch  # ensure torch is available even when running this cell alone
from transformers import AutoTokenizer

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN, padding=False)
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item

ds_train = SentimentDataset(train_df["text"], train_df["label_id"])
ds_val   = SentimentDataset(val_df["text"], val_df["label_id"])
ds_test  = SentimentDataset(test_df["text"], test_df["label_id"])
print("Tokenized:", len(ds_train), "train /", len(ds_val), "val /", len(ds_test), "test")

# %% [markdown]
# ## 8. Load the model and configure training
#
# Every artifact from this run (checkpoints, logs, metrics, figures) goes into its own
# timestamped RUN directory, for reproducibility and pulling numbers into the report.
# On Colab, mount Drive first (top cell of section 8) so RUN_DIR lives on Drive and
# survives session end.

# %%
import os, json, datetime

# (Colab) Mount Drive for persistent storage. Uncomment if desired.
# from google.colab import drive; drive.mount("/content/drive")
# BASE_OUT = "/content/drive/MyDrive/phobert-runs"
BASE_OUT = "runs"   # default: save locally; remember to download in section 12

RUN_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# To RESUME after interruption: paste the old RUN directory here (e.g. "runs/phobert-sentiment-20260826-101500").
# Empty ("") = start a new RUN. On resume, keep the same seed/hyperparameters/split as before.
RESUME_RUN_DIR = ""

if RESUME_RUN_DIR:
    RUN_DIR = RESUME_RUN_DIR
    assert os.path.isdir(RUN_DIR), f"RUN directory to resume not found: {RUN_DIR}"
    print("Will RESUME into existing RUN_DIR:", RUN_DIR)
else:
    RUN_DIR = os.path.join(BASE_OUT, f"phobert-sentiment-{RUN_ID}")
    os.makedirs(RUN_DIR, exist_ok=True)
    print("Created new RUN_DIR:", RUN_DIR)

# %%
import numpy as np
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

ID2LABEL = {i: n for i, n in enumerate(LABELS)}
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
    }

args = TrainingArguments(
    output_dir=os.path.join(RUN_DIR, "checkpoints"),
    logging_dir=os.path.join(RUN_DIR, "logs"),   # TensorBoard logs
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,           # keep the 3 latest checkpoints so we can RESUME after interruption
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    seed=SEED,
    logging_steps=20,
    report_to="none",             # switch to "tensorboard" for visual monitoring
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=ds_train, eval_dataset=ds_val,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# %% [markdown]
# ## 9. Train (with RESUME + safe per-step logging)
#
# - **RESUME:** if Colab/Kaggle drops mid-run, rerun the notebook FROM THE TOP but point
#   `RESUME_RUN_DIR` (section 8) at the old RUN directory, then run this cell. The Trainer
#   loads the latest checkpoint and continues, without losing completed epochs. Empty = new run.
# - **Safe logging:** a callback writes each log line (loss/lr/metric) to `train_log.jsonl`
#   as it happens, so even a mid-run crash leaves data to debug and write up.

# %%
import json
from transformers import TrainerCallback

class JsonlLoggerCallback(TrainerCallback):
    """Append each log line to jsonl immediately (safe if training crashes)."""
    def __init__(self, path):
        self.path = path
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": state.global_step, "epoch": state.epoch, **logs}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

trainer.add_callback(JsonlLoggerCallback(os.path.join(RUN_DIR, "train_log.jsonl")))

# Find a checkpoint to resume: prefer RESUME_RUN_DIR (section 8); if empty, scan RUN_DIR.
from transformers.trainer_utils import get_last_checkpoint
ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
resume_ckpt = None
if os.path.isdir(ckpt_dir):
    resume_ckpt = get_last_checkpoint(ckpt_dir)   # None if no checkpoint yet
if resume_ckpt:
    print("Resuming from checkpoint:", resume_ckpt)
else:
    print("Fresh training (no checkpoint found to resume).")

train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

# Persist the training history (per-epoch loss/metric) to file so it survives session end.
import pandas as pd
log_hist = pd.DataFrame(trainer.state.log_history)
log_hist.to_csv(os.path.join(RUN_DIR, "train_log_history.csv"), index=False)

# Plot the learning curve (train loss + val macro-F1) for the report and convergence debugging.
try:
    import matplotlib.pyplot as plt
    lh = log_hist.copy()
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    tr = lh.dropna(subset=["loss"]) if "loss" in lh else lh.iloc[0:0]
    if len(tr):
        ax1.plot(tr["step"], tr["loss"], color="tab:red", label="train loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss", color="tab:red")
    if "eval_macro_f1" in lh:
        ev = lh.dropna(subset=["eval_macro_f1"])
        ax2 = ax1.twinx()
        ax2.plot(ev["step"], ev["eval_macro_f1"], color="tab:blue",
                 marker="o", label="val macro-F1")
        ax2.set_ylabel("val macro-F1", color="tab:blue")
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "learning_curve.png"), dpi=150)
    plt.show()
except Exception as e:
    print("Skipping learning-curve plot:", e)

print("Saved training log:", os.path.join(RUN_DIR, "train_log_history.csv"))
print("Per-step safe log:", os.path.join(RUN_DIR, "train_log.jsonl"))
print(log_hist.tail(6))

# %% [markdown]
# ## 10. Evaluate on the test set + SAVE report and confusion matrix
#
# Every metric is written to file (not just printed) for pulling into the report.

# %%
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

pred = trainer.predict(ds_test)
y_pred = np.argmax(pred.predictions, axis=-1)
y_true = pred.label_ids

macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
report_dict = classification_report(y_true, y_pred, target_names=LABELS,
                                    zero_division=0, output_dict=True)
report_txt = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
print("Macro-F1 test:", round(macro_f1, 4))
print(report_txt)

# Save the classification report (txt + json)
with open(os.path.join(RUN_DIR, "test_classification_report.txt"), "w") as f:
    f.write(f"Macro-F1: {macro_f1:.4f}\n\n{report_txt}")
with open(os.path.join(RUN_DIR, "test_report.json"), "w") as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# Confusion matrix (save both the csv numbers and the png figure for the report)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
    os.path.join(RUN_DIR, "confusion_matrix.csv"))
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, rotation=45, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix (macro-F1={macro_f1:.3f})")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Saved report + confusion matrix to", RUN_DIR)

# SAVE the misclassified predictions for qualitative error analysis in the report.
# Includes per-class probabilities and the confidence of each wrong prediction
# (sorted descending, to surface the most confidently wrong cases first).
probs_test = np.exp(pred.predictions - pred.predictions.max(axis=1, keepdims=True))
probs_test = probs_test / probs_test.sum(axis=1, keepdims=True)
err_df = pd.DataFrame({
    "text": list(test_df["text"]),
    "true": [LABELS[i] for i in y_true],
    "pred": [LABELS[i] for i in y_pred],
    "p_NEGATIVE": probs_test[:, 0],
    "p_NEUTRAL":  probs_test[:, 1],
    "p_POSITIVE": probs_test[:, 2],
})
err_df["pred_conf"] = probs_test[np.arange(len(y_pred)), y_pred]
err_df["correct"] = (y_true == y_pred)
mis = err_df[~err_df["correct"]].sort_values("pred_conf", ascending=False)
mis.to_csv(os.path.join(RUN_DIR, "misclassified.csv"), index=False)
# Also save ALL test predictions, to re-analyze later without rerunning the model.
err_df.to_csv(os.path.join(RUN_DIR, "test_predictions.csv"), index=False)
print(f"Wrong cases: {len(mis)}/{len(err_df)}. Saved misclassified.csv + test_predictions.csv")
print(mis.head(10)[["text", "true", "pred", "pred_conf"]].to_string(index=False))

# %% [markdown]
# ## 11. Save the model + reproducibility MANIFEST
#
# The manifest records everything needed to reproduce the run and pull numbers into the
# report: library versions, seed, hyperparameters, split sizes, and final metrics. This is
# the evidence for acceptance criterion #8 (reproducible experiment) in the proposal.

# %%
SAVE_DIR = os.path.join(RUN_DIR, "best")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

manifest = {
    "run_id": RUN_ID,
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "seed": SEED,
    "labels": LABELS,
    "hyperparams": {
        "epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
    },
    "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "class_distribution": df["label_id"].value_counts().sort_index().to_dict(),
    "test_metrics": {
        "macro_f1": macro_f1,
        "accuracy": report_dict["accuracy"],
        "per_class": {lab: report_dict[lab] for lab in LABELS},
    },
    "versions": {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "timestamp": datetime.datetime.now().isoformat(),
}
with open(os.path.join(RUN_DIR, "manifest.json"), "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("Saved model + manifest.json to", RUN_DIR)
print(json.dumps(manifest["test_metrics"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 12. Package the whole RUN for download / Drive
#
# Zip the entire RUN directory (model + logs + report + manifest + figures).

# %%
import shutil
zip_path = shutil.make_archive(RUN_DIR, "zip", RUN_DIR)
print("Created:", zip_path)

# Colab: download to your machine
# from google.colab import files; files.download(zip_path)

# Kaggle: the file lands in /kaggle/working and shows up in the Output tab to download.
# Or save to Drive (if mounted in section 8):
# import shutil; shutil.copytree(RUN_DIR, f"/content/drive/MyDrive/{os.path.basename(RUN_DIR)}")

# %% [markdown]
# ## 13. Generate sentiment probabilities for the news corpus (inference)
#
# With the trained model, produce 3-class probabilities for all crawled Vietstock news
# (upload articles.parquet, or clone the repo and read data/raw/news/articles.parquet).

# %%
def predict_proba(texts, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    texts = list(texts)  # avoid index-based slicing errors when texts is a pandas Series
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)

# Example:
demo = ["Lợi nhuận quý 3 tăng mạnh vượt kỳ vọng", "Khối ngoại bán ròng liên tục"]
print(predict_proba(demo))
