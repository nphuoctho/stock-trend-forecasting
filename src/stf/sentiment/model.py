"""Fine-tune PhoBERT for 3-class sentiment classification and produce probabilities.

Uses the HuggingFace Trainer. Auto-detects the device (CPU/GPU). On a machine with no
CUDA, fine-tuning is VERY slow: use smoke-test mode (few samples, 1 epoch) to check the
code, and run real training on a GPU (Colab/Kaggle). Every param + result is saved to the
manifest for reproducibility.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from stf import config
from stf.sentiment.dataset import Split
from stf.sentiment.labels import ID2LABEL, LABEL2ID, NUM_LABELS

MODEL_NAME = "vinai/phobert-base"
MAX_LEN = 256  # PhoBERT-base token ceiling


@dataclass
class TrainConfig:
    """Fine-tune hyperparameters. Saved to the manifest for reproducibility."""
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    epochs: float = 3.0
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    seed: int = 42


def set_seed(seed: int) -> None:
    """Fix the seed for reproducibility (python, numpy, torch)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _TextDataset:
    """A torch Dataset wrapping tokenized (text, label) pairs as the Trainer expects."""

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.enc = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding=False
        )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def _compute_metrics(eval_pred):
    from stf.sentiment.metrics import classification_metrics

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = classification_metrics(labels, preds)
    # Trainer picks the best model by macro_f1.
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"]}


def fine_tune(
    split: Split,
    cfg: TrainConfig | None = None,
    *,
    out_dir: Path | None = None,
) -> dict:
    """Fine-tune PhoBERT on split.train, pick the best by macro-F1 on val,
    evaluate on test. Return a result dict and save checkpoint + manifest.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    cfg = cfg or TrainConfig()
    out_dir = out_dir or config.SENTIMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    ds_train = _TextDataset(split.train["text"], split.train["label_id"], tokenizer, cfg.max_len)
    ds_val = _TextDataset(split.val["text"], split.val["label_id"], tokenizer, cfg.max_len)
    ds_test = _TextDataset(split.test["text"], split.test["label_id"], tokenizer, cfg.max_len)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg.seed,
        logging_steps=20,
        report_to=[],  # don't auto-log to W&B; enable later if needed
        use_cpu=(device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    # Full evaluation on test.
    from stf.sentiment.metrics import classification_metrics

    pred = trainer.predict(ds_test)
    y_pred = np.argmax(pred.predictions, axis=-1)
    test_metrics = classification_metrics(pred.label_ids, y_pred)

    # Save model + tokenizer + manifest.
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    manifest = {
        "config": asdict(cfg),
        "device": device,
        "split_sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "test_metrics": test_metrics,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def predict_proba(texts, model_dir: Path | None = None, *, batch_size: int = 32) -> np.ndarray:
    """Produce 3-class probabilities for a list of texts (using the fine-tuned model).

    Returns an array of shape (len(texts), 3) — class order follows labels.LABELS.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or (config.SENTIMENT_DIR / "best")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    texts = list(texts)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)
