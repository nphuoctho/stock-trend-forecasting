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
TRUNCATION_STRATEGIES: tuple[str, ...] = ("head", "tail", "head_tail")


@dataclass
class TrainConfig:
    """Fine-tune hyperparameters. Saved to the manifest for reproducibility."""

    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    truncation_strategy: str = "head"
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
def truncate_token_ids(
    token_ids: list[int], max_len: int, strategy: str
) -> list[int]:
    """Keep the requested part of a token sequence before adding specials."""
    if strategy not in TRUNCATION_STRATEGIES:
        raise ValueError(
            f"Unknown truncation strategy {strategy!r}; "
            f"expected one of {TRUNCATION_STRATEGIES}."
        )
    if max_len < 1:
        raise ValueError("max_len must be at least 1.")
    if len(token_ids) <= max_len:
        return list(token_ids)
    if strategy == "head":
        return list(token_ids[:max_len])
    if strategy == "tail":
        return list(token_ids[-max_len:])
    head_len = (max_len + 1) // 2
    tail_len = max_len - head_len
    tail = list(token_ids[-tail_len:]) if tail_len else []
    return list(token_ids[:head_len]) + tail


class _TextDataset:
    """A torch Dataset wrapping tokenized (text, label) pairs for Trainer."""

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len: int,
        truncation_strategy: str = "head",
    ):
        if max_len < tokenizer.num_special_tokens_to_add(pair=False) + 1:
            raise ValueError("max_len is too small for PhoBERT special tokens.")
        self.enc = []
        content_max_len = max_len - tokenizer.num_special_tokens_to_add(pair=False)
        for text in texts:
            token_ids = tokenizer(
                str(text), add_special_tokens=False, truncation=False
            )["input_ids"]
            selected = truncate_token_ids(
                token_ids, content_max_len, truncation_strategy
            )
            encoded = tokenizer.prepare_for_model(
                selected,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
            )
            self.enc.append(
                {
                    key: value
                    for key, value in encoded.items()
                    if key in {"input_ids", "attention_mask", "token_type_ids"}
                }
            )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.enc[i])
        item["labels"] = int(self.labels[i])
        return item


def _compute_metrics(eval_pred):
    from stf.sentiment.metrics import classification_metrics

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = classification_metrics(labels, preds)
    # Trainer picks the best model by macro_f1.
    return {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
    }


def fine_tune(
    split: Split,
    cfg: TrainConfig | None = None,
    *,
    out_dir: Path | None = None,
) -> dict:
    """Fine-tune PhoBERT on split.train, pick the best by macro-F1 on val,
    evaluate on test. Return a result dict and save checkpoint + manifest.
    """
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

    ds_train = _TextDataset(
        split.train["text"],
        split.train["label_id"],
        tokenizer,
        cfg.max_len,
        cfg.truncation_strategy,
    )
    ds_val = _TextDataset(
        split.val["text"],
        split.val["label_id"],
        tokenizer,
        cfg.max_len,
        cfg.truncation_strategy,
    )
    ds_test = _TextDataset(
        split.test["text"],
        split.test["label_id"],
        tokenizer,
        cfg.max_len,
        cfg.truncation_strategy,
    )

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
        "split_sizes": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "test_metrics": test_metrics,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def predict_proba(
    texts,
    model_dir: Path | None = None,
    *,
    batch_size: int = 32,
    truncation_strategy: str = "head",
) -> np.ndarray:
    """Return one probability row per input text."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    texts = list(texts)
    if not texts:
        return np.empty((0, NUM_LABELS), dtype=np.float32)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or (config.SENTIMENT_DIR / "best")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(
        device
    )
    model.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokenized = _TextDataset(
                batch,
                [0] * len(batch),
                tokenizer,
                MAX_LEN,
                truncation_strategy,
            )
            enc = tokenizer.pad(
                tokenized.enc,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0)
