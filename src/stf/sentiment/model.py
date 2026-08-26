"""Fine-tune PhoBERT cho phân loại cảm xúc 3 lớp và sinh xác suất.

Dùng HuggingFace Trainer. Tự phát hiện device (CPU/GPU). Trên máy không có CUDA,
fine-tune sẽ RẤT chậm: dùng chế độ smoke-test (ít mẫu, 1 epoch) để kiểm code, và
chạy training thật trên GPU (Colab/Kaggle). Mọi tham số + kết quả được lưu manifest
để tái lập.
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
MAX_LEN = 256  # trần token của PhoBERT-base


@dataclass
class TrainConfig:
    """Siêu tham số fine-tune. Lưu vào manifest để tái lập."""
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    epochs: float = 3.0
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    seed: int = 42


def set_seed(seed: int) -> None:
    """Cố định seed cho tái lập (python, numpy, torch)."""
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
    """torch Dataset gói (text, label) đã tokenize theo yêu cầu của Trainer."""

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
    # Trainer chọn best model theo macro_f1.
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"]}


def fine_tune(
    split: Split,
    cfg: TrainConfig | None = None,
    *,
    out_dir: Path | None = None,
) -> dict:
    """Fine-tune PhoBERT trên split.train, chọn best theo macro-F1 trên val,
    đánh giá trên test. Trả dict kết quả và lưu checkpoint + manifest.
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
        report_to=[],  # không tự log lên W&B; bật sau nếu cần
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

    # Đánh giá đầy đủ trên test.
    from stf.sentiment.metrics import classification_metrics

    pred = trainer.predict(ds_test)
    y_pred = np.argmax(pred.predictions, axis=-1)
    test_metrics = classification_metrics(pred.label_ids, y_pred)

    # Lưu model + tokenizer + manifest.
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
    """Sinh xác suất 3 lớp cho danh sách text (dùng model đã fine-tune).

    Trả mảng shape (len(texts), 3) — thứ tự lớp theo labels.LABELS.
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
