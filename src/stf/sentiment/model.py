"""Fine-tune PhoBERT for 3-class sentiment classification and produce probabilities.

Uses the HuggingFace Trainer. Auto-detects the device (CPU/GPU). On a machine with no
CUDA, fine-tuning is VERY slow: use smoke-test mode (few samples, 1 epoch) to check the
code, and run real training on a GPU (Colab/Kaggle). Every param + result is saved to the
manifest for reproducibility.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from stf import config
from stf.sentiment import dataset
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
    warmup_ratio: float = 0.1
    seed: int = 42
    class_weighting: str = "none"


def compute_class_weights(
    labels: list[int] | np.ndarray, num_classes: int = NUM_LABELS
) -> np.ndarray:
    """Return inverse-frequency weights computed from one training split."""
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional array.")
    if (values < 0).any() or (values >= num_classes).any():
        raise ValueError("labels must be valid class ids.")
    counts = np.bincount(values, minlength=num_classes)
    if (counts == 0).any():
        raise ValueError("Every class must occur in the training split.")
    return values.size / (num_classes * counts.astype(np.float64))

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


def bounded_warmup_steps(
    num_examples: int,
    batch_size: int,
    epochs: float,
    warmup_ratio: float,
) -> int:
    """Warmup steps as a fraction of total optimizer steps, capped at the total.

    Replaces a fixed ``warmup_steps`` that could silently exceed the number of
    optimizer steps on small datasets (producing a warmup that never ends).
    Returns an integer in ``[0, total_steps]``.
    """
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1.")
    if num_examples < 1 or batch_size < 1:
        raise ValueError("num_examples and batch_size must be at least 1.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    steps_per_epoch = math.ceil(num_examples / batch_size)
    total_steps = max(1, math.ceil(steps_per_epoch * epochs))
    warmup = round(total_steps * warmup_ratio)
    return max(0, min(warmup, total_steps))


def _load_manifest(model_dir: Path) -> dict | None:
    """Load the training manifest for a checkpoint, if one exists.

    ``fine_tune`` saves the model under ``<out_dir>/best`` and the manifest at
    ``<out_dir>/manifest.json``, so both the checkpoint dir and its parent are
    checked.
    """
    model_dir = Path(model_dir)
    for candidate in (model_dir / "manifest.json", model_dir.parent / "manifest.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
    return None


def resolve_truncation_strategy(
    model_dir: Path, override: str | None = None
) -> str:
    """Pick the truncation strategy for inference on a checkpoint.

    An explicit, valid ``override`` always wins. Otherwise the strategy saved in
    the checkpoint manifest is used, so inference matches training. Falls back to
    ``"head"`` only when no manifest strategy is available.
    """
    if override is not None:
        if override not in TRUNCATION_STRATEGIES:
            raise ValueError(
                f"Unknown truncation strategy {override!r}; "
                f"expected one of {TRUNCATION_STRATEGIES}."
            )
        return override
    manifest = _load_manifest(model_dir)
    if manifest:
        saved = manifest.get("config", {}).get("truncation_strategy")
        if saved in TRUNCATION_STRATEGIES:
            return saved
    return "head"


def resolve_inference_config(
    model_dir: Path,
    *,
    truncation_strategy: str | None = None,
    max_len: int | None = None,
) -> tuple[str, int]:
    """Resolve the ``(truncation_strategy, max_len)`` pair used to train a checkpoint.

    Explicit overrides always win. Otherwise both values come from the checkpoint
    manifest saved by :func:`fine_tune`, so inference matches training exactly
    instead of silently using the module's default ``MAX_LEN``. Falls back to
    ``"head"``/``MAX_LEN`` only when no manifest value is available.
    """
    strategy = resolve_truncation_strategy(model_dir, truncation_strategy)
    if max_len is not None:
        if max_len < 1:
            raise ValueError("max_len must be at least 1.")
        return strategy, max_len
    manifest = _load_manifest(model_dir)
    if manifest:
        saved = manifest.get("config", {}).get("max_len")
        if isinstance(saved, int) and saved >= 1:
            return strategy, saved
    return strategy, MAX_LEN


def reproducibility_metadata() -> dict:
    """Data-independent environment metadata for manifests.

    Records the interpreter and key library versions so a run can be
    reconstructed. Contains no secrets, paths, or fabricated hashes; unavailable
    optional libraries are simply omitted.
    """
    import platform

    meta: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for name, module in (("transformers", "transformers"), ("torch", "torch")):
        try:
            meta[name] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - optional at metadata time
            pass
    return meta


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
    source_path: str | Path | None = None,
) -> dict:
    """Fine-tune PhoBERT on split.train, pick the best by macro-F1 on val,
    evaluate on test. Return a result dict and save the best model + manifest.

    ``source_path``, when given, is hashed (never copied) into the manifest's
    provenance block alongside deterministic fingerprints of the train/val/test
    frames, for reproducibility without embedding raw text.
    """
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    out_dir = out_dir or config.SENTIMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    set_seed(cfg.seed)
    device = get_device()
    if cfg.class_weighting not in ("none", "inverse_frequency"):
        raise ValueError("class_weighting must be 'none' or 'inverse_frequency'.")
    class_weights = (
        compute_class_weights(split.train["label_id"].to_numpy())
        if cfg.class_weighting == "inverse_frequency"
        else None
    )

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

    warmup_steps = bounded_warmup_steps(
        len(split.train), cfg.batch_size, cfg.epochs, cfg.warmup_ratio
    )

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg.seed,
        logging_steps=20,
        report_to=[],  # don't auto-log to W&B; enable later if needed
        use_cpu=(device == "cpu"),
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": ds_train,
        "eval_dataset": ds_val,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": _compute_metrics,
    }
    if class_weights is None:
        trainer = Trainer(**trainer_kwargs)
    else:
        import torch

        class WeightedTrainer(Trainer):
            def __init__(self, *args, class_weights, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = torch.as_tensor(
                    class_weights, dtype=torch.float32
                )
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits,
                    labels,
                    weight=self.class_weights.to(outputs.logits.device),
                )
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(class_weights=class_weights, **trainer_kwargs)
    trainer.train()

    # Full evaluation on test.
    from stf.sentiment.metrics import classification_metrics

    pred = trainer.predict(ds_test)
    y_pred = np.argmax(pred.predictions, axis=-1)
    test_metrics = classification_metrics(pred.label_ids, y_pred)

    # Save the selected model; per-epoch checkpoints are transient for CV.
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    manifest = {
        "config": asdict(cfg),
        "device": device,
        "warmup_steps": warmup_steps,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "split_sizes": {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        },
        "test_metrics": test_metrics,
        "reproducibility": reproducibility_metadata(),
        "provenance": {
            "source_file_sha256": (
                dataset.file_fingerprint(source_path) if source_path is not None else None
            ),
            "split_fingerprints": {
                "train": dataset.frame_fingerprint(split.train),
                "val": dataset.frame_fingerprint(split.val),
                "test": dataset.frame_fingerprint(split.test),
            },
        },
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
    truncation_strategy: str | None = None,
    max_len: int | None = None,
) -> np.ndarray:
    """Return one probability row per input text.

    When ``truncation_strategy`` or ``max_len`` is ``None``, the value saved in the
    checkpoint manifest is used so inference matches training; pass an explicit
    value to override either one (see :func:`resolve_inference_config`).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    texts = list(texts)
    if not texts:
        return np.empty((0, NUM_LABELS), dtype=np.float32)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = model_dir or (config.SENTIMENT_DIR / "best")
    strategy, resolved_max_len = resolve_inference_config(
        model_dir, truncation_strategy=truncation_strategy, max_len=max_len
    )
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
                resolved_max_len,
                strategy,
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
