"""Forecasting model core: baselines, LSTM classifiers, and a minimal training harness.

Two deep models follow the thesis floor: a price-only LSTM and a two-branch price+sentiment
LSTM whose branches are fused by concatenating their final hidden states. Deterministic
majority/random baselines provide the reference floor. The training harness runs real
gradient steps; it does not fabricate trained metrics. All code is offline and downloads
nothing.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn

from stf.forecasting.labels import TREND2ID, TREND_LABELS

NUM_TREND_CLASSES = len(TREND_LABELS)


def set_seed(seed: int = 42) -> None:
    """Seed torch and numpy for reproducible training/inference."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class PriceLSTM(nn.Module):
    """Price-only LSTM classifier over a ``(batch, seq_len, n_features)`` window."""

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 32,
        num_layers: int = 1,
        num_classes: int = NUM_TREND_CLASSES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a price window to class logits ``(batch, num_classes)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, features); got shape {tuple(x.shape)}.")
        if x.size(-1) != self.n_features:
            raise ValueError(f"expected {self.n_features} features; got {x.size(-1)}.")
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class PriceSentimentLSTM(nn.Module):
    """Two-branch LSTM fusing a price window and a sentiment window by concatenation."""

    def __init__(
        self,
        n_price: int,
        n_sent: int,
        *,
        hidden: int = 32,
        num_layers: int = 1,
        num_classes: int = NUM_TREND_CLASSES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_price = n_price
        self.n_sent = n_sent
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.price_lstm = nn.LSTM(n_price, hidden, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.sent_lstm = nn.LSTM(n_sent, hidden, num_layers=num_layers, batch_first=True, dropout=lstm_dropout)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, price: torch.Tensor, sent: torch.Tensor) -> torch.Tensor:
        """Map aligned price/sentiment windows to class logits ``(batch, num_classes)``."""
        if price.dim() != 3 or sent.dim() != 3:
            raise ValueError("price and sent must be (batch, seq_len, features).")
        if price.size(0) != sent.size(0) or price.size(1) != sent.size(1):
            raise ValueError("price and sent must share batch and seq_len.")
        if price.size(-1) != self.n_price or sent.size(-1) != self.n_sent:
            raise ValueError("branch feature dimensions do not match the configured sizes.")
        p_out, _ = self.price_lstm(price)
        s_out, _ = self.sent_lstm(sent)
        fused = torch.cat([p_out[:, -1, :], s_out[:, -1, :]], dim=-1)
        return self.head(fused)


class _GLU(nn.Module):
    """Gated Linear Unit: halves the last dimension with a sigmoid gate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class _GRN(nn.Module):
    """Gated Residual Network (Lim et al. 2021): ELU -> linear -> GLU -> add&norm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.gate = _GLU(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.elu(self.fc1(x))
        gated = self.gate(self.drop(self.fc2(hidden)))
        return self.norm(self.skip(x) + gated)


class TemporalFusionClassifier(nn.Module):
    """Minimal TFT-style classifier over a ``(batch, seq_len, n_features)`` window.

    Follows the TFT encoder path of Lim et al. (2021) reduced to what a univariate
    panel without static covariates needs: per-feature linear embeddings, a variable
    selection network that re-weights features per timestep, a single-layer LSTM
    encoder, multi-head self-attention with a gated residual, and a GRN head over
    the last position. ``n_features`` is the concatenated price (+ sentiment)
    feature count, so the same class serves both the price-only and the fused arm.
    """

    def __init__(
        self,
        n_features: int,
        *,
        hidden: int = 32,
        num_heads: int = 4,
        num_classes: int = NUM_TREND_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by num_heads={num_heads}.")
        self.n_features = n_features
        # One embedding per variable. A single shared nn.Linear(1, hidden) would make
        # every column share w and b, so the variable-selection sum would telescope to
        # w * sum_j(alpha_j * x_j) + b -- a rank-2 signal regardless of hidden size,
        # erasing feature identity before the encoder.
        self.feature_weight = nn.Parameter(torch.empty(n_features, hidden))
        self.feature_bias = nn.Parameter(torch.zeros(n_features, hidden))
        nn.init.xavier_uniform_(self.feature_weight)
        self.variable_selection = _GRN(n_features, n_features, dropout)
        self.encoder = nn.LSTM(hidden, hidden, num_layers=1, batch_first=True)
        self.attention = nn.MultiheadAttention(
            hidden, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_gate = _GLU(hidden)
        self.attn_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            _GRN(hidden, hidden, dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a feature window to class logits ``(batch, num_classes)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, features); got shape {tuple(x.shape)}.")
        if x.size(-1) != self.n_features:
            raise ValueError(f"expected {self.n_features} features; got {x.size(-1)}.")
        # Per-feature embeddings: (B, T, F) -> (B, T, F, H).
        embedded = x.unsqueeze(-1) * self.feature_weight + self.feature_bias
        # Variable selection weights from the raw timestep vector.
        weights = torch.softmax(self.variable_selection(x), dim=-1)
        selected = (embedded * weights.unsqueeze(-1)).sum(dim=2)
        encoded, _ = self.encoder(selected)
        attended, _ = self.attention(encoded, encoded, encoded)
        fused = self.attn_norm(encoded + self.attn_gate(attended))
        return self.head(fused[:, -1, :])


class MajorityBaseline:
    """Predict the most frequent training class for every sample (deterministic)."""

    def __init__(self, num_classes: int = NUM_TREND_CLASSES) -> None:
        self.num_classes = num_classes
        self.majority_: int | None = None

    def fit(self, labels) -> "MajorityBaseline":
        ids = _as_ids(labels)
        if ids.size == 0:
            raise ValueError("Cannot fit MajorityBaseline on empty labels.")
        counts = Counter(int(v) for v in ids)
        # Ties break to the lowest class id for determinism.
        self.majority_ = min(counts, key=lambda c: (-counts[c], c))
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.majority_ is None:
            raise RuntimeError("MajorityBaseline is not fitted.")
        return np.full(int(n), self.majority_, dtype=int)


class RandomBaseline:
    """Sample classes from the training distribution with a fixed seed (deterministic)."""

    def __init__(self, num_classes: int = NUM_TREND_CLASSES, *, seed: int = 42) -> None:
        self.num_classes = num_classes
        self.seed = seed
        self.probs_: np.ndarray | None = None

    def fit(self, labels) -> "RandomBaseline":
        ids = _as_ids(labels)
        if ids.size == 0:
            raise ValueError("Cannot fit RandomBaseline on empty labels.")
        counts = np.bincount(ids, minlength=self.num_classes).astype(float)
        self.probs_ = counts / counts.sum()
        return self

    def predict(self, n: int) -> np.ndarray:
        if self.probs_ is None:
            raise RuntimeError("RandomBaseline is not fitted.")
        rng = np.random.default_rng(self.seed)
        return rng.choice(self.num_classes, size=int(n), p=self.probs_)


class ClassicalBaseline:
    """Multinomial logistic regression over the last session's flattened features.

    A single-row classical learner for the price-only / price-plus-sentiment
    contrast the method chapter requires. It consumes the final timestep of the
    same windows the LSTMs use, so both model families are scored on identical
    rows and identical test dates.
    """

    def __init__(self, *, seed: int = 42, max_iter: int = 2000) -> None:
        self.seed = seed
        self.max_iter = max_iter
        self.model_ = None
        self.classes_: np.ndarray | None = None

    def fit(self, X, y) -> "ClassicalBaseline":
        from sklearn.linear_model import LogisticRegression

        flat = _last_step(X)
        ids = np.asarray(y, dtype=int)
        if flat.shape[0] == 0:
            raise ValueError("Cannot fit ClassicalBaseline on an empty set.")
        self.model_ = LogisticRegression(
            max_iter=self.max_iter, random_state=self.seed
        ).fit(flat, ids)
        self.classes_ = np.asarray(self.model_.classes_, dtype=int)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ClassicalBaseline is not fitted.")
        raw = self.model_.predict_proba(_last_step(X))
        # Re-expand to the fixed three-class layout when a train block lacked a class.
        probs = np.zeros((raw.shape[0], NUM_TREND_CLASSES), dtype=float)
        for col, cls in enumerate(self.classes_):
            probs[:, int(cls)] = raw[:, col]
        return probs


def _last_step(X) -> np.ndarray:
    """Flatten sequence windows to their final timestep, or pass 2-D input through."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 3:
        return arr[:, -1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError("Expected a 2-D or 3-D feature array.")


def _as_ids(labels) -> np.ndarray:
    """Coerce string/int labels to trend class ids, dropping missing values."""
    arr = np.asarray(list(labels), dtype=object)
    ids: list[int] = []
    for value in arr:
        if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
            continue
        if isinstance(value, str):
            ids.append(TREND2ID[value])
        else:
            ids.append(int(value))
    return np.asarray(ids, dtype=int)


def make_sequences(
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    seq_len: int = 5,
    label_col: str = "target_label",
    date_col: str = "observation_date",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build per-ticker sliding windows for the LSTM models.

    For each ticker (sorted by date), emit windows of ``seq_len`` consecutive observed
    trading-session rows whose features are all finite and whose target label is present.
    Calendar gaps such as exchange holidays are intentionally treated as adjacent trading
    sessions; windows never cross tickers and never read past the observation date.

    Returns ``(X, y, meta)`` where ``X`` has shape ``(n, seq_len, len(feature_cols))``,
    ``y`` holds trend class ids, and ``meta`` records ticker/observation_date/target_date.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing feature columns {missing}.")

    windows: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict] = []
    for ticker, group in panel.groupby("ticker", sort=True):
        g = group.sort_values(date_col).reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=float)
        labels = g[label_col]
        for i in range(seq_len - 1, len(g)):
            window = feats[i - seq_len + 1 : i + 1]
            if not np.isfinite(window).all():
                continue
            label = labels.iloc[i]
            if label is None or pd.isna(label):
                continue
            windows.append(window)
            ys.append(TREND2ID[str(label)])
            meta.append(
                {
                    "ticker": str(ticker),
                    "observation_date": g[date_col].iloc[i],
                    "target_date": g["target_date"].iloc[i] if "target_date" in g.columns else pd.NaT,
                }
            )
    n_features = len(feature_cols)
    if windows:
        X = np.stack(windows).astype(np.float32)
    else:
        X = np.empty((0, seq_len, n_features), dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, pd.DataFrame(meta, columns=["ticker", "observation_date", "target_date"])


def make_two_branch_sequences(
    panel: pd.DataFrame,
    price_cols: list[str],
    sent_cols: list[str],
    *,
    seq_len: int = 5,
    label_col: str = "target_label",
    date_col: str = "observation_date",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build aligned price and sentiment windows for :class:`PriceSentimentLSTM`.

    A window is kept only when the price window, the sentiment window, and the label are
    all present, so the two returned arrays share the same rows and ordering. Windows
    never cross tickers or read past the observation date.

    Returns ``(X_price, X_sent, y, meta)`` with matching first dimensions.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")
    for name, cols in (("price", price_cols), ("sent", sent_cols)):
        missing = [c for c in cols if c not in panel.columns]
        if missing:
            raise ValueError(f"panel missing {name} columns {missing}.")

    price_windows: list[np.ndarray] = []
    sent_windows: list[np.ndarray] = []
    ys: list[int] = []
    meta: list[dict] = []
    for ticker, group in panel.groupby("ticker", sort=True):
        g = group.sort_values(date_col).reset_index(drop=True)
        price_feats = g[price_cols].to_numpy(dtype=float)
        sent_feats = g[sent_cols].to_numpy(dtype=float)
        labels = g[label_col]
        for i in range(seq_len - 1, len(g)):
            p_win = price_feats[i - seq_len + 1 : i + 1]
            s_win = sent_feats[i - seq_len + 1 : i + 1]
            if not (np.isfinite(p_win).all() and np.isfinite(s_win).all()):
                continue
            label = labels.iloc[i]
            if label is None or pd.isna(label):
                continue
            price_windows.append(p_win)
            sent_windows.append(s_win)
            ys.append(TREND2ID[str(label)])
            meta.append(
                {
                    "ticker": str(ticker),
                    "observation_date": g[date_col].iloc[i],
                    "target_date": g["target_date"].iloc[i] if "target_date" in g.columns else pd.NaT,
                }
            )
    if price_windows:
        X_price = np.stack(price_windows).astype(np.float32)
        X_sent = np.stack(sent_windows).astype(np.float32)
    else:
        X_price = np.empty((0, seq_len, len(price_cols)), dtype=np.float32)
        X_sent = np.empty((0, seq_len, len(sent_cols)), dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    return X_price, X_sent, y, pd.DataFrame(meta, columns=["ticker", "observation_date", "target_date"])


def predict_lstm(model: nn.Module, X, X_sent=None, *, batch_size: int = 512) -> np.ndarray:
    """Return class-probability rows for ``X`` under ``model`` in eval mode."""
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if X_t.size(0) == 0:
        return np.empty((0, NUM_TREND_CLASSES), dtype=np.float64)
    sent_t = None if X_sent is None else torch.as_tensor(np.asarray(X_sent), dtype=torch.float32)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X_t.size(0), batch_size):
            stop = start + batch_size
            logits = (
                model(X_t[start:stop])
                if sent_t is None
                else model(X_t[start:stop], sent_t[start:stop])
            )
            chunks.append(torch.softmax(logits, dim=1).numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def fit_lstm(
    model: nn.Module,
    X,
    y,
    *,
    X_sent=None,
    X_val=None,
    y_val=None,
    X_sent_val=None,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
    class_weights=None,
    seed: int = 42,
) -> dict:
    """Train ``model`` with mini-batch Adam and return the training history.

    Pass ``X_sent`` to train a :class:`PriceSentimentLSTM` (two-branch forward).
    When a validation set is supplied, the epoch with the best validation
    macro-F1 is restored into ``model`` and training stops after ``patience``
    epochs without improvement; validation data never reaches a gradient step.
    Without a validation set the final epoch is kept.
    """
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Require epochs>=1, batch_size>=1, patience>=1.")
    set_seed(seed)
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    if X_t.size(0) == 0:
        raise ValueError("Cannot train on an empty sequence set.")
    sent_t = None if X_sent is None else torch.as_tensor(np.asarray(X_sent), dtype=torch.float32)

    weight_t = (
        None
        if class_weights is None
        else torch.as_tensor(np.asarray(class_weights), dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    generator = torch.Generator().manual_seed(seed)

    has_val = X_val is not None and y_val is not None and len(np.asarray(y_val)) > 0
    y_val_arr = np.asarray(y_val) if has_val else None
    best_score = -np.inf
    best_state: dict | None = None
    best_epoch = 0
    stale = 0
    loss_history: list[float] = []
    val_history: list[float] = []

    n = X_t.size(0)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        order = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(X_t[idx]) if sent_t is None else model(X_t[idx], sent_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * idx.numel()
        loss_history.append(epoch_loss / n)

        if not has_val:
            continue
        probs = predict_lstm(model, X_val, X_sent_val)
        score = evaluate_predictions(y_val_arr, probs.argmax(axis=1))["macro_f1"]
        val_history.append(score)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "loss_history": loss_history,
        "val_macro_f1_history": val_history,
        "best_epoch": best_epoch if has_val else len(loss_history),
        "best_val_macro_f1": float(best_score) if has_val else None,
        "epochs_run": len(loss_history),
    }


def evaluate_predictions(y_true, y_pred, y_proba=None) -> dict:
    """Score forecasting predictions over the fixed DOWN/FLAT/UP class ids.

    Reports accuracy, macro-F1, balanced accuracy, per-class
    precision/recall/F1/support and the confusion matrix. Supplying ``y_proba``
    adds the macro one-vs-rest ROC-AUC; it is omitted when a test block does not
    contain every class, because OvR-AUC is undefined for an absent class.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length.")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty set.")
    labels = list(range(NUM_TREND_CLASSES))
    if not np.isin(y_true, labels).all() or not np.isin(y_pred, labels).all():
        raise ValueError(f"Labels must use the fixed ids {labels}.")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class_precision": {TREND_LABELS[i]: float(precision[i]) for i in labels},
        "per_class_recall": {TREND_LABELS[i]: float(recall[i]) for i in labels},
        "per_class_f1": {TREND_LABELS[i]: float(f1[i]) for i in labels},
        "support": {TREND_LABELS[i]: int(support[i]) for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "macro_ovr_auc": None,
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if proba.shape != (len(y_true), NUM_TREND_CLASSES):
            raise ValueError("y_proba must have shape (n_samples, n_classes).")
        if len(np.unique(y_true)) == NUM_TREND_CLASSES:
            metrics["macro_ovr_auc"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
            )
    return metrics
