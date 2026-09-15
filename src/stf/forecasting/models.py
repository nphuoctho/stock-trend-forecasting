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


def fit_lstm(
    model: nn.Module,
    X,
    y,
    *,
    X_sent=None,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
) -> list[float]:
    """Train ``model`` with full-batch Adam + cross-entropy; return the loss history.

    Pass ``X_sent`` to train a :class:`PriceSentimentLSTM` (two-branch forward). This is a
    real, minimal optimization loop — it does not claim any evaluation result.
    """
    set_seed(seed)
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    if X_t.size(0) == 0:
        raise ValueError("Cannot train on an empty sequence set.")
    sent_t = None if X_sent is None else torch.as_tensor(np.asarray(X_sent), dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history: list[float] = []
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(X_t) if sent_t is None else model(X_t, sent_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
