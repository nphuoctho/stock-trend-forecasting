"""Persisted forecasting arms for daily inference.

``run_experiment`` trains every arm inside a walk-forward window and discards the
weights; nothing it produces can serve a new session. This module freezes one arm
end-to-end — label thresholds, feature scalers, and per-seed model weights — so a
daily job can score the latest session without refitting.

Artifact layout under ``<model_dir>``::

    manifest.json     arm, config, thresholds, feature columns, provenance
    scalers.npz       price/sentiment FeatureScaler parameters
    seed_<n>.pt       LSTM state_dict per seed (family == "lstm")
    seed_<n>.joblib   ClassicalBaseline per seed (family == "classical")

Inference averages softmax probabilities across seeds, matching the seed-averaged
metrics the experiment reports.
"""

from __future__ import annotations

import json
import sys
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stf.forecasting.experiment import SENTIMENT_FEATURES, ForecastConfig, frame_hash
from stf.forecasting.features import FeatureScaler, price_feature_columns
from stf.forecasting.labels import (
    ID2TREND,
    TREND2ID,
    TREND_LABELS,
    apply_labels,
    fit_thresholds,
    label_panel,
)

ARM_FAMILIES: dict[str, tuple[str, bool]] = {
    "logreg_price": ("classical", False),
    "logreg_price_sentiment": ("classical", True),
    "lstm_price": ("lstm", False),
    "lstm_price_sentiment": ("lstm", True),
    "tft_price": ("tft", False),
    "tft_price_sentiment": ("tft", True),
}


@dataclass(frozen=True)
class LoadedArm:
    """A refit arm ready for inference."""

    name: str
    family: str
    use_sentiment: bool
    cfg: ForecastConfig
    thresholds: tuple[float, float]
    price_scaler: FeatureScaler
    sent_scaler: FeatureScaler
    models: tuple[object, ...]
    manifest: dict


def _final_split(panel: pd.DataFrame, val_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the last ``val_size`` distinct target dates for early stopping."""
    dates = pd.to_datetime(panel["target_date"]).dropna().sort_values().unique()
    if len(dates) <= val_size:
        raise ValueError(
            f"Need more than val_size={val_size} labeled sessions; got {len(dates)}."
        )
    val_dates = set(dates[-val_size:])
    is_val = pd.to_datetime(panel["target_date"]).isin(val_dates).to_numpy()
    return np.where(~is_val)[0], np.where(is_val)[0]


def _save_scalers(path: Path, price: FeatureScaler, sent: FeatureScaler) -> None:
    np.savez(
        path,
        price_columns=np.asarray(price.columns),
        price_mean=price.mean,
        price_scale=price.scale,
        sent_columns=np.asarray(sent.columns),
        sent_mean=sent.mean,
        sent_scale=sent.scale,
    )


def _load_scalers(path: Path) -> tuple[FeatureScaler, FeatureScaler]:
    data = np.load(path, allow_pickle=False)
    price = FeatureScaler(
        tuple(str(c) for c in data["price_columns"]),
        data["price_mean"],
        data["price_scale"],
    )
    sent = FeatureScaler(
        tuple(str(c) for c in data["sent_columns"]),
        data["sent_mean"],
        data["sent_scale"],
    )
    return price, sent


def refit_arm(
    panel: pd.DataFrame,
    arm: str,
    *,
    cfg: ForecastConfig | None = None,
    output_dir: Path,
    provenance: dict | None = None,
) -> dict:
    """Fit one ladder arm on the full labeled panel and persist it.

    The label thresholds and both feature scalers are fit once on the training
    rows and frozen into the artifact — at inference they are applied verbatim,
    never refit, so live features stay on the training distribution.
    """
    import torch

    from stf.forecasting.models import (
        ClassicalBaseline,
        PriceLSTM,
        PriceSentimentLSTM,
        TemporalFusionClassifier,
        fit_lstm,
        make_two_branch_sequences,
        set_seed,
    )

    if arm not in ARM_FAMILIES:
        raise ValueError(f"Unknown arm {arm!r}; choose from {sorted(ARM_FAMILIES)}.")
    family, use_sentiment = ARM_FAMILIES[arm]
    cfg = cfg or ForecastConfig()

    thresholds = fit_thresholds(panel["target_return"].dropna().to_numpy())
    labeled = label_panel(panel, thresholds)

    train_idx, val_idx = _final_split(labeled, cfg.val_size)
    price_cols = price_feature_columns()
    sent_cols = list(SENTIMENT_FEATURES)
    price_scaler = FeatureScaler.fit(labeled.iloc[train_idx], price_cols)
    sent_scaler = FeatureScaler.fit(labeled.iloc[train_idx], sent_cols)
    scaled = sent_scaler.transform(price_scaler.transform(labeled))

    X_price, X_sent, y, meta = make_two_branch_sequences(
        scaled, price_cols, sent_cols, seq_len=cfg.seq_len
    )
    if len(y) == 0:
        raise RuntimeError("Refit produced no usable sequences.")

    obs = pd.to_datetime(meta["observation_date"])
    train_dates = set(pd.to_datetime(labeled.iloc[train_idx]["observation_date"]))
    val_dates = set(pd.to_datetime(labeled.iloc[val_idx]["observation_date"]))
    train_mask = obs.isin(train_dates).to_numpy()
    val_mask = obs.isin(val_dates).to_numpy()
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("Refit split produced an empty train or val partition.")

    X_both = np.concatenate([X_price, X_sent], axis=2)
    models: list[object] = []
    histories: list[dict] = []
    for seed in cfg.seeds:
        if family == "classical":
            key = X_both if use_sentiment else X_price
            model = ClassicalBaseline(seed=seed).fit(key[train_mask], y[train_mask])
        else:
            set_seed(seed)
            if family == "tft":
                n_in = X_both.shape[-1] if use_sentiment else X_price.shape[-1]
                net: torch.nn.Module = TemporalFusionClassifier(
                    n_in,
                    hidden=cfg.hidden,
                    num_classes=len(TREND_LABELS),
                )
                key = X_both if use_sentiment else X_price
                history = fit_lstm(
                    net,
                    key[train_mask],
                    y[train_mask],
                    X_val=key[val_mask],
                    y_val=y[val_mask],
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    patience=cfg.patience,
                    seed=seed,
                )
            elif use_sentiment:
                net = PriceSentimentLSTM(
                    len(price_cols),
                    len(sent_cols),
                    hidden=cfg.hidden,
                    num_layers=cfg.num_layers,
                    num_classes=len(TREND_LABELS),
                )
                history = fit_lstm(
                    net,
                    X_price[train_mask],
                    y[train_mask],
                    X_sent=X_sent[train_mask],
                    X_val=X_price[val_mask],
                    y_val=y[val_mask],
                    X_sent_val=X_sent[val_mask],
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    patience=cfg.patience,
                    seed=seed,
                )
            else:
                net = PriceLSTM(
                    len(price_cols),
                    hidden=cfg.hidden,
                    num_layers=cfg.num_layers,
                    num_classes=len(TREND_LABELS),
                )
                history = fit_lstm(
                    net,
                    X_price[train_mask],
                    y[train_mask],
                    X_val=X_price[val_mask],
                    y_val=y[val_mask],
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    lr=cfg.lr,
                    patience=cfg.patience,
                    seed=seed,
                )
            histories.append({"seed": seed, **history})
            model = net
        models.append(model)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed, model in zip(cfg.seeds, models):
        if family in ("lstm", "tft"):
            torch.save(model.state_dict(), output_dir / f"seed_{seed}.pt")
        else:
            import joblib

            joblib.dump(model, output_dir / f"seed_{seed}.joblib")
    _save_scalers(output_dir / "scalers.npz", price_scaler, sent_scaler)
    manifest = {
        "arm": arm,
        "family": family,
        "use_sentiment": use_sentiment,
        "config": cfg.as_dict(),
        "seeds": list(cfg.seeds),
        "thresholds": [float(thresholds[0]), float(thresholds[1])],
        "price_features": price_cols,
        "sentiment_features": sent_cols,
        "class_labels": list(TREND_LABELS),
        "panel_rows": int(len(panel)),
        "panel_hash": frame_hash(panel),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "lstm_history": histories or None,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "provenance": provenance or {},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return manifest


def load_arm(model_dir: Path) -> LoadedArm:
    """Load a refit arm directory produced by :func:`refit_arm`."""
    import torch

    from stf.forecasting.models import PriceLSTM, PriceSentimentLSTM, TemporalFusionClassifier

    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = ForecastConfig(**{k: v for k, v in manifest["config"].items() if k in ForecastConfig.__dataclass_fields__})
    price_scaler, sent_scaler = _load_scalers(model_dir / "scalers.npz")

    models: list[object] = []
    for seed in manifest["seeds"]:
        if manifest["family"] in ("lstm", "tft"):
            if manifest["family"] == "tft":
                n_in = (
                    len(manifest["price_features"]) + len(manifest["sentiment_features"])
                    if manifest["use_sentiment"]
                    else len(manifest["price_features"])
                )
                net: torch.nn.Module = TemporalFusionClassifier(
                    n_in,
                    hidden=cfg.hidden,
                    num_classes=len(TREND_LABELS),
                )
            elif manifest["use_sentiment"]:
                net: torch.nn.Module = PriceSentimentLSTM(
                    len(manifest["price_features"]),
                    len(manifest["sentiment_features"]),
                    hidden=cfg.hidden,
                    num_layers=cfg.num_layers,
                    num_classes=len(TREND_LABELS),
                )
            else:
                net = PriceLSTM(
                    len(manifest["price_features"]),
                    hidden=cfg.hidden,
                    num_layers=cfg.num_layers,
                    num_classes=len(TREND_LABELS),
                )
            net.load_state_dict(
                torch.load(model_dir / f"seed_{seed}.pt", weights_only=True)
            )
            net.eval()
            models.append(net)
        else:
            import joblib

            models.append(joblib.load(model_dir / f"seed_{seed}.joblib"))

    return LoadedArm(
        name=manifest["arm"],
        family=manifest["family"],
        use_sentiment=manifest["use_sentiment"],
        cfg=cfg,
        thresholds=(manifest["thresholds"][0], manifest["thresholds"][1]),
        price_scaler=price_scaler,
        sent_scaler=sent_scaler,
        models=tuple(models),
        manifest=manifest,
    )


def _inference_windows(
    scaled: pd.DataFrame,
    price_cols: list[str],
    sent_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Take the latest ``seq_len``-row window per ticker; no label required."""
    price_windows: list[np.ndarray] = []
    sent_windows: list[np.ndarray] = []
    meta: list[dict] = []
    for ticker, group in scaled.groupby("ticker", sort=True):
        g = group.sort_values("observation_date").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        tail = g.iloc[-seq_len:]
        p_win = tail[price_cols].to_numpy(dtype=float)
        s_win = tail[sent_cols].to_numpy(dtype=float)
        if not (np.isfinite(p_win).all() and np.isfinite(s_win).all()):
            continue
        price_windows.append(p_win)
        sent_windows.append(s_win)
        meta.append(
            {
                "ticker": str(ticker),
                "observation_date": g["observation_date"].iloc[-1],
                "has_news": int(g["has_news"].iloc[-1]),
            }
        )
    if not price_windows:
        return (
            np.empty((0, seq_len, len(price_cols)), dtype=np.float32),
            np.empty((0, seq_len, len(sent_cols)), dtype=np.float32),
            pd.DataFrame(columns=["ticker", "observation_date", "has_news"]),
        )
    return (
        np.stack(price_windows).astype(np.float32),
        np.stack(sent_windows).astype(np.float32),
        pd.DataFrame(meta),
    )


def predict_latest(panel: pd.DataFrame, arm: LoadedArm) -> pd.DataFrame:
    """Score the latest session per ticker with a refit arm.

    Returns one row per ticker with per-class probabilities, the argmax label,
    and the news flag of the observation day.
    """
    from stf.forecasting.models import predict_lstm

    price_cols = list(arm.manifest["price_features"])
    sent_cols = list(arm.manifest["sentiment_features"])
    scaled = arm.sent_scaler.transform(arm.price_scaler.transform(panel))
    X_price, X_sent, meta = _inference_windows(
        scaled, price_cols, sent_cols, arm.cfg.seq_len
    )
    if len(meta) == 0:
        raise RuntimeError("No ticker has enough history for an inference window.")

    probas: list[np.ndarray] = []
    for model in arm.models:
        if arm.family == "lstm":
            probs = predict_lstm(
                model, X_price, X_sent if arm.use_sentiment else None
            )
        elif arm.family == "tft":
            key = (
                np.concatenate([X_price, X_sent], axis=2)
                if arm.use_sentiment
                else X_price
            )
            probs = predict_lstm(model, key)
        else:
            key = (
                np.concatenate([X_price, X_sent], axis=2)
                if arm.use_sentiment
                else X_price
            )
            probs = model.predict_proba(key)
        probas.append(probs)
    proba = np.mean(probas, axis=0)
    pred = proba.argmax(axis=1)

    out = meta.copy()
    out["prob_down"] = proba[:, TREND2ID["DOWN"]]
    out["prob_flat"] = proba[:, TREND2ID["FLAT"]]
    out["prob_up"] = proba[:, TREND2ID["UP"]]
    out["y_pred"] = [ID2TREND[int(i)] for i in pred]
    out["arm"] = arm.name
    return out


def resolve_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    thresholds: tuple[float, float],
) -> pd.DataFrame:
    """Join stored predictions with realized next-session labels.

    ``predictions`` needs ``ticker``, ``observation_date``, ``y_pred``, ``arm``.
    Rows whose target session has not traded yet keep ``y_true`` empty.
    """
    labeled = label_panel(panel, thresholds)
    truth = labeled[["ticker", "observation_date", "target_date", "target_label"]]
    merged = predictions.merge(
        truth, on=["ticker", "observation_date"], how="left", validate="many_to_one"
    )
    merged = merged.rename(columns={"target_label": "y_true"})
    merged["correct"] = np.where(
        merged["y_true"].isna(), pd.NA, merged["y_true"] == merged["y_pred"]
    )
    return merged
