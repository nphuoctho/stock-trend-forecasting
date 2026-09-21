"""Serving-path tests: refit → persist → predict → resolve.

uv run pytest tests/test_forecast_serve.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stf.forecasting import assemble
from stf.forecasting.experiment import ForecastConfig
from stf.forecasting.serve import (
    load_arm,
    predict_latest,
    refit_arm,
    resolve_predictions,
)


def _prices(n: int = 90, tickers: tuple[str, ...] = ("FPT", "VNM")) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    frames = []
    for offset, ticker in enumerate(tickers):
        close = pd.Series(
            100.0
            + offset * 10
            + np.cumsum(np.where(np.arange(n) % 3 == 0, 1.5, -0.4)),
            dtype="float64",
        )
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "time": dates,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000_000 + np.arange(n) * 10_000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def panel() -> pd.DataFrame:
    return assemble(_prices())


def test_refit_predict_resolve_roundtrip(panel, tmp_path):
    cfg = ForecastConfig(val_size=10, seeds=(42,))
    manifest = refit_arm(
        panel, "logreg_price", cfg=cfg, output_dir=tmp_path / "logreg_price"
    )
    assert manifest["arm"] == "logreg_price"
    assert (tmp_path / "logreg_price" / "manifest.json").is_file()
    assert (tmp_path / "logreg_price" / "scalers.npz").is_file()
    assert (tmp_path / "logreg_price" / "seed_42.joblib").is_file()

    arm = load_arm(tmp_path / "logreg_price")
    assert arm.name == "logreg_price"
    assert arm.family == "classical"
    assert not arm.use_sentiment

    predictions = predict_latest(panel, arm)
    assert sorted(predictions["ticker"]) == ["FPT", "VNM"]
    probs = predictions[["prob_down", "prob_flat", "prob_up"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert predictions["y_pred"].isin(["DOWN", "FLAT", "UP"]).all()

    resolved = resolve_predictions(predictions, panel, arm.thresholds)
    # The last session per ticker has no next session, so nothing resolves yet.
    assert resolved["y_true"].isna().all()
    assert resolved["correct"].isna().all()


def test_resolve_scores_older_observation(panel, tmp_path):
    cfg = ForecastConfig(val_size=10, seeds=(42,))
    refit_arm(panel, "logreg_price", cfg=cfg, output_dir=tmp_path / "arm")
    arm = load_arm(tmp_path / "arm")

    # Predict from a panel truncated by one session so the target day exists.
    cutoff = panel["observation_date"].sort_values().unique()[-2]
    past_panel = panel[panel["observation_date"] <= cutoff]
    predictions = predict_latest(past_panel, arm)

    resolved = resolve_predictions(predictions, panel, arm.thresholds)
    assert resolved["y_true"].notna().all()
    assert resolved["correct"].isin([True, False]).all()


def test_sentiment_arm_requires_sentiment_columns(panel, tmp_path):
    cfg = ForecastConfig(val_size=10, seeds=(42,))
    refit_arm(
        panel,
        "logreg_price_sentiment",
        cfg=cfg,
        output_dir=tmp_path / "logreg_price_sentiment",
    )
    arm = load_arm(tmp_path / "logreg_price_sentiment")
    assert arm.use_sentiment
    predictions = predict_latest(panel, arm)
    assert len(predictions) == 2


def test_unknown_arm_rejected(panel, tmp_path):
    with pytest.raises(ValueError, match="Unknown arm"):
        refit_arm(panel, "not_an_arm", output_dir=tmp_path / "x")
