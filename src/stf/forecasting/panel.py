"""Assemble the point-in-time model panel: one row per ticker-day.

The panel joins causal price features with the next-session target and daily sentiment
aggregates. No-news days receive a neutral prior with ``has_news=0`` so they stay
distinguishable from neutral-news days. Feature columns never read future values; only
``target_return`` / ``target_date`` / ``target_label`` do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stf import config
from stf.forecasting import calendar as cal
from stf.forecasting.features import MA_WINDOW, VOL_WINDOW, price_feature_columns, price_features
from stf.forecasting.labels import add_target
from stf.forecasting.sentiment_agg import SENTIMENT_COLUMNS, daily_sentiment

# No-news days: neutral prior for the probability vector, zeroed counts/ratios.
NO_NEWS_FILL: dict[str, float] = {
    "news_count": 0,
    "has_news": 0,
    "sent_prob_negative_mean": 0.0,
    "sent_prob_neutral_mean": 1.0,
    "sent_prob_positive_mean": 0.0,
    "sent_pos_ratio": 0.0,
    "sent_neg_ratio": 0.0,
    "sent_pos_minus_neg": 0.0,
    "sent_dispersion": 0.0,
}


def panel_columns(ma_window: int = MA_WINDOW, vol_window: int = VOL_WINDOW) -> list[str]:
    """Return the stable ordered panel schema for the given feature windows."""
    return [
        "ticker",
        "observation_date",
        "target_date",
        "as_of",
        "close",
        *price_feature_columns(ma_window, vol_window),
        *SENTIMENT_COLUMNS,
        "target_return",
        "target_label",
        "split",
    ]


def build_panel(
    prices: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    *,
    ma_window: int = MA_WINDOW,
    vol_window: int = VOL_WINDOW,
    cutoff: str = config.SESSION_CUTOFF,
    tz: str = config.TIMEZONE,
) -> pd.DataFrame:
    """Build the model panel from prices and optional daily sentiment.

    Parameters
    ----------
    prices:
        Rows with ``ticker,time,open,high,low,close,volume``.
    daily:
        Optional output of :func:`stf.forecasting.sentiment_agg.daily_sentiment`. When
        omitted, every row is treated as a no-news day.

    Returns a frame following :func:`panel_columns`. ``target_label`` and ``split`` start
    empty (``pd.NA``); fill them with :mod:`stf.forecasting.labels` and
    :mod:`stf.forecasting.split` after fitting on training rows only.
    """
    feats = price_features(prices, ma_window=ma_window, vol_window=vol_window)
    feats = add_target(feats)

    as_of = cal.session_as_of(feats["observation_date"], cutoff, tz)
    feats = feats.assign(as_of=as_of)

    if daily is not None and not daily.empty:
        merged = feats.merge(daily, on=["ticker", "observation_date"], how="left")
    else:
        merged = feats.copy()
        for col in SENTIMENT_COLUMNS:
            merged[col] = np.nan

    for col, value in NO_NEWS_FILL.items():
        merged[col] = merged[col].fillna(value)
    merged["news_count"] = merged["news_count"].astype("int64")
    merged["has_news"] = merged["has_news"].astype("int64")
    float_sentiment = [col for col in SENTIMENT_COLUMNS if col not in {"news_count", "has_news"}]
    merged[float_sentiment] = merged[float_sentiment].astype("float64")

    merged["target_label"] = pd.array([pd.NA] * len(merged), dtype="string")
    merged["split"] = pd.array([pd.NA] * len(merged), dtype="string")

    cols = panel_columns(ma_window, vol_window)
    return merged[cols].sort_values(["ticker", "observation_date"]).reset_index(drop=True)


def assemble(
    prices: pd.DataFrame,
    news: pd.DataFrame | None = None,
    *,
    ma_window: int = MA_WINDOW,
    vol_window: int = VOL_WINDOW,
    cutoff: str = config.SESSION_CUTOFF,
    tz: str = config.TIMEZONE,
) -> pd.DataFrame:
    """End-to-end convenience: align news, aggregate daily, then build the panel."""
    daily = None
    if news is not None and not news.empty:
        aligned = cal.align_news_to_sessions(news, prices, cutoff=cutoff, tz=tz)
        daily = daily_sentiment(aligned)
    return build_panel(
        prices, daily, ma_window=ma_window, vol_window=vol_window, cutoff=cutoff, tz=tz
    )
