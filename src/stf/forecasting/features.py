"""Causal price features for the market branch.

Every feature at observation date ``t`` uses only rows up to and including ``t``; no
column ever reads ``t+1``. Price-scaled quantities are turned into ratios or relative
differences so the model does not latch onto absolute price levels. Rows without enough
history carry ``NaN`` and are dropped downstream, never imputed with future data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stf import config
from stf.forecasting.calendar import to_local

# Default rolling window lengths (trading days).
MA_WINDOW = 5
VOL_WINDOW = 5

_REQUIRED = ("ticker", "time", "close", "volume")


def price_feature_columns(ma_window: int = MA_WINDOW, vol_window: int = VOL_WINDOW) -> list[str]:
    """Return the ordered feature column names for the given windows."""
    return [
        "ret_1d",
        "log_ret_1d",
        f"roll_vol_{vol_window}",
        f"ma_ratio_{ma_window}",
        "vol_change_1d",
    ]


def price_features(
    prices: pd.DataFrame,
    *,
    ma_window: int = MA_WINDOW,
    vol_window: int = VOL_WINDOW,
    tz: str = config.TIMEZONE,
) -> pd.DataFrame:
    """Build per-``(ticker, observation_date)`` causal price features.

    Features
    --------
    ``ret_1d``
        Simple daily return ``close_t / close_{t-1} - 1``.
    ``log_ret_1d``
        Log return ``ln(close_t / close_{t-1})``.
    ``roll_vol_{w}``
        Rolling standard deviation of ``ret_1d`` over the trailing ``vol_window`` sessions.
    ``ma_ratio_{w}``
        Close divided by its trailing ``ma_window`` moving average (scale-free ~1.0).
    ``vol_change_1d``
        Relative change in traded volume ``volume_t / volume_{t-1} - 1``.

    Returns a frame sorted by ``(ticker, observation_date)`` with the ``close`` retained
    for downstream target construction. Insufficient-history rows hold ``NaN``.
    Session dates are routed through the same study-timezone policy as news alignment
    (:func:`stf.forecasting.calendar.to_local`) before being normalized, so naive and
    timezone-aware price feeds land on the same calendar; a ticker with more than one
    row on the same normalized date raises rather than silently corrupting the
    rolling-window calculations.
    """
    missing = [c for c in _REQUIRED if c not in prices.columns]
    if missing:
        raise ValueError(f"prices missing columns {missing}.")
    if ma_window < 1 or vol_window < 1:
        raise ValueError("window lengths must be >= 1.")

    vol_col = f"roll_vol_{vol_window}"
    ma_col = f"ma_ratio_{ma_window}"
    frames: list[pd.DataFrame] = []
    for ticker, group in prices.groupby("ticker", sort=True):
        g = group.sort_values("time")
        local_dates = to_local(g["time"], tz).dt.tz_localize(None).dt.normalize()
        if local_dates.duplicated().any():
            dupes = sorted(
                local_dates[local_dates.duplicated()]
                .dt.strftime("%Y-%m-%d")
                .unique()
                .tolist()
            )
            raise ValueError(f"{ticker}: duplicate normalized session dates {dupes}.")
        close = g["close"].astype(float).reset_index(drop=True)
        volume = g["volume"].astype(float).reset_index(drop=True)
        ret = close.pct_change(fill_method=None)
        log_ret = np.log(close / close.shift(1))
        roll_vol = ret.rolling(vol_window, min_periods=vol_window).std()
        ma = close.rolling(ma_window, min_periods=ma_window).mean()
        ma_ratio = close / ma
        vol_change = volume.pct_change(fill_method=None)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": str(ticker),
                    "observation_date": local_dates.to_numpy(),
                    "close": close.to_numpy(),
                    "ret_1d": ret.to_numpy(),
                    "log_ret_1d": log_ret.to_numpy(),
                    vol_col: roll_vol.to_numpy(),
                    ma_col: ma_ratio.to_numpy(),
                    "vol_change_1d": vol_change.to_numpy(),
                }
            )
        )
    if not frames:
        cols = ["ticker", "observation_date", "close", *price_feature_columns(ma_window, vol_window)]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class FeatureScaler:
    """Train-only standardization for numeric forecasting features."""

    columns: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str] | tuple[str, ...]):
        names = tuple(columns)
        if not names or not set(names) <= set(frame.columns):
            raise ValueError("Feature scaler columns must exist and be non-empty.")
        values = frame.loc[:, names].to_numpy(dtype=float)
        finite = np.isfinite(values).all(axis=1)
        if not finite.any():
            raise ValueError("Feature scaler needs at least one finite training row.")
        mean = values[finite].mean(axis=0)
        scale = values[finite].std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(names, mean, scale)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with this training fit applied to the selected columns."""
        if not set(self.columns) <= set(frame.columns):
            raise ValueError("Frame is missing one or more scaler columns.")
        out = frame.copy()
        values = out.loc[:, self.columns].to_numpy(dtype=float)
        scaled = (values - self.mean) / self.scale
        for index, column in enumerate(self.columns):
            out[column] = scaled[:, index]
        return out
