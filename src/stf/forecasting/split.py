"""Chronological split utilities: locked holdout and walk-forward windows.

Splits are assigned by whole trading date, never by row, so every ticker sharing a date
lands in the same partition and no ticker leaks market state to another through an uneven
cut. There is no random fallback: a time-aware split that lacks dates raises rather than
shuffling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Positional indices into a panel for one chronological train/val/test split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def frames(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return ``(train, val, test)`` DataFrames tagged with a ``split`` column."""
        return (
            _tag(panel.iloc[self.train], "train"),
            _tag(panel.iloc[self.val], "val"),
            _tag(panel.iloc[self.test], "test"),
        )

    def assign_split(self, panel: pd.DataFrame, *, col: str = "split") -> pd.DataFrame:
        """Return a copy of ``panel`` with ``col`` set to train/val/test per row."""
        out = panel.copy()
        tag = np.array([pd.NA] * len(out), dtype=object)
        tag[self.train] = "train"
        tag[self.val] = "val"
        tag[self.test] = "test"
        out[col] = pd.array(tag, dtype="string")
        return out


def _tag(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = name
    return out


def _sorted_dates(panel: pd.DataFrame, date_col: str) -> np.ndarray:
    if date_col not in panel.columns:
        raise ValueError(f"panel missing time column '{date_col}'.")
    dates = pd.to_datetime(panel[date_col], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Chronological split requires at least one valid target date.")
    return np.sort(np.unique(dates.to_numpy("datetime64[ns]")))


def _indices_for(panel: pd.DataFrame, date_col: str, date_set: np.ndarray) -> np.ndarray:
    col = pd.to_datetime(panel[date_col]).to_numpy("datetime64[ns]")
    return np.nonzero(np.isin(col, date_set))[0]


def chronological_split(
    panel: pd.DataFrame,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    date_col: str = "target_date",
) -> TimeSplit:
    """Split by date into train < val < test with a locked holdout, no overlap.

    ``val_frac`` / ``test_frac`` are fractions of unique dates (each at least one date).
    Raises if the panel is empty or the fractions leave no training dates.
    """
    if panel.empty:
        raise ValueError("Cannot split an empty panel.")
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in (0,1) and leave training dates.")
    dates = _sorted_dates(panel, date_col)
    n = len(dates)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n - n_val - n_test < 1:
        raise ValueError("Not enough unique dates for train/val/test.")
    train_dates = dates[: n - n_val - n_test]
    val_dates = dates[n - n_val - n_test : n - n_test]
    test_dates = dates[n - n_test :]
    return TimeSplit(
        _indices_for(panel, date_col, train_dates),
        _indices_for(panel, date_col, val_dates),
        _indices_for(panel, date_col, test_dates),
    )


def walk_forward_windows(
    panel: pd.DataFrame,
    *,
    n_windows: int,
    test_size: int,
    val_size: int = 0,
    min_train: int | None = None,
    expanding: bool = True,
    date_col: str = "target_date",
) -> list[TimeSplit]:
    """Build chronological walk-forward windows over unique dates.

    Each window's train dates precede its validation dates, which precede its test dates;
    successive test blocks step forward by ``test_size`` dates and never overlap. With
    ``expanding=True`` the train window grows from the start; otherwise it rolls with a
    fixed length of ``min_train`` dates. Sizes are counted in unique trading dates.
    """
    if n_windows < 1 or test_size < 1 or val_size < 0:
        raise ValueError("Require n_windows>=1, test_size>=1, val_size>=0.")
    dates = _sorted_dates(panel, date_col)
    total = len(dates)
    train_len = min_train if min_train is not None else max(1, total - n_windows * test_size - val_size)
    if train_len < 1:
        raise ValueError("min_train resolved to < 1 date.")

    windows: list[TimeSplit] = []
    train_end = train_len
    while len(windows) < n_windows and train_end + val_size + test_size <= total:
        train_start = 0 if expanding else max(0, train_end - train_len)
        train_dates = dates[train_start:train_end]
        val_dates = dates[train_end : train_end + val_size]
        test_dates = dates[train_end + val_size : train_end + val_size + test_size]
        windows.append(
            TimeSplit(
                _indices_for(panel, date_col, train_dates),
                _indices_for(panel, date_col, val_dates),
                _indices_for(panel, date_col, test_dates),
            )
        )
        train_end += test_size
    if not windows:
        raise ValueError("Not enough dates to build a single walk-forward window.")
    return windows
