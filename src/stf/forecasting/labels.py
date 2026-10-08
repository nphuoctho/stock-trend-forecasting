"""Next-session target return and train-only trend thresholds.

The forecast target is the next trading session's simple return. UP/FLAT/DOWN classes are
defined by two return quantiles that are fit **only on training returns** and then applied
verbatim to validation/test, so the class boundary never peeks at held-out data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed class order for the trend target; index = id.
TREND_LABELS: tuple[str, ...] = ("DOWN", "FLAT", "UP")
TREND2ID: dict[str, int] = {name: i for i, name in enumerate(TREND_LABELS)}
ID2TREND: dict[int, str] = {i: name for i, name in enumerate(TREND_LABELS)}

# Default terciles: symmetric split into roughly balanced DOWN/FLAT/UP.
LOW_Q = 1.0 / 3.0
HIGH_Q = 2.0 / 3.0


def fit_thresholds(returns, *, low_q: float = LOW_Q, high_q: float = HIGH_Q) -> tuple[float, float]:
    """Fit ``(q_low, q_high)`` return quantiles from training returns only.

    Non-finite returns are dropped. Raises if no finite return remains or the quantiles
    are misordered. The returned pair is passed unchanged to :func:`apply_labels`.
    """
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("Require 0 <= low_q < high_q <= 1.")
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("Cannot fit thresholds from empty/all-NaN returns.")
    q_low = float(np.quantile(r, low_q))
    q_high = float(np.quantile(r, high_q))
    return q_low, q_high


def apply_labels(returns, thresholds: tuple[float, float]) -> np.ndarray:
    """Label returns as DOWN/FLAT/UP using fixed thresholds; no refitting.

    Rule: ``r < q_low`` -> DOWN, ``q_low <= r <= q_high`` -> FLAT, ``r > q_high`` -> UP.
    Non-finite returns map to ``None`` (object array) so callers keep them as missing.
    """
    q_low, q_high = thresholds
    if q_low > q_high:
        raise ValueError("thresholds must satisfy q_low <= q_high.")
    r = np.asarray(returns, dtype=float)
    out = np.full(r.shape, "FLAT", dtype=object)
    out[r < q_low] = "DOWN"
    out[r > q_high] = "UP"
    out[~np.isfinite(r)] = None
    return out


def add_target(panel: pd.DataFrame, *, return_col: str = "target_return") -> pd.DataFrame:
    """Add next-session ``target_return`` and ``target_date`` per ticker.

    Requires ``ticker``, ``observation_date`` and ``close``. The last session per ticker
    has no next session, so its target is ``NaN``/``NaT``. Only these target columns ever
    reference future values; feature columns stay causal.
    """
    required = {"ticker", "observation_date", "close"}
    if not required <= set(panel.columns):
        raise ValueError(f"panel needs columns {sorted(required)}.")
    frames: list[pd.DataFrame] = []
    for _, group in panel.groupby("ticker", sort=True):
        g = group.sort_values("observation_date").copy()
        next_close = g["close"].shift(-1)
        g[return_col] = next_close / g["close"] - 1.0
        g["target_date"] = g["observation_date"].shift(-1)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def retarget_horizon(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Re-point the target at the close ``horizon`` sessions ahead instead of one.

    A one-session horizon is the hardest case: whatever a news item implies has to
    show up within a single close-to-close move or not at all. Widening the horizon
    tests whether a null at ``h=1`` is specific to that timing rather than a property
    of the signal.

    The cost is overlapping labels. At ``h=5`` consecutive rows share four of their
    five sessions, so the effective number of independent observations is roughly
    ``n/h``: point estimates stay usable but intervals computed as if the rows were
    independent are too narrow, and the walk-forward date blocks are the only thing
    keeping train and test outcomes disjoint. Report the horizon next to every number
    taken from a run like this.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    required = {"ticker", "observation_date", "close"}
    if not required <= set(panel.columns):
        raise ValueError(f"panel needs columns {sorted(required)}.")
    frames: list[pd.DataFrame] = []
    for _, group in panel.groupby("ticker", sort=True):
        g = group.sort_values("observation_date").copy()
        g["target_return"] = g["close"].shift(-horizon) / g["close"] - 1.0
        g["target_date"] = g["observation_date"].shift(-horizon)
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def cross_sectional_excess(
    panel: pd.DataFrame, *, return_col: str = "target_return"
) -> pd.DataFrame:
    """Replace the target with the return in excess of the same session's cross-section.

    Roughly half of a VN30 constituent's daily return is the common market move:
    over 2020--2025 the ten study tickers have a mean pairwise return correlation of
    0.42 and a mean :math:`R^2` of 0.48 against their equal-weighted average. A
    ticker-specific news signal can only act on what is left, so predicting the raw
    return dilutes the effect being measured. Subtracting the equal-weighted mean of
    the same ``target_date`` turns the task into "does this ticker beat the basket
    tomorrow", which is the question company news can actually answer.

    This is a change of label, not a feature: the subtracted mean is realized on the
    target date together with the return itself, so nothing is known earlier than
    before. Rows whose target date holds a single finite return are dropped to
    ``NaN`` -- a one-stock cross-section has no meaningful excess return.
    """
    if return_col not in panel.columns:
        raise ValueError(f"panel missing '{return_col}'.")
    if "target_date" not in panel.columns:
        raise ValueError("panel needs 'target_date' to build a cross-section.")
    out = panel.copy()
    returns = pd.to_numeric(out[return_col], errors="coerce")
    groups = out["target_date"]
    mean = returns.groupby(groups).transform("mean")
    breadth = returns.groupby(groups).transform("count")
    excess = returns - mean
    out[return_col] = excess.where(breadth > 1)
    return out


def label_panel(
    panel: pd.DataFrame,
    thresholds: tuple[float, float],
    *,
    return_col: str = "target_return",
    label_col: str = "target_label",
) -> pd.DataFrame:
    """Return a copy of ``panel`` with ``label_col`` filled from ``thresholds``.

    Rows with a non-finite target return keep ``pd.NA``.
    """
    if return_col not in panel.columns:
        raise ValueError(f"panel missing '{return_col}'.")
    out = panel.copy()
    labels = apply_labels(out[return_col].to_numpy(), thresholds)
    out[label_col] = pd.array([lab if lab is not None else pd.NA for lab in labels], dtype="string")
    return out
