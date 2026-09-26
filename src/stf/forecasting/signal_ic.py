"""Model-free rank association between the daily sentiment score and returns.

The forecasting arms answer "can this architecture use the signal". A null ablation
alone is consistent both with a weak estimator and with a weak signal. The rank
correlation adds one piece of evidence: it is computed without fitting anything, so
it is not contaminated by the choice of architecture.

What it does **not** establish
------------------------------
A Spearman coefficient near zero rules out a *monotone* association between *this
particular scalar score* and the return. It is not an upper bound on predictability
and not a measure of information. A perfectly predictable non-monotone relation --
``y = x**2`` for symmetric ``x`` is the standard counterexample -- has zero rank
correlation. Nor does it speak to a different aggregation, a conditional or
interaction effect, a longer horizon, or a better sentiment measurement. Read a null
here as "no monotone association detected for this score at this horizon", and treat
it as exploratory: the contrasts below were not pre-registered.

Two contrasts are reported:

``same`` vs ``next``
    Sentiment measured against the session it is already anchored to, versus the
    session after it. A score that co-moves with its own session but not the next
    one is consistent with the news being reflected in the price by the close --
    consistent with, not proof of: the same pattern also arises from reverse
    causality (prices driving the tone of the coverage) and from the score simply
    being too noisy to survive one more day of dilution.

``raw`` vs ``excess``
    Against the ticker's own return, versus its return net of the equal-weighted
    cross-section. Company news should act on the idiosyncratic part, and roughly
    half of a VN30 daily return is the shared market move.

Intervals resample whole sessions, matching the dependence structure of the panel
and the convention used by the forecasting experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Signals worth testing: the per-session score and its trailing decay.
IC_SIGNALS: tuple[str, ...] = ("sent_pos_minus_neg", "sent_pos_minus_neg_ewm")

# (name, shift, cross-sectionally demeaned) - shift 0 is the session the news is
# anchored to, shift -1 the session the model is asked to predict.
IC_TARGETS: tuple[tuple[str, int, bool], ...] = (
    ("same_session_return", 0, False),
    ("same_session_excess_return", 0, True),
    ("next_session_return", -1, False),
    ("next_session_excess_return", -1, True),
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so Spearman is Pearson on ranks even with ties."""
    return pd.Series(values).rank(method="average").to_numpy()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def build_return_targets(prices: pd.DataFrame) -> pd.DataFrame:
    """Per ``(ticker, observation_date)`` realized returns, raw and cross-sectional."""
    required = {"ticker", "observation_date", "close"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"prices missing columns {missing}.")
    out = prices.sort_values(["ticker", "observation_date"]).copy()
    out["_same"] = out.groupby("ticker")["close"].pct_change(fill_method=None)
    out["_next"] = out.groupby("ticker")["_same"].shift(-1)
    market = out.groupby("observation_date")[["_same", "_next"]].transform("mean")
    frame = pd.DataFrame(
        {
            "ticker": out["ticker"].to_numpy(),
            "observation_date": out["observation_date"].to_numpy(),
            "same_session_return": out["_same"].to_numpy(),
            "next_session_return": out["_next"].to_numpy(),
            "same_session_excess_return": (out["_same"] - market["_same"]).to_numpy(),
            "next_session_excess_return": (out["_next"] - market["_next"]).to_numpy(),
        }
    )
    return frame


def information_coefficients(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    samples: int = 2000,
    seed: int = 7,
    news_only: bool = True,
) -> dict:
    """Rank correlation of each signal with each return horizon, with date-block CIs."""
    merged = panel.merge(targets, on=["ticker", "observation_date"], how="inner")
    if news_only:
        merged = merged[merged["has_news"] == 1]
    if merged.empty:
        raise ValueError("No rows left to measure an information coefficient on.")

    rng = np.random.default_rng(seed)
    dates = np.sort(merged["observation_date"].unique())
    index = {date: i for i, date in enumerate(dates)}
    slot = merged["observation_date"].map(index).to_numpy()
    draws = rng.integers(0, len(dates), size=(samples, len(dates)))
    # date -> row positions, so a resampled session carries its whole cross-section.
    rows_by_date = [np.flatnonzero(slot == i) for i in range(len(dates))]

    results = []
    for signal in IC_SIGNALS:
        if signal not in merged.columns:
            continue
        for target, _, _ in IC_TARGETS:
            block = merged[[signal, target]].to_numpy(dtype=float)
            keep = np.isfinite(block).all(axis=1)
            if keep.sum() < 30:
                continue
            point = _spearman(block[keep, 0], block[keep, 1])
            boot = np.empty(samples)
            for s in range(samples):
                picked = np.concatenate([rows_by_date[i] for i in draws[s]])
                sub = block[picked]
                sub = sub[np.isfinite(sub).all(axis=1)]
                boot[s] = _spearman(sub[:, 0], sub[:, 1])
            boot = boot[np.isfinite(boot)]
            results.append(
                {
                    "signal": signal,
                    "target": target,
                    "n": int(keep.sum()),
                    "ic": point,
                    "low": float(np.quantile(boot, 0.025)),
                    "high": float(np.quantile(boot, 0.975)),
                    "one_sided_p_le_zero": float((boot <= 0).mean()),
                }
            )

    return {
        "rows": results,
        "sessions": int(len(dates)),
        "news_only": news_only,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def cross_sectional_ic(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    signal: str = "sent_pos_minus_neg",
    target: str = "next_session_excess_return",
    min_names: int = 4,
) -> dict:
    """Mean of the per-session rank correlation, the way a quant signal is actually used.

    Pooling every ticker-day into one correlation lets a few volatile sessions
    dominate. Ranking inside each session and averaging gives each session one vote
    and yields a standard error from the session-to-session spread.
    """
    merged = panel.merge(targets, on=["ticker", "observation_date"], how="inner")
    merged = merged[merged["has_news"] == 1]
    per_session = []
    for date, block in merged.groupby("observation_date", sort=True):
        pair = block[[signal, target]].to_numpy(dtype=float)
        pair = pair[np.isfinite(pair).all(axis=1)]
        if len(pair) < min_names or np.unique(pair[:, 0]).size < 3:
            continue
        value = _spearman(pair[:, 0], pair[:, 1])
        if np.isfinite(value):
            per_session.append({"observation_date": date, "ic": value})
    if not per_session:
        raise ValueError("No session had enough names to rank.")
    values = np.asarray([row["ic"] for row in per_session])
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    return {
        "signal": signal,
        "target": target,
        "sessions": int(values.size),
        "mean_ic": float(values.mean()),
        "standard_error": se,
        "t_statistic": float(values.mean() / se) if se > 0 else 0.0,
    }
