"""Daily per-ticker aggregation of article-level sentiment probabilities.

Only news anchored to a trading session (see :mod:`stf.forecasting.calendar`) with a
finite probability vector contributes. No-news days are intentionally absent from this
frame; the panel builder fills them with a neutral prior and ``has_news=0`` so that
"no news" stays distinguishable from "neutral news".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stf.forecasting.calendar import MAPPED_STATUSES, PROB_COLS

# Column order emitted per (ticker, observation_date).
SENTIMENT_COLUMNS: tuple[str, ...] = (
    "news_count",
    "has_news",
    "sent_prob_negative_mean",
    "sent_prob_neutral_mean",
    "sent_prob_positive_mean",
    "sent_pos_ratio",
    "sent_neg_ratio",
    "sent_pos_minus_neg",
    "sent_dispersion",
)

# Trailing-window features derived on the panel, after no-news days are filled in.
# They must be computed on the full session calendar (not on ``daily_sentiment``,
# which omits no-news days) so a quiet stretch actually decays the signal.
ROLLING_WINDOW = 5

ROLLING_SENTIMENT_COLUMNS: tuple[str, ...] = (
    "sent_pos_minus_neg_roll",
    "sent_pos_minus_neg_ewm",
    "news_count_roll_sum",
    "has_news_roll_mean",
)


def add_rolling_sentiment(
    panel: pd.DataFrame, *, window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Append trailing sentiment aggregates to a panel, per ticker and causally.

    Each value covers the ``window`` sessions ending at the row's own observation
    date, so it only uses information available at that session's cutoff. A
    single-row classical model sees a flat sentiment signal on the ~62% of
    ticker-days without news; these trailing columns carry recent news forward so
    the price-versus-price-plus-sentiment comparison is not null by construction.
    """
    if window < 1:
        raise ValueError("window must be >= 1.")
    required = {"ticker", "observation_date", "sent_pos_minus_neg", "news_count", "has_news"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing columns {missing} for rolling sentiment.")

    out = panel.sort_values(["ticker", "observation_date"]).copy()
    grouped = out.groupby("ticker", sort=False)
    score = out["sent_pos_minus_neg"].astype("float64")
    out["sent_pos_minus_neg_roll"] = (
        score.groupby(out["ticker"]).rolling(window, min_periods=1).mean().to_numpy()
    )
    out["sent_pos_minus_neg_ewm"] = (
        score.groupby(out["ticker"]).ewm(span=window, adjust=False).mean().to_numpy()
    )
    out["news_count_roll_sum"] = (
        grouped["news_count"]
        .rolling(window, min_periods=1)
        .sum()
        .to_numpy()
        .astype("float64")
    )
    out["has_news_roll_mean"] = (
        grouped["has_news"].rolling(window, min_periods=1).mean().to_numpy()
    )
    return out.reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "observation_date", *SENTIMENT_COLUMNS])


def daily_sentiment(aligned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate aligned news into daily sentiment features.

    Parameters
    ----------
    aligned:
        Output of :func:`stf.forecasting.calendar.align_news_to_sessions`, carrying
        ``ticker``, ``observation_date``, ``mapping_status`` and the :data:`PROB_COLS`.

    Returns
    -------
    DataFrame
        One row per ``(ticker, observation_date)`` that has at least one usable article:

        ``news_count`` count of usable articles, ``has_news`` always 1 here, class
        probability means, ``sent_pos_ratio`` / ``sent_neg_ratio`` (fraction of articles
        whose argmax class is POS / NEG), ``sent_pos_minus_neg`` (mean POS minus mean NEG)
        and ``sent_dispersion`` (population std of the per-article POS-minus-NEG polarity).

    Rows with an unmapped status, a non-finite probability vector, a probability outside
    ``[0, 1]``, or a probability vector that does not sum to ~1 (within ``1e-3``) are
    dropped explicitly rather than silently aggregated.
    """
    if aligned.empty or not {"ticker", "observation_date", "mapping_status"} <= set(aligned.columns):
        return _empty()

    df = aligned.copy()
    for col in PROB_COLS:
        if col not in df.columns:
            df[col] = np.nan
    probs = df[list(PROB_COLS)].apply(pd.to_numeric, errors="coerce")
    in_bounds = probs.ge(0).all(axis=1) & probs.le(1).all(axis=1)
    sums_to_one = np.isclose(probs.sum(axis=1), 1.0, atol=1e-3)
    usable = (
        df["mapping_status"].isin(MAPPED_STATUSES)
        & df["observation_date"].notna()
        & probs.notna().all(axis=1)
        & in_bounds
        & sums_to_one
    )
    if not usable.any():
        return _empty()

    use = df.loc[usable, ["ticker", "observation_date"]].copy()
    p = probs.loc[usable]
    use["_neg"] = p["prob_negative"].to_numpy()
    use["_neu"] = p["prob_neutral"].to_numpy()
    use["_pos"] = p["prob_positive"].to_numpy()
    pred = p[list(PROB_COLS)].to_numpy().argmax(axis=1)  # 0=NEG,1=NEU,2=POS
    use["_is_pos"] = (pred == 2).astype(float)
    use["_is_neg"] = (pred == 0).astype(float)
    use["_score"] = use["_pos"] - use["_neg"]

    grouped = use.groupby(["ticker", "observation_date"], sort=True)
    agg = grouped.agg(
        news_count=("_neg", "size"),
        sent_prob_negative_mean=("_neg", "mean"),
        sent_prob_neutral_mean=("_neu", "mean"),
        sent_prob_positive_mean=("_pos", "mean"),
        sent_pos_ratio=("_is_pos", "mean"),
        sent_neg_ratio=("_is_neg", "mean"),
    )
    # Population std (ddof=0), fully vectorized; a single-article group yields 0.0.
    agg["sent_dispersion"] = grouped["_score"].std(ddof=0)
    agg["sent_pos_minus_neg"] = (
        agg["sent_prob_positive_mean"] - agg["sent_prob_negative_mean"]
    )
    agg["news_count"] = agg["news_count"].astype("int64")
    agg["has_news"] = np.int64(1)
    agg = agg.reset_index()
    agg["ticker"] = agg["ticker"].astype(str)
    return agg[["ticker", "observation_date", *SENTIMENT_COLUMNS]]
