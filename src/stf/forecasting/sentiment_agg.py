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

    Rows with an unmapped status or a non-finite probability vector are dropped explicitly.
    """
    if aligned.empty or not {"ticker", "observation_date", "mapping_status"} <= set(aligned.columns):
        return _empty()

    df = aligned.copy()
    for col in PROB_COLS:
        if col not in df.columns:
            df[col] = np.nan
    probs = df[list(PROB_COLS)].apply(pd.to_numeric, errors="coerce")
    usable = (
        df["mapping_status"].isin(MAPPED_STATUSES)
        & df["observation_date"].notna()
        & probs.notna().all(axis=1)
    )
    if not usable.any():
        return _empty()

    use = df.loc[usable, ["ticker", "observation_date"]].copy()
    p = probs.loc[usable]
    use["_neg"] = p["prob_negative"].to_numpy()
    use["_neu"] = p["prob_neutral"].to_numpy()
    use["_pos"] = p["prob_positive"].to_numpy()
    use["_pred"] = p[list(PROB_COLS)].to_numpy().argmax(axis=1)  # 0=NEG,1=NEU,2=POS
    use["_score"] = use["_pos"] - use["_neg"]

    rows: list[dict] = []
    for (ticker, date), g in use.groupby(["ticker", "observation_date"], sort=True):
        n = len(g)
        neg_mean = float(g["_neg"].mean())
        pos_mean = float(g["_pos"].mean())
        rows.append(
            {
                "ticker": str(ticker),
                "observation_date": date,
                "news_count": int(n),
                "has_news": 1,
                "sent_prob_negative_mean": neg_mean,
                "sent_prob_neutral_mean": float(g["_neu"].mean()),
                "sent_prob_positive_mean": pos_mean,
                "sent_pos_ratio": float((g["_pred"] == 2).mean()),
                "sent_neg_ratio": float((g["_pred"] == 0).mean()),
                "sent_pos_minus_neg": pos_mean - neg_mean,
                "sent_dispersion": float(g["_score"].std(ddof=0)) if n > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["ticker", "observation_date", *SENTIMENT_COLUMNS])
