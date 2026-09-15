"""Point-in-time trading-day alignment for financial news.

News is only usable for the model on the first trading session whose 15:00 cutoff
falls at or after the article timestamp. A story filed at 14:00 on a trading day can
inform that day's next-session forecast; a story filed at 16:00, on a weekend, or on a
holiday rolls forward to the next available session. Timestamps that fall past the last
session in the price calendar cannot be anchored and are reported, never silently kept.

All functions are deterministic and never look at future price values: the mapping uses
only the article timestamp and the fixed trading calendar derived from observed prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stf import config

# News rows carry these per-class probabilities once the sentiment model has scored them.
PROB_COLS: tuple[str, ...] = ("prob_negative", "prob_neutral", "prob_positive")

# Terminal mapping states retained on every aligned row.
MAPPED_STATUSES: tuple[str, ...] = ("same_session", "next_session")


def parse_cutoff(cutoff: str = config.SESSION_CUTOFF) -> tuple[int, int]:
    """Parse an ``"HH:MM"`` session cutoff into an ``(hour, minute)`` pair."""
    hh, _, mm = cutoff.partition(":")
    hour, minute = int(hh), int(mm or 0)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Invalid cutoff {cutoff!r}; expected HH:MM.")
    return hour, minute


def to_local(values, tz: str = config.TIMEZONE) -> pd.Series:
    """Coerce timestamps to a tz-aware Series in the study timezone.

    Naive inputs are localized to ``tz``; tz-aware inputs are converted. Unparseable
    timestamps become ``NaT`` rather than raising, so callers can report them.
    """
    parsed = pd.to_datetime(pd.Series(values).reset_index(drop=True), errors="coerce", utc=False)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    return parsed.dt.tz_convert(tz)


def trading_sessions(prices: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return ``{ticker: sorted unique session dates}`` as normalized ``datetime64[ns]``.

    The session dates are the observed price rows: the price calendar is ground truth for
    which days a ticker actually traded.
    """
    if not {"ticker", "time"} <= set(prices.columns):
        raise ValueError("prices must have 'ticker' and 'time' columns.")
    out: dict[str, np.ndarray] = {}
    times = pd.to_datetime(prices["time"], errors="raise").dt.normalize()
    frame = pd.DataFrame({"ticker": prices["ticker"].to_numpy(), "date": times.to_numpy()})
    for ticker, group in frame.groupby("ticker", sort=True):
        dates = np.unique(group["date"].to_numpy("datetime64[ns]"))
        out[str(ticker)] = np.sort(dates)
    return out


def session_as_of(dates, cutoff: str = config.SESSION_CUTOFF, tz: str = config.TIMEZONE) -> pd.Series:
    """Map trading dates to their tz-aware point-in-time anchor (the session's cutoff)."""
    hour, minute = parse_cutoff(cutoff)
    base = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    stamped = base + pd.Timedelta(hours=hour, minutes=minute)
    mask = stamped.notna()
    out = pd.Series(pd.NaT, index=stamped.index, dtype="datetime64[ns]")
    if mask.any():
        localized = stamped[mask].dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        out = out.astype(object)
        out[mask] = localized
    return out


def align_news_to_sessions(
    news: pd.DataFrame,
    prices: pd.DataFrame | dict[str, np.ndarray],
    *,
    cutoff: str = config.SESSION_CUTOFF,
    tz: str = config.TIMEZONE,
) -> pd.DataFrame:
    """Anchor each news row to the trading session it may first inform.

    Parameters
    ----------
    news:
        Rows with ``ticker`` and ``published_at`` plus any payload columns (title/body,
        the :data:`PROB_COLS`). Rows are preserved; original columns pass through.
    prices:
        Either raw price rows or a prebuilt calendar from :func:`trading_sessions`.

    Returns
    -------
    DataFrame
        The input rows with three added columns:

        ``observation_date``
            Normalized trading session the row is anchored to (``NaT`` when unmapped).
        ``as_of``
            tz-aware cutoff datetime of that session; the earliest moment the row is usable.
        ``mapping_status``
            One of ``same_session`` / ``next_session`` (mapped), or ``invalid`` (bad
            timestamp), ``no_calendar`` (unknown ticker), ``unmapped`` (past last session).
    """
    if not {"ticker", "published_at"} <= set(news.columns):
        raise ValueError("news must have 'ticker' and 'published_at' columns.")
    calendar = prices if isinstance(prices, dict) else trading_sessions(prices)
    hour, minute = parse_cutoff(cutoff)
    cutoff_minutes = hour * 60 + minute

    df = news.reset_index(drop=True).copy()
    local = to_local(df["published_at"], tz)
    naive = local.dt.tz_localize(None)
    local_date = naive.dt.normalize()
    after_cutoff = (naive.dt.hour * 60 + naive.dt.minute) > cutoff_minutes
    effective = local_date + pd.to_timedelta(after_cutoff.fillna(False).astype(int), unit="D")

    observation = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    status = pd.Series("invalid", index=df.index, dtype=object)
    valid = local.notna().to_numpy()

    for ticker, positions in df.groupby("ticker", sort=False).groups.items():
        idx = np.asarray(positions)
        row_valid = valid[idx]
        sessions = calendar.get(str(ticker))
        if sessions is None or len(sessions) == 0:
            status.iloc[idx[row_valid]] = "no_calendar"
            continue
        eff = effective.iloc[idx].to_numpy("datetime64[ns]")
        pos = np.searchsorted(sessions, eff, side="left")
        in_range = pos < len(sessions)
        mapped = row_valid & in_range
        status.iloc[idx[row_valid & ~in_range]] = "unmapped"
        if mapped.any():
            observation.iloc[idx[mapped]] = sessions[pos[mapped]]
            status.iloc[idx[mapped]] = "mapped"

    obs_present = observation.notna()
    same = obs_present & (observation == local_date)
    status[same] = "same_session"
    status[obs_present & ~same] = "next_session"

    df["observation_date"] = observation
    df["as_of"] = session_as_of(observation, cutoff, tz)
    df["mapping_status"] = status
    return df


def alignment_report(aligned: pd.DataFrame) -> dict[str, int]:
    """Summarize mapping outcomes as ``{status: count}`` for coverage/leakage auditing."""
    counts = aligned["mapping_status"].value_counts().to_dict()
    report = {status: 0 for status in ("same_session", "next_session", "invalid", "no_calendar", "unmapped")}
    for key, value in counts.items():
        report[str(key)] = int(value)
    report["mapped"] = report["same_session"] + report["next_session"]
    report["dropped"] = report["invalid"] + report["no_calendar"] + report["unmapped"]
    return report
