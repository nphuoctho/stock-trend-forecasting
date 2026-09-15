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

    Naive inputs are interpreted in ``tz``; aware inputs are converted. Mixed
    offsets are handled row by row instead of raising.
    """
    converted = []
    for value in pd.Series(values).reset_index(drop=True):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            converted.append(pd.NaT)
        elif parsed.tzinfo is None:
            converted.append(parsed.tz_localize(tz))
        else:
            converted.append(parsed.tz_convert(tz))
    return pd.Series(pd.array(converted, dtype=f"datetime64[ns, {tz}]"))


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


def session_as_of(
    dates, cutoff: str = config.SESSION_CUTOFF, tz: str = config.TIMEZONE
) -> pd.Series:
    """Map trading dates to a tz-aware point-in-time anchor."""
    hour, minute = parse_cutoff(cutoff)
    base = pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce")
    if base.dt.tz is None:
        base = base.dt.tz_localize(tz)
    else:
        base = base.dt.tz_convert(tz)
    return base + pd.Timedelta(hours=hour, minutes=minute)


def align_news_to_sessions(
    news: pd.DataFrame,
    prices: pd.DataFrame | dict[str, np.ndarray],
    *,
    cutoff: str = config.SESSION_CUTOFF,
    tz: str = config.TIMEZONE,
    max_rollforward_days: int = 12,
) -> pd.DataFrame:
    """Anchor each news row to the trading session it may first inform.

    Rows that require more than ``max_rollforward_days`` before the first
    available session are marked ``stale`` and excluded from aggregation.
    """
    if not {"ticker", "published_at"} <= set(news.columns):
        raise ValueError("news must have 'ticker' and 'published_at' columns.")
    if max_rollforward_days < 0:
        raise ValueError("max_rollforward_days must be non-negative.")
    calendar = prices if isinstance(prices, dict) else trading_sessions(prices)
    hour, minute = parse_cutoff(cutoff)
    cutoff_seconds = hour * 3600 + minute * 60

    df = news.reset_index(drop=True).copy()
    local = to_local(df["published_at"], tz)
    naive = local.dt.tz_localize(None)
    local_date = naive.dt.normalize()
    after_cutoff = (
        naive.dt.hour * 3600
        + naive.dt.minute * 60
        + naive.dt.second
        > cutoff_seconds
    )
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
            roll_days = (
                sessions[pos[mapped]] - eff[mapped]
            ).astype("timedelta64[D]").astype(int)
            fresh = mapped.copy()
            fresh[mapped] = roll_days <= max_rollforward_days
            stale = mapped & ~fresh
            status.iloc[idx[stale]] = "stale"
            if fresh.any():
                observation.iloc[idx[fresh]] = sessions[pos[fresh]]
                status.iloc[idx[fresh]] = "mapped"

    obs_present = observation.notna()
    same = obs_present & (observation == local_date)
    status[same] = "same_session"
    status[obs_present & ~same] = "next_session"

    df["observation_date"] = observation
    df["as_of"] = session_as_of(observation, cutoff, tz)
    df["mapping_status"] = status
    return df


def alignment_report(aligned: pd.DataFrame) -> dict[str, int]:
    """Summarize mapping outcomes for coverage and leakage auditing."""
    counts = aligned["mapping_status"].value_counts().to_dict()
    report = {
        status: 0
        for status in (
            "same_session",
            "next_session",
            "invalid",
            "no_calendar",
            "unmapped",
            "stale",
        )
    }
    for key, value in counts.items():
        report[str(key)] = int(value)
    report["mapped"] = report["same_session"] + report["next_session"]
    report["dropped"] = (
        report["invalid"]
        + report["no_calendar"]
        + report["unmapped"]
        + report["stale"]
    )
    return report
