"""Adjusted OHLCV price loader for the 10 VN30 tickers via vnstock (VCI source).

Uses central config, normalizes the output schema, and reports coverage clearly.
Each ticker is saved to one parquet in data/raw/prices/.

Run:
    uv run python -m stf.cli prices            # full window from config
    uv run python -m stf.cli prices --limit 1  # quick 1-symbol test
"""

from __future__ import annotations

import sys

import pandas as pd

from stf import config

# Normalized output schema: the time column is named "time", the rest are OHLCV.
_EXPECTED_COLS = ("time", "open", "high", "low", "close", "volume")


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Return daily adjusted OHLCV for one ticker, or None on error/empty.

    Import vnstock inside the function to avoid its startup cost when only reading old parquet.
    """
    from vnstock import Vnstock

    try:
        quote = Vnstock().stock(symbol=symbol, source="VCI").quote
        df = quote.history(start=start, end=end, interval="1D")
    except Exception as exc:  # noqa: BLE001 - record the broken ticker, don't kill the whole job
        print(f"  ! {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    return _normalize(df, symbol)


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize columns to the _EXPECTED_COLS schema and add a ticker column."""
    df = df.copy()
    # vnstock returns a "time" column; rename the first column to "time" just in case.
    if "time" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time"})
    df["time"] = pd.to_datetime(df["time"])
    df["ticker"] = symbol
    keep = [c for c in _EXPECTED_COLS if c in df.columns] + ["ticker"]
    return df[keep].sort_values("time").reset_index(drop=True)


def collect(
    tickers: tuple[str, ...] | None = None,
    start: str = config.DATE_START,
    end: str = config.DATE_END,
    *,
    limit: int | None = None,
) -> dict[str, int]:
    """Pull prices for the ticker list, save parquet, return {ticker: row count}.

    limit: take only the first `limit` tickers (for a quick check, not a full pull).
    """
    config.ensure_dirs()
    tickers = tickers or config.TICKERS
    if limit is not None:
        tickers = tickers[:limit]

    result: dict[str, int] = {}
    for sym in tickers:
        df = fetch_one(sym, start, end)
        if df is None:
            result[sym] = 0
            print(f"[prices] {sym}: FAIL")
            continue
        out = config.PRICES_DIR / f"{sym}.parquet"
        df.to_parquet(out, index=False)
        result[sym] = len(df)
        lo, hi = str(df["time"].min())[:10], str(df["time"].max())[:10]
        print(f"[prices] {sym}: {len(df)} phiên ({lo} -> {hi})")
    return result


def summary(result: dict[str, int]) -> None:
    """Print a price coverage summary."""
    ok = sum(1 for n in result.values() if n > 0)
    total = sum(result.values())
    print(f"\n[prices] {ok}/{len(result)} mã OK, tổng {total} dòng giá")
