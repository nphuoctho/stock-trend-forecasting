"""Spike: kéo giá ngày điều chỉnh 10 mã VN30 (2020-2025) qua vnstock -> parquet.

Chỉ để chứng minh nguồn giá khả thi trong ngày spike. Không phải loader chính thức.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TICKERS = ["VCB", "VHM", "HPG", "FPT", "MWG", "VNM", "VIC", "TCB", "MBB", "GAS"]
START, END = "2020-01-01", "2025-12-31"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices"


def fetch_one(symbol: str) -> pd.DataFrame | None:
    """Trả về OHLCV ngày (điều chỉnh) cho 1 mã, hoặc None nếu lỗi."""
    from vnstock import Vnstock

    try:
        q = Vnstock().stock(symbol=symbol, source="VCI").quote
        df = q.history(start=START, end=END, interval="1D")
        if df is None or df.empty:
            return None
        return df
    except Exception as exc:  # noqa: BLE001 - spike: chỉ cần biết mã nào hỏng
        print(f"  ! {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    ok = 0
    for sym in TICKERS:
        df = fetch_one(sym)
        if df is None:
            rows.append((sym, 0, "", "", "FAIL"))
            continue
        df.to_parquet(OUT / f"{sym}.parquet", index=False)
        tcol = "time" if "time" in df.columns else df.columns[0]
        rows.append(
            (sym, len(df), str(df[tcol].min())[:10], str(df[tcol].max())[:10], "OK")
        )
        ok += 1

    print("\n=== KẾT QUẢ GIÁ ===")
    print(f"{'mã':<6}{'dòng':>7}  {'từ':<12}{'đến':<12}status")
    for sym, n, lo, hi, st in rows:
        print(f"{sym:<6}{n:>7}  {lo:<12}{hi:<12}{st}")
    print(f"\n{ok}/{len(TICKERS)} mã OK. Cột mẫu: {list(df.columns) if ok else 'n/a'}")
    return 0 if ok >= 8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
