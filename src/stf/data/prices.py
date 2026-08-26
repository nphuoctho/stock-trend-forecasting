"""Loader giá OHLCV điều chỉnh cho 10 mã VN30 qua vnstock (nguồn VCI).

Thay cho spike_prices.py: dùng config tập trung, chuẩn hóa schema đầu ra và
báo cáo coverage rõ ràng. Mỗi mã lưu 1 parquet trong data/raw/prices/.

Chạy:
    uv run python -m stf.cli prices            # toàn bộ cửa sổ config
    uv run python -m stf.cli prices --limit 1  # thử nhanh 1 mã
"""

from __future__ import annotations

import sys

import pandas as pd

from stf import config

# Schema chuẩn hóa đầu ra: cột thời gian tên "time", còn lại OHLCV.
_EXPECTED_COLS = ("time", "open", "high", "low", "close", "volume")


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Trả OHLCV ngày (điều chỉnh) cho 1 mã, hoặc None nếu lỗi/rỗng.

    Import vnstock trong hàm để tránh chi phí khởi tạo khi chỉ đọc parquet cũ.
    """
    from vnstock import Vnstock

    try:
        quote = Vnstock().stock(symbol=symbol, source="VCI").quote
        df = quote.history(start=start, end=end, interval="1D")
    except Exception as exc:  # noqa: BLE001 - ghi nhận mã hỏng, không chết cả job
        print(f"  ! {symbol}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    return _normalize(df, symbol)


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Chuẩn hóa cột về schema _EXPECTED_COLS và thêm cột ticker."""
    df = df.copy()
    # vnstock trả cột "time"; phòng trường hợp tên khác thì đổi về "time".
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
    """Kéo giá cho danh sách mã, lưu parquet, trả {ticker: số dòng}.

    limit: chỉ lấy `limit` mã đầu (dùng để verify nhanh, không kéo toàn bộ).
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
    """In tổng kết coverage giá."""
    ok = sum(1 for n in result.values() if n > 0)
    total = sum(result.values())
    print(f"\n[prices] {ok}/{len(result)} mã OK, tổng {total} dòng giá")
