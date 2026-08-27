"""Unified CLI for the data and model pipeline.

    uv run python -m stf.cli prices [--limit N]
    uv run python -m stf.cli news [--limit-urls N] [--refresh]
    uv run python -m stf.cli verify

`verify` re-reads existing data and prints a coverage report (no network).
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from stf import config


def cmd_prices(args: argparse.Namespace) -> int:
    from stf.data import prices

    result = prices.collect(limit=args.limit)
    prices.summary(result)
    return 0 if any(v > 0 for v in result.values()) else 1


def cmd_news(args: argparse.Namespace) -> int:
    from stf.data import news

    articles = news.crawl(refresh=args.refresh, limit_urls=args.limit_urls,
                          max_new=args.batch)
    news.summary(articles)
    return 0 if len(articles) else 1


def cmd_verify(_args: argparse.Namespace) -> int:
    """Read saved data and print a coverage report. No network access."""
    print("=== VERIFY DỮ LIỆU (offline) ===\n")
    ok = True

    # --- Prices ---
    price_files = sorted(config.PRICES_DIR.glob("*.parquet"))
    if not price_files:
        print("[giá] CHƯA CÓ dữ liệu giá")
        ok = False
    else:
        total = 0
        for f in price_files:
            df = pd.read_parquet(f)
            total += len(df)
        print(f"[giá] {len(price_files)} mã, tổng {total} phiên")

    # --- News ---
    if not config.ARTICLES_PQ.exists():
        print("[tin] CHƯA CÓ articles.parquet")
        ok = False
    else:
        a = pd.read_parquet(config.ARTICLES_PQ)
        with_ts = a["published_at"].notna().sum() if "published_at" in a else 0
        with_body = a["body"].notna().sum() if "body" in a else 0
        print(f"[tin] {len(a)} bài | {with_ts} có timestamp | {with_body} có nội dung")
        if "published_at" in a:
            pa = pd.to_datetime(a["published_at"], errors="coerce", utc=True)
            if pa.notna().any():
                print(f"      khoảng: {str(pa.min())[:10]} -> {str(pa.max())[:10]}")

    if config.LISTINGS_PQ.exists():
        listings = pd.read_parquet(config.LISTINGS_PQ)
        print(f"[map] {len(listings)} listing, {listings['ticker'].nunique()} mã "
              f"(ánh xạ tin->mã)")

    print("\nKẾT QUẢ:", "ĐẠT" if ok else "THIẾU DỮ LIỆU")
    return 0 if ok else 1


def cmd_sentiment_smoke(args: argparse.Namespace) -> int:
    """Smoke-test the PhoBERT pipeline on FAKE data, just to prove the code runs.

    Not real results. Few samples + 1 epoch to check train/eval end-to-end.
    """
    from stf.sentiment import dataset, model

    print("=== SMOKE-TEST PhoBERT (dữ liệu giả, KHÔNG phải kết quả thật) ===\n")
    df = dataset.synthetic_dataset(n=args.n, seed=42)
    split = dataset.make_split(df, seed=42, time_aware=False)
    print("Split:", split.describe())
    cfg = model.TrainConfig(epochs=1.0, batch_size=8)
    print(f"Device: {model.get_device()} | model: {cfg.model_name}\n")
    manifest = model.fine_tune(split, cfg)
    print("\nTest macro-F1:", manifest["test_metrics"]["macro_f1"])
    print("Manifest lưu tại:", (config.SENTIMENT_DIR / "manifest.json"))
    return 0


def cmd_sentiment_train(args: argparse.Namespace) -> int:
    """Fine-tune PhoBERT on a real LABEL FILE (.csv/.parquet with text, label columns)."""
    from stf.sentiment import dataset, model

    df = dataset.load_labeled(args.data)
    split = dataset.make_split(df, seed=args.seed, time_aware=not args.no_time_split)
    print("Split:", split.describe())
    cfg = model.TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    print(f"Device: {model.get_device()} | model: {cfg.model_name}\n")
    manifest = model.fine_tune(split, cfg)
    print("\nTest macro-F1:", manifest["test_metrics"]["macro_f1"])
    print("Per-class F1:", manifest["test_metrics"]["per_class_f1"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stf", description="Stock Trend Forecasting CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prices = sub.add_parser("prices", help="kéo giá OHLCV điều chỉnh")
    p_prices.add_argument("--limit", type=int, default=None, help="chỉ lấy N mã đầu")
    p_prices.set_defaults(func=cmd_prices)

    p_news = sub.add_parser("news", help="crawl tin (timestamp + tiêu đề + nội dung)")
    p_news.add_argument("--limit-urls", type=int, default=None, help="chỉ lấy N bài đầu")
    p_news.add_argument("--batch", type=int, default=None,
                        help="chỉ crawl thêm N bài CHƯA có body rồi dừng (cho cron)")
    p_news.add_argument("--refresh", action="store_true", help="cào lại listing")
    p_news.set_defaults(func=cmd_news)

    p_verify = sub.add_parser("verify", help="báo cáo coverage dữ liệu đã có (offline)")
    p_verify.set_defaults(func=cmd_verify)

    p_smoke = sub.add_parser("sentiment-smoke",
                             help="smoke-test PhoBERT bằng dữ liệu giả (verify code)")
    p_smoke.add_argument("--n", type=int, default=60, help="số mẫu giả")
    p_smoke.set_defaults(func=cmd_sentiment_smoke)

    p_train = sub.add_parser("sentiment-train", help="fine-tune PhoBERT trên file nhãn thật")
    p_train.add_argument("--data", required=True, help="file nhãn .csv/.parquet (cột text,label)")
    p_train.add_argument("--epochs", type=float, default=3.0)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--no-time-split", action="store_true",
                         help="chia ngẫu nhiên thay vì tách thời gian")
    p_train.set_defaults(func=cmd_sentiment_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
