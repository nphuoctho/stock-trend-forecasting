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
    return 0 if result and all(value > 0 for value in result.values()) else 1


def cmd_news(args: argparse.Namespace) -> int:
    from stf.data import news

    articles = news.crawl(
        refresh=args.refresh, limit_urls=args.limit_urls, max_new=args.batch
    )
    news.summary(articles)
    return 0 if len(articles) else 1


def cmd_verify(_args: argparse.Namespace) -> int:
    """Read saved data and print a coverage report without using the network."""
    print("Data coverage report (offline)\n")
    ok = True

    required_price_cols = {"time", "open", "high", "low", "close", "volume"}
    price_files = {
        path.stem: path for path in config.PRICES_DIR.glob("*.parquet")
    }
    missing_tickers = sorted(set(config.TICKERS) - price_files.keys())
    if missing_tickers:
        print(f"[prices] missing tickers: {', '.join(missing_tickers)}")
        ok = False
    total_prices = 0
    for ticker, path in sorted(price_files.items()):
        df = pd.read_parquet(path)
        missing_cols = required_price_cols - set(df.columns)
        if missing_cols or df.empty:
            print(f"[prices] {ticker}: invalid or empty file")
            ok = False
            continue
        if "ticker" not in df.columns:
            print(f"[prices] {ticker}: legacy file without ticker column")
        total_prices += len(df)
    if price_files:
        print(f"[prices] {len(price_files)} files, {total_prices} sessions total")
    else:
        print("[prices] no price data")
        ok = False

    if not config.ARTICLES_PQ.exists():
        print("[news] articles.parquet missing")
        ok = False
    else:
        articles = pd.read_parquet(config.ARTICLES_PQ)
        required = {"url", "published_at", "title", "body"}
        if not required.issubset(articles.columns) or articles.empty:
            print("[news] articles.parquet has an invalid schema or no rows")
            ok = False
        else:
            with_ts = pd.to_datetime(
                articles["published_at"], errors="coerce", utc=True
            ).notna().sum()
            with_body = (
                articles["body"].fillna("").astype("string").str.strip().ne("").sum()
            )
            print(
                f"[news] {len(articles)} articles | {with_ts} with timestamp | "
                f"{with_body} with body"
            )

    if config.LISTINGS_PQ.exists():
        listings = pd.read_parquet(config.LISTINGS_PQ)
        required = {"ticker", "url", "list_date"}
        if not required.issubset(listings.columns):
            print("[map] listings.parquet has an invalid schema")
            ok = False
        else:
            print(
                f"[map] {len(listings)} listings, {listings['ticker'].nunique()} tickers "
                "(news to ticker)"
            )
    else:
        print("[map] listings.parquet missing")
        ok = False

    print("\nResult:", "OK" if ok else "MISSING OR INVALID DATA")
    return 0 if ok else 1


def cmd_sentiment_smoke(args: argparse.Namespace) -> int:
    """Smoke-test the PhoBERT pipeline on FAKE data, just to prove the code runs.

    Not real results. Few samples + 1 epoch to check train/eval end-to-end.
    """
    from stf.sentiment import dataset, model

    print("PhoBERT smoke-test (fake data, not real results)\n")
    df = dataset.synthetic_dataset(n=args.n, seed=42)
    split = dataset.make_split(df, seed=42, time_aware=False)
    print("Split:", split.describe())
    cfg = model.TrainConfig(epochs=1.0, batch_size=8)
    print(f"Device: {model.get_device()} | model: {cfg.model_name}\n")
    manifest = model.fine_tune(split, cfg)
    print("\nTest macro-F1:", manifest["test_metrics"]["macro_f1"])
    print("Manifest saved to:", (config.SENTIMENT_DIR / "manifest.json"))
    return 0


def cmd_sentiment_train(args: argparse.Namespace) -> int:
    """Fine-tune PhoBERT on a real LABEL FILE (.csv/.parquet with text, label columns)."""
    from stf.sentiment import dataset, model

    df = dataset.load_labeled(args.data)
    split = dataset.make_split(df, seed=args.seed, time_aware=not args.no_time_split)
    print("Split:", split.describe())
    cfg = model.TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
    )
    print(f"Device: {model.get_device()} | model: {cfg.model_name}\n")
    manifest = model.fine_tune(split, cfg)
    print("\nTest macro-F1:", manifest["test_metrics"]["macro_f1"])
    print("Per-class F1:", manifest["test_metrics"]["per_class_f1"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stf", description="Stock Trend Forecasting CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prices = sub.add_parser("prices", help="fetch adjusted OHLCV prices")
    p_prices.add_argument(
        "--limit", type=int, default=None, help="only take the first N tickers"
    )
    p_prices.set_defaults(func=cmd_prices)

    p_news = sub.add_parser("news", help="crawl news (timestamp + title + body)")
    p_news.add_argument(
        "--limit-urls", type=int, default=None, help="only take the first N articles"
    )
    p_news.add_argument(
        "--batch",
        type=int,
        default=None,
        help="crawl N more articles missing a body, then stop (for cron)",
    )
    p_news.add_argument("--refresh", action="store_true", help="re-scrape the listing")
    p_news.set_defaults(func=cmd_news)

    p_verify = sub.add_parser(
        "verify", help="coverage report for existing data (offline)"
    )
    p_verify.set_defaults(func=cmd_verify)

    p_smoke = sub.add_parser(
        "sentiment-smoke", help="smoke-test PhoBERT on fake data (verify code runs)"
    )
    p_smoke.add_argument("--n", type=int, default=60, help="number of fake samples")
    p_smoke.set_defaults(func=cmd_sentiment_smoke)

    p_train = sub.add_parser(
        "sentiment-train", help="fine-tune PhoBERT on a real label file"
    )
    p_train.add_argument(
        "--data", required=True, help="label file .csv/.parquet (text, label columns)"
    )
    p_train.add_argument("--epochs", type=float, default=3.0)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument(
        "--no-time-split",
        action="store_true",
        help="random split instead of time-based split",
    )
    p_train.set_defaults(func=cmd_sentiment_train)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
