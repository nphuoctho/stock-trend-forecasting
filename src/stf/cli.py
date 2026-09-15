"""Unified CLI for the data and model pipeline.

    uv run python -m stf.cli prices [--limit N]
    uv run python -m stf.cli news [--limit-urls N] [--refresh]
    uv run python -m stf.cli verify

`verify` re-reads existing data and prints a coverage report (no network).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    from stf.forecasting.calendar import to_local

    print("Data coverage report (offline)\n")
    ok = True
    articles: pd.DataFrame | None = None
    listings: pd.DataFrame | None = None

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
        try:
            numeric = df[["open", "high", "low", "close", "volume"]].apply(
                pd.to_numeric, errors="coerce"
            )
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError("non-finite OHLCV values")
            if (numeric["close"] <= 0).any() or (numeric["volume"] <= 0).any():
                raise ValueError("close and volume must be positive")
            dates = to_local(df["time"], config.TIMEZONE).dt.tz_localize(None)
            dates = dates.dt.normalize()
            if dates.isna().any() or dates.duplicated().any():
                raise ValueError("invalid or duplicate normalized trading dates")
        except (TypeError, ValueError) as exc:
            print(f"[prices] {ticker}: invalid values ({exc})")
            ok = False
            continue
        if "ticker" not in df.columns:
            print(f"[prices] {ticker}: legacy file without ticker column")
        elif set(df["ticker"].dropna().astype(str)) != {ticker}:
            print(f"[prices] {ticker}: ticker column does not match file name")
            ok = False
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

    if (
        articles is not None
        and listings is not None
        and {"url", "published_at", "title", "body"} <= set(articles.columns)
        and {"ticker", "url"} <= set(listings.columns)
    ):
        from stf.data.news import join_listings_articles

        joined = join_listings_articles(listings, articles)
        print(
            f"[map] joined {len(joined)} ticker-article links, "
            f"{joined['url'].nunique()} unique article urls"
        )

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
    """Fine-tune PhoBERT on a real LABEL FILE."""
    from stf.sentiment import dataset, model

    df = dataset.load_labeled(args.data, allow_preliminary=args.allow_preliminary)
    df = dataset.prepare_model_input(df, args.input_variant)
    split = dataset.make_split(df, seed=args.seed, time_aware=not args.no_time_split)
    print("Split:", split.describe())
    cfg = model.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        truncation_strategy=args.truncation_strategy,
        class_weighting=args.class_weighting,
    )
    print(f"Device: {model.get_device()} | model: {cfg.model_name}\n")
    manifest = model.fine_tune(split, cfg, source_path=args.data)
    print("\nTest macro-F1:", manifest["test_metrics"]["macro_f1"])
    print("Per-class F1:", manifest["test_metrics"]["per_class_f1"])
    return 0


def cmd_sentiment_cv(args: argparse.Namespace) -> int:
    """Run one leakage-safe input/truncation configuration with 5-fold CV."""
    from stf.sentiment import dataset, experiments, model

    df = dataset.load_labeled(args.data, allow_preliminary=args.allow_preliminary)
    cfg = model.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        class_weighting=args.class_weighting,
    )
    result = experiments.run_cross_validation(
        df,
        input_variant=args.input_variant,
        truncation_strategy=args.truncation_strategy,
        cfg=cfg,
        folds=args.folds,
        seed=args.seed,
        out_dir=Path(args.output),
        allow_preliminary=args.allow_preliminary,
        source_path=args.data,
    )
    print("Data:", result["data_size"], "| folds:", result["folds"])
    for metric, values in result["aggregate"].items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print("Results:", Path(args.output) / "cv_results.json")
    return 0


def cmd_sentiment_ablation(args: argparse.Namespace) -> int:
    """Run all title/context and truncation combinations sequentially."""
    from stf.sentiment import dataset, experiments, model

    df = dataset.load_labeled(args.data, allow_preliminary=args.allow_preliminary)
    cfg = model.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        class_weighting=args.class_weighting,
    )
    summary = experiments.run_ablation(
        df,
        cfg=cfg,
        folds=args.folds,
        seed=args.seed,
        out_dir=Path(args.output),
        allow_preliminary=args.allow_preliminary,
        source_path=args.data,
    )
    print(summary.to_string(index=False))
    print("Summary:", Path(args.output) / "ablation_summary.csv")
    return 0


def cmd_score_news(args: argparse.Namespace) -> int:
    """Score crawled ticker articles with a trained PhoBERT checkpoint.

    Loads the persisted listing/article join, builds the selected model input
    variant, and writes per-article sentiment probabilities to a parquet file.
    Requires a checkpoint already produced by sentiment-train/-cv/-ablation.
    """
    from stf.data.news import load_ticker_articles
    from stf.sentiment.dataset import build_input_text
    from stf.sentiment.model import predict_proba

    articles = load_ticker_articles()
    if args.limit is not None:
        articles = articles.head(args.limit)
    if articles.empty:
        print("score-news: no ticker/article rows to score.", file=sys.stderr)
        return 1
    scored = build_input_text(articles, args.input_variant)
    probs = predict_proba(
        scored["text"].tolist(),
        Path(args.model_dir),
        batch_size=args.batch_size,
        truncation_strategy=args.truncation_strategy,
    )
    out = pd.DataFrame(
        {
            "ticker": scored["ticker"].to_numpy(),
            "url": scored["url"].to_numpy(),
            "published_at": scored["published_at"].to_numpy(),
            "prob_negative": probs[:, 0],
            "prob_neutral": probs[:, 1],
            "prob_positive": probs[:, 2],
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"[score-news] scored {len(out)} rows -> {output_path}")
    return 0


def _forecast_smoke_prices(n: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(
        100.0 + np.cumsum(np.where(np.arange(n) % 3 == 0, 1.5, -0.4)),
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "ticker": "FPT",
            "time": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000 + pd.Series(range(n), dtype="int64") * 10_000,
        }
    )


def cmd_forecast_smoke(args: argparse.Namespace) -> int:
    """Run the forecasting panel and LSTM path on deterministic local data."""

    from stf.forecasting import (
        SENTIMENT_COLUMNS,
        TREND_LABELS,
        FeatureScaler,
        assemble,
        chronological_split,
        fit_thresholds,
        label_panel,
        make_sequences,
        price_feature_columns,
    )
    from stf.forecasting.models import (
        PriceLSTM,
        PriceSentimentLSTM,
        fit_lstm,
        make_two_branch_sequences,
    )

    if args.n < 12:
        print("forecast-smoke: --n must be at least 12.", file=sys.stderr)
        return 2
    if args.window < 1 or args.epochs < 1:
        print("forecast-smoke: --window and --epochs must be positive.", file=sys.stderr)
        return 2
    prices = _forecast_smoke_prices(args.n)
    dates = prices["time"]
    news = pd.DataFrame(
        {
            "ticker": "FPT",
            "published_at": dates.dt.strftime("%Y-%m-%d") + " 14:00:00+07:00",
            "title": "Tin thử nghiệm",
            "prob_negative": 0.2,
            "prob_neutral": 0.3,
            "prob_positive": 0.5,
        }
    )
    panel = assemble(prices, news)
    split = chronological_split(panel, val_frac=0.2, test_frac=0.2)
    train_returns = panel.iloc[split.train]["target_return"].dropna().to_numpy()
    thresholds = fit_thresholds(train_returns)
    labeled = label_panel(panel, thresholds)
    feature_cols = price_feature_columns()
    sent_cols = list(SENTIMENT_COLUMNS)
    price_scaler = FeatureScaler.fit(labeled.iloc[split.train], feature_cols)
    sent_scaler = FeatureScaler.fit(labeled.iloc[split.train], sent_cols)
    scaled = sent_scaler.transform(price_scaler.transform(labeled))
    X, y, meta = make_sequences(scaled, feature_cols, seq_len=args.window)
    if len(X) == 0:
        raise RuntimeError("Forecast smoke test produced no finite labeled sequences.")
    X_price, X_sent, y2, meta2 = make_two_branch_sequences(
        scaled, feature_cols, sent_cols, seq_len=args.window
    )
    if len(X_price) == 0:
        raise RuntimeError("Forecast smoke test produced no finite two-branch sequences.")
    train_dates = set(pd.to_datetime(panel.iloc[split.train]["target_date"]).dropna())
    val_dates = set(pd.to_datetime(panel.iloc[split.val]["target_date"]).dropna())
    test_dates = set(pd.to_datetime(panel.iloc[split.test]["target_date"]).dropna())
    train_mask = meta["target_date"].isin(train_dates).to_numpy()
    val_mask = meta["target_date"].isin(val_dates).to_numpy()
    test_mask = meta["target_date"].isin(test_dates).to_numpy()
    train_mask2 = meta2["target_date"].isin(train_dates).to_numpy()
    if not train_mask.any() or not train_mask2.any():
        raise RuntimeError("Forecast smoke test produced no training sequences.")
    net = PriceLSTM(len(feature_cols), hidden=8, num_layers=1, num_classes=len(TREND_LABELS))
    history = fit_lstm(net, X[train_mask], y[train_mask], epochs=args.epochs, seed=args.seed)
    net2 = PriceSentimentLSTM(
        len(feature_cols),
        len(sent_cols),
        hidden=8,
        num_layers=1,
        num_classes=len(TREND_LABELS),
    )
    history2 = fit_lstm(
        net2,
        X_price[train_mask2],
        y2[train_mask2],
        X_sent=X_sent[train_mask2],
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        f"Forecast smoke: panel={len(labeled)} rows | "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)} | "
        f"sequences=train:{train_mask.sum()} val:{val_mask.sum()} test:{test_mask.sum()} | "
        f"price_lstm_final_loss={history[-1]:.6f} | "
        f"price_sentiment_lstm_final_loss={history2[-1]:.6f}"
    )
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
        "--input-variant",
        choices=("title", "context", "title_context"),
        default="title",
        help="text supplied to PhoBERT",
    )
    p_train.add_argument(
        "--truncation-strategy",
        choices=("head", "tail", "head_tail"),
        default="head",
        help="how inputs longer than 256 tokens are shortened",
    )
    p_train.add_argument(
        "--class-weighting",
        choices=("none", "inverse_frequency"),
        default="none",
        help="class weighting computed from the training split",
    )
    p_train.add_argument(
        "--no-time-split",
        action="store_true",
        help="random split instead of time-based split",
    )
    p_train.add_argument(
        "--allow-preliminary",
        action="store_true",
        help="include unreviewed preliminary in-domain labels (diagnostics only)",
    )
    p_train.set_defaults(func=cmd_sentiment_train)

    def add_experiment_args(subparser):
        subparser.add_argument("--data", required=True)
        subparser.add_argument("--epochs", type=float, default=3.0)
        subparser.add_argument("--batch-size", type=int, default=16)
        subparser.add_argument("--folds", type=int, default=5)
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument(
            "--input-variant",
            choices=("title", "context", "title_context"),
            default="title",
        )
        subparser.add_argument(
            "--truncation-strategy",
            choices=("head", "tail", "head_tail"),
            default="head",
        )
        subparser.add_argument(
            "--class-weighting",
            choices=("none", "inverse_frequency"),
            default="none",
            help="class weighting computed from each training fold",
        )
        subparser.add_argument(
            "--allow-preliminary",
            action="store_true",
            help="include unreviewed preliminary in-domain labels (diagnostics only)",
        )
        subparser.add_argument("--output", required=True)

    p_cv = sub.add_parser(
        "sentiment-cv", help="evaluate one PhoBERT configuration with K-fold CV"
    )
    add_experiment_args(p_cv)
    p_cv.set_defaults(func=cmd_sentiment_cv)

    p_ablation = sub.add_parser(
        "sentiment-ablation",
        help="evaluate all title/context and truncation combinations",
    )
    p_ablation.add_argument("--data", required=True)
    p_ablation.add_argument("--epochs", type=float, default=3.0)
    p_ablation.add_argument("--batch-size", type=int, default=16)
    p_ablation.add_argument("--folds", type=int, default=5)
    p_ablation.add_argument("--seed", type=int, default=42)
    p_ablation.add_argument(
        "--class-weighting",
        choices=("none", "inverse_frequency"),
        default="none",
        help="class weighting computed from each training fold",
    )
    p_ablation.add_argument(
        "--allow-preliminary",
        action="store_true",
        help="include unreviewed preliminary in-domain labels (diagnostics only)",
    )
    p_ablation.add_argument("--output", required=True)
    p_ablation.set_defaults(func=cmd_sentiment_ablation)


    p_forecast_smoke = sub.add_parser(
        "forecast-smoke",
        help="run the point-in-time panel and LSTM path on deterministic data",
    )
    p_forecast_smoke.add_argument("--n", type=int, default=40)
    p_forecast_smoke.add_argument("--window", type=int, default=5)
    p_forecast_smoke.add_argument("--epochs", type=int, default=2)
    p_forecast_smoke.add_argument("--seed", type=int, default=42)
    p_forecast_smoke.set_defaults(func=cmd_forecast_smoke)

    p_score = sub.add_parser(
        "score-news", help="score crawled ticker articles with a trained checkpoint"
    )
    p_score.add_argument(
        "--model-dir", required=True, help="trained checkpoint directory"
    )
    p_score.add_argument(
        "--input-variant",
        choices=("title", "context", "title_context"),
        default="title",
    )
    p_score.add_argument(
        "--truncation-strategy",
        choices=("head", "tail", "head_tail"),
        default=None,
        help="override the checkpoint's saved truncation strategy",
    )
    p_score.add_argument("--batch-size", type=int, default=32)
    p_score.add_argument(
        "--limit", type=int, default=None, help="only score the first N rows"
    )
    p_score.add_argument("--output", required=True, help="output parquet path")
    p_score.set_defaults(func=cmd_score_news)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
