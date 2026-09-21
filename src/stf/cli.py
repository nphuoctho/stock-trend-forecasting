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
        eval_strata=tuple(args.eval_strata) if args.eval_strata else None,
    )
    print("Data:", result["data_size"], "| folds:", result["folds"])
    if result.get("eval_strata"):
        print(
            "Holdout restricted to strata",
            result["eval_strata"],
            f"| evaluated rows: {result['holdout_pool_size']}",
        )
    for metric, values in result["aggregate"].items():
        print(f"{metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print("Results:", Path(args.output) / "cv_results.json")
    return 0




def cmd_sentiment_refit(args: argparse.Namespace) -> int:
    """Fit a cross-validated PhoBERT configuration on every reviewed label."""
    from stf.sentiment import dataset, experiments, model

    df = dataset.load_labeled(args.data)
    frame = dataset.prepare_model_input(df, args.input_variant)
    cfg = model.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        truncation_strategy=args.truncation_strategy,
        class_weighting=args.class_weighting,
    )
    evaluation_reference = experiments.validate_full_refit_reference(
        args.cv_results,
        frame,
        input_variant=args.input_variant,
        cfg=cfg,
        source_path=args.data,
    )
    print(
        f"Refitting {len(frame)} reviewed rows with the locked "
        f"{evaluation_reference['folds']}-fold configuration."
    )
    manifest = model.refit_full_data(
        frame,
        cfg,
        out_dir=Path(args.output),
        source_path=args.data,
        evaluation_reference=evaluation_reference,
    )
    print("Checkpoint:", Path(args.output) / "best")
    print("Manifest:", Path(args.output) / "manifest.json")
    print("Training rows:", manifest["training_size"])
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
    Requires a checkpoint produced by sentiment-train or sentiment-cv; ablation
    outputs metrics for selection and removes its fold models to limit disk use.
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
    scored = build_input_text(
        articles, args.input_variant, context_chars=args.context_chars
    )
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
        f"price_lstm_final_loss={history['loss_history'][-1]:.6f} | "
        f"price_sentiment_lstm_final_loss={history2['loss_history'][-1]:.6f}"
    )
    return 0


def _load_prices() -> pd.DataFrame:
    """Concatenate every configured ticker's persisted OHLCV parquet."""
    frames = []
    for ticker in config.TICKERS:
        path = config.PRICES_DIR / f"{ticker}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing price file {path}; run `stf.cli prices` first.")
        frame = pd.read_parquet(path)
        if "ticker" not in frame.columns:
            frame = frame.assign(ticker=ticker)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run the real walk-forward forecasting ladder and export its metrics.

    Without ``--news-sentiment`` every session is treated as a no-news day, which
    yields the price-only ladder. With a scored-news parquet the two-branch arms
    receive real sentiment and the sentiment contribution is reported as a paired
    per-window delta with a bootstrap interval.
    """
    from stf.forecasting import assemble
    from stf.forecasting.experiment import ForecastConfig, frame_hash, run_experiment

    prices = _load_prices()
    provenance: dict = {
        "tickers": list(config.TICKERS),
        "date_start": config.DATE_START,
        "date_end": config.DATE_END,
        "session_cutoff": config.SESSION_CUTOFF,
        "timezone": config.TIMEZONE,
        "prices_hash": frame_hash(prices),
        "news_sentiment": None,
    }

    news = None
    news_path_arg = args.news_sentiment or args.neutral_news_sentiment
    if news_path_arg is not None:
        news_path = Path(news_path_arg)
        if not news_path.exists():
            print(f"forecast: missing {news_path}", file=sys.stderr)
            return 2
        news = pd.read_parquet(news_path)
        source_hash = frame_hash(news)
        mode = "real"
        if args.neutral_news_sentiment is not None:
            probability_columns = [
                "prob_negative",
                "prob_neutral",
                "prob_positive",
            ]
            missing = set(probability_columns) - set(news.columns)
            if missing:
                print(
                    "forecast: neutral control is missing probability columns "
                    f"{sorted(missing)}",
                    file=sys.stderr,
                )
                return 2
            news = news.copy()
            news.loc[:, probability_columns] = (0.0, 1.0, 0.0)
            mode = "neutral_prior"
        provenance["news_sentiment"] = {
            "mode": mode,
            "source_path": str(news_path),
            "source_hash": source_hash,
            "feature_hash": frame_hash(news),
            "rows": int(len(news)),
        }

    panel = assemble(prices, news)
    if "alignment_report" in panel.attrs:
        provenance["alignment_report"] = panel.attrs["alignment_report"]
    provenance["panel_hash"] = frame_hash(panel)
    news_days = int(panel["has_news"].sum())
    provenance["news_coverage"] = {
        "panel_rows": int(len(panel)),
        "rows_with_news": news_days,
        "fraction": round(news_days / len(panel), 6) if len(panel) else 0.0,
    }

    cfg = ForecastConfig(
        seq_len=args.seq_len,
        n_windows=args.windows,
        test_size=args.test_size,
        val_size=args.val_size,
        expanding=not args.rolling,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=tuple(args.seeds),
    )
    record = run_experiment(
        panel, cfg=cfg, output_dir=Path(args.output), provenance=provenance
    )

    print(
        f"[forecast] panel={record['panel_rows']} rows | "
        f"news coverage={provenance['news_coverage']['fraction']:.3f} | "
        f"windows={len(record['windows'])} | seeds={list(cfg.seeds)} | "
        f"chance={record['chance_level']:.3f}"
    )
    for arm, metrics in record["summary"].items():
        mean = metrics["macro_f1"]["mean"]
        std = metrics["macro_f1"]["std"]
        bal = metrics["balanced_accuracy"]["mean"]
        print(f"  {arm:<26} macro_f1={mean:.4f} (sd {std:.4f})  balanced_acc={bal:.4f}")
    for pair, metrics in record["ablation"].items():
        print(f"  D {pair}:")
        for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
            delta = metrics[metric]
            win = delta["window_bootstrap"]
            date = delta["date_block_bootstrap"]

            def span(ci: dict) -> str:
                low, high = ci.get("low"), ci.get("high")
                return "n/a" if low is None else f"[{low:+.4f}, {high:+.4f}]"

            print(
                f"      {metric:<18} {win['mean']:+.4f}  "
                f"window-CI {span(win)}  date-CI {span(date)}"
            )
    print(f"[forecast] artifacts -> {args.output}")
    return 0


def cmd_label_candidates(args: argparse.Namespace) -> int:
    """Draw a stratified annotation batch for expanding the in-domain label set.

    Produces one random evaluation stratum, which preserves the corpus label
    prior, plus minority-enriched training strata selected by a Vietnamese
    financial polarity lexicon and the current checkpoint's probabilities. The
    retrieval signals choose what a human reads; they never assign a label.
    """
    from stf.data.news import load_ticker_articles
    from stf.sentiment.sampling import (
        DEFAULT_QUOTAS,
        SamplingPlan,
        draw_candidates,
        prepare_pool,
        sampling_manifest,
        to_label_template,
        write_manifest,
    )

    articles = load_ticker_articles()
    exclude: set[str] = set()
    for path in args.exclude or []:
        frame = pd.read_csv(path)
        if "url" not in frame.columns:
            print(f"label-candidates: {path} has no 'url' column.", file=sys.stderr)
            return 2
        exclude.update(frame["url"].dropna().astype(str))

    scored = None
    if args.news_sentiment is not None:
        scored_path = Path(args.news_sentiment)
        if not scored_path.exists():
            print(f"label-candidates: missing {scored_path}", file=sys.stderr)
            return 2
        scored = pd.read_parquet(scored_path)

    quotas = dict(DEFAULT_QUOTAS)
    for stratum, value in (
        ("eval_random", args.eval_random),
        ("train_negative", args.negative),
        ("train_positive", args.positive),
        ("train_active", args.active),
    ):
        if value is not None:
            quotas[stratum] = value
    plan = SamplingPlan(quotas=quotas, seed=args.seed, body_context_chars=args.context_chars)

    pool = prepare_pool(articles, scored=scored, exclude_urls=exclude)
    if pool.empty:
        print("label-candidates: candidate pool is empty.", file=sys.stderr)
        return 1
    candidates = draw_candidates(pool, plan)
    template = to_label_template(candidates, plan)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output, index=False)
    manifest = sampling_manifest(pool, candidates, plan, excluded=len(exclude))
    manifest_path = output.with_name(f"{output.stem}_manifest.json")
    write_manifest(manifest_path, manifest)

    print(
        f"[label-candidates] pool={manifest['pool']['candidates_after_dedup']} "
        f"(excluded {len(exclude)} already-labelled) -> drew {len(template)} rows"
    )
    for stratum, info in manifest["strata"].items():
        prob = info.get("inclusion_probability")
        share = "n/a" if prob is None else f"{prob:.4f}"
        print(
            f"  {stratum:<20} drawn={info['drawn']:<5} eligible={info['eligible']:<6} "
            f"inclusion_p={share}"
        )
    print(f"[label-candidates] {output} + {manifest_path}")
    return 0


def cmd_label_audit(args: argparse.Namespace) -> int:
    """Validate a filled annotation file and report distribution and precision."""
    from stf.sentiment.labels import LABELS
    from stf.sentiment.sampling import audit_labels

    path = Path(args.data)
    if not path.exists():
        print(f"label-audit: missing {path}", file=sys.stderr)
        return 2
    frame = pd.read_csv(path)
    report = audit_labels(frame, valid_labels=tuple(LABELS))

    print(
        f"[label-audit] {path}: {report['labelled']}/{report['rows']} labelled, "
        f"{report['pending']} pending"
    )
    print(f"  distribution: {report['distribution']}")
    for stratum, counts in report["by_stratum"].items():
        print(f"    {stratum:<20} {counts}")
    full = report["power_full"]
    fold = report["power_5fold"]
    print(
        f"  smallest class n={report['smallest_class']}: recall 95% CI half-width "
        f"+/-{full['ci_half_width']:.3f} on the full set, "
        f"+/-{fold['ci_half_width']:.3f} per 5-fold test block"
    )
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ISSUE: {issue}", file=sys.stderr)
        return 1
    print("  no structural issues found")
    return 0


def cmd_label_finalize(args: argparse.Namespace) -> int:
    """Mark reviewed annotation rows as human-reviewed so training accepts them.

    Running this command is the act of attesting that a human has confirmed every
    row carrying a valid label. Both provenance fields are rewritten, because the
    training loader rejects a row when either one still marks it preliminary.
    """
    from stf.sentiment.labels import LABELS
    from stf.sentiment.sampling import finalize_labels

    path = Path(args.data)
    if not path.exists():
        print(f"label-finalize: missing {path}", file=sys.stderr)
        return 2
    frame = pd.read_csv(path)
    out, report = finalize_labels(frame, valid_labels=tuple(LABELS))
    if report["finalized"] == 0:
        print("label-finalize: no row carries a valid label.", file=sys.stderr)
        return 1

    target = Path(args.output) if args.output else path
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(
        f"[label-finalize] {report['finalized']}/{report['rows']} rows marked reviewed"
        f" ({report['left_preliminary']} left preliminary) -> {target}"
    )
    print(f"  distribution: {report['distribution']}")
    if report["changed_from_prelabel"] is not None:
        print(
            f"  differs from preliminary label: {report['changed_from_prelabel']} rows"
            " (diagnostic of the prelabelling step, not an inter-rater measure)"
        )
    return 0


def cmd_label_merge(args: argparse.Namespace) -> int:
    """Union several annotation files into one training file with unique ids."""
    from stf.sentiment.labels import LABELS
    from stf.sentiment.sampling import merge_label_files

    sources: dict[str, pd.DataFrame] = {}
    for spec in args.inputs:
        if "=" not in spec:
            print(
                f"label-merge: expected <batch>=<path>, got {spec!r}", file=sys.stderr
            )
            return 2
        batch, raw_path = spec.split("=", 1)
        source = Path(raw_path)
        if not source.exists():
            print(f"label-merge: missing {source}", file=sys.stderr)
            return 2
        sources[batch] = pd.read_csv(source)

    merged, report = merge_label_files(sources, valid_labels=tuple(LABELS))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)

    print(f"[label-merge] {report['rows']} rows -> {output}")
    for batch, info in report["per_batch"].items():
        print(f"  {batch:<10} {info['rows']:<5} {info['distribution']}")
    print(f"  duplicate urls removed: {report['duplicate_urls_removed']}")
    print(f"  total distribution:     {report['distribution']}")
    print(
        f"  evaluable rows:         {report['evaluable_rows']} "
        f"{report['evaluable_distribution']}"
    )
    print(f"  train-only rows:        {report['train_only_rows']}")
    full = report["power_eval_full"]
    fold = report["power_eval_per_fold"]
    print(
        f"  smallest evaluable class n={full['n_minority']}: recall 95% CI half-width"
        f" +/-{full['ci_half_width']:.3f} overall, +/-{fold['ci_half_width']:.3f} per fold"
    )
    return 0


def cmd_forecast_compare(args: argparse.Namespace) -> int:
    """Split the two-branch arm's gain into architecture and information effects."""
    import json as _json

    from stf.forecasting.experiment import compare_information_gain

    report = compare_information_gain(
        Path(args.real), Path(args.control), arm=args.arm, price_arm=args.price_arm
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"[forecast-compare] {args.arm} vs {args.price_arm}")
    for metric, effect in report["effects"].items():
        win = effect["information_gain_window_bootstrap"]
        date = effect["information_gain_date_bootstrap"]
        print(f"  {metric}:")
        print(f"      price-only                 {effect['price_only']:.4f}")
        print(f"      two-branch neutral prior   {effect['two_branch_neutral_prior']:.4f}")
        print(f"      two-branch real sentiment  {effect['two_branch_real_sentiment']:.4f}")
        print(f"      architecture effect        {effect['architecture_effect']:+.4f}")
        print(
            f"      information gain           {effect['information_gain']:+.4f}  "
            f"window-CI [{win['low']:+.4f}, {win['high']:+.4f}]  "
            f"date-CI [{date['low']:+.4f}, {date['high']:+.4f}]"
        )
        print(f"      naive delta (conflated)    {effect['naive_delta']:+.4f}")
    print(f"[forecast-compare] -> {output}")
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
    p_cv.add_argument(
        "--eval-strata",
        nargs="*",
        default=None,
        metavar="STRATUM",
        help=(
            "restrict the holdout to these sampling strata (e.g. eval_random "
            "baseline_random); minority-enriched rows then train only"
        ),
    )
    p_cv.set_defaults(func=cmd_sentiment_cv)

    p_refit = sub.add_parser(
        "sentiment-refit",
        help="fit a cross-validated PhoBERT configuration on every reviewed label",
    )
    p_refit.add_argument("--data", required=True)
    p_refit.add_argument(
        "--cv-results",
        required=True,
        help="cv_results.json that selected this exact configuration",
    )
    p_refit.add_argument("--epochs", type=float, required=True)
    p_refit.add_argument("--batch-size", type=int, required=True)
    p_refit.add_argument("--seed", type=int, required=True)
    p_refit.add_argument(
        "--input-variant",
        choices=("title", "context", "title_context"),
        required=True,
    )
    p_refit.add_argument(
        "--truncation-strategy",
        choices=("head", "tail", "head_tail"),
        required=True,
    )
    p_refit.add_argument(
        "--class-weighting",
        choices=("none", "inverse_frequency"),
        required=True,
    )
    p_refit.add_argument("--output", required=True)
    p_refit.set_defaults(func=cmd_sentiment_refit)

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

    p_forecast = sub.add_parser(
        "forecast",
        help="run the real walk-forward forecasting ladder and export metrics",
    )
    news_input = p_forecast.add_mutually_exclusive_group()
    news_input.add_argument(
        "--news-sentiment",
        default=None,
        help="parquet from score-news with observed sentiment probabilities",
    )
    news_input.add_argument(
        "--neutral-news-sentiment",
        default=None,
        help="parquet from score-news; preserve article timing and volume, force neutral probabilities",
    )
    p_forecast.add_argument("--output", default="outputs/forecast", help="artifact directory")
    p_forecast.add_argument("--seq-len", type=int, default=5)
    p_forecast.add_argument("--windows", type=int, default=5, help="walk-forward windows")
    p_forecast.add_argument("--test-size", type=int, default=60, help="test dates per window")
    p_forecast.add_argument("--val-size", type=int, default=60, help="validation dates per window")
    p_forecast.add_argument(
        "--rolling",
        action="store_true",
        help="use a fixed-length rolling train window instead of an expanding one",
    )
    p_forecast.add_argument("--hidden", type=int, default=32)
    p_forecast.add_argument("--epochs", type=int, default=30)
    p_forecast.add_argument("--batch-size", type=int, default=128)
    p_forecast.add_argument("--patience", type=int, default=5)
    p_forecast.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="seeds averaged per window"
    )
    p_forecast.set_defaults(func=cmd_forecast)

    p_cmp = sub.add_parser(
        "forecast-compare",
        help="split the two-branch gain into architecture and information effects",
    )
    p_cmp.add_argument("--real", required=True, help="run directory with observed sentiment")
    p_cmp.add_argument(
        "--control",
        required=True,
        help="run directory with --neutral-news-sentiment from the same scored-news parquet",
    )
    p_cmp.add_argument("--arm", default="lstm_price_sentiment")
    p_cmp.add_argument("--price-arm", default="lstm_price")
    p_cmp.add_argument("--output", default="outputs/forecast/information_gain.json")
    p_cmp.set_defaults(func=cmd_forecast_compare)

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
        "--context-chars",
        type=int,
        default=None,
        help="cap the article context before tokenizing, matching the annotation cap",
    )
    p_score.add_argument(
        "--limit", type=int, default=None, help="only score the first N rows"
    )
    p_score.add_argument("--output", required=True, help="output parquet path")
    p_score.set_defaults(func=cmd_score_news)

    p_cand = sub.add_parser(
        "label-candidates",
        help="draw a stratified annotation batch to expand the in-domain label set",
    )
    p_cand.add_argument(
        "--output",
        default="data/labeled/indomain/to_label_batch2.csv",
        help="annotation file to write",
    )
    p_cand.add_argument(
        "--exclude",
        nargs="*",
        default=["data/labeled/indomain/to_label_r1.csv"],
        help="existing label files whose urls must not be drawn again",
    )
    p_cand.add_argument(
        "--news-sentiment",
        default="data/processed/news_sentiment.parquet",
        help="score-news parquet reused as a retrieval prior",
    )
    p_cand.add_argument(
        "--eval-random",
        type=int,
        default=None,
        help="uniform, model-free stratum; the only one valid for evaluation",
    )
    p_cand.add_argument("--negative", type=int, default=None)
    p_cand.add_argument("--positive", type=int, default=None)
    p_cand.add_argument(
        "--active",
        type=int,
        default=None,
        help="model-informed low-confidence stratum; train-only",
    )
    p_cand.add_argument("--context-chars", type=int, default=2000)
    p_cand.add_argument("--seed", type=int, default=42)
    p_cand.set_defaults(func=cmd_label_candidates)

    p_audit = sub.add_parser(
        "label-audit", help="validate a filled annotation file and report its power"
    )
    p_audit.add_argument("--data", required=True, help="annotation csv to check")
    p_audit.set_defaults(func=cmd_label_audit)

    p_final = sub.add_parser(
        "label-finalize",
        help="mark reviewed rows as human-reviewed so training accepts them",
    )
    p_final.add_argument("--data", required=True, help="reviewed annotation csv")
    p_final.add_argument(
        "--output", default=None, help="write here instead of editing in place"
    )
    p_final.set_defaults(func=cmd_label_finalize)

    p_merge = sub.add_parser(
        "label-merge", help="union annotation files into one training file"
    )
    p_merge.add_argument(
        "inputs",
        nargs="+",
        metavar="BATCH=PATH",
        help="labelled files to merge, e.g. r1=data/.../to_label_r1.csv",
    )
    p_merge.add_argument(
        "--output",
        default="data/labeled/indomain/labeled_merged.csv",
        help="merged training file",
    )
    p_merge.set_defaults(func=cmd_label_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
