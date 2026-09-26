"""Unified CLI for the data and model pipeline.

    uv run python -m stf.cli prices [--limit N]
    uv run python -m stf.cli news [--limit-urls N] [--refresh]
    uv run python -m stf.cli verify

`verify` re-reads existing data and prints a coverage report (no network).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stf import config
from stf.forecasting.serve import ARM_FAMILIES

def cmd_prices(args: argparse.Namespace) -> int:
    from stf.data import prices

    result = prices.collect(limit=args.limit, end=args.end)
    prices.summary(result)
    return 0 if result and all(value > 0 for value in result.values()) else 1


def cmd_news(args: argparse.Namespace) -> int:
    from stf.data import news

    articles = news.crawl(
        refresh=args.refresh,
        limit_urls=args.limit_urls,
        max_new=args.batch,
        end=args.end,
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
    from stf.forecasting.calendar import to_local
    from stf.sentiment import dataset, experiments, model

    if args.before_date and args.sample_like:
        print(
            "sentiment-refit: --before-date and --sample-like are mutually exclusive.",
            file=sys.stderr,
        )
        return 2
    df = dataset.load_labeled(args.data)
    subset_provenance = None
    if args.before_date:
        if "published_at" not in df.columns:
            print(
                "sentiment-refit: --before-date requires a published_at column.",
                file=sys.stderr,
            )
            return 2
        try:
            cutoff = pd.Timestamp(args.before_date)
        except ValueError:
            print(
                f"sentiment-refit: --before-date {args.before_date!r} is not a date.",
                file=sys.stderr,
            )
            return 2
        if cutoff != cutoff.normalize():
            print(
                "sentiment-refit: --before-date must be a calendar date "
                f"(YYYY-MM-DD); got a time component in {args.before_date!r}.",
                file=sys.stderr,
            )
            return 2
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize(config.TIMEZONE)
        else:
            cutoff = cutoff.tz_convert(config.TIMEZONE)
        try:
            dates = to_local(df["published_at"])
        except (ValueError, TypeError):
            print(
                "sentiment-refit: published_at column is not parseable as timestamps.",
                file=sys.stderr,
            )
            return 2
        undated = int(dates.isna().sum())
        if undated:
            print(
                f"sentiment-refit: {undated} rows lack a parseable published_at; "
                "cannot prove point-in-time.",
                file=sys.stderr,
            )
            return 2
        df = df.loc[dates < cutoff].reset_index(drop=True)
        if df.empty:
            print(
                f"sentiment-refit: no labels strictly before {args.before_date}.",
                file=sys.stderr,
            )
            return 2
        print(
            f"Refit restricted to {len(df)} labels published before {args.before_date}."
        )
    elif args.sample_like:
        # Training-size control for a point-in-time run. Matching only the row count
        # would leave the class prior free to differ, and matching it proportionally
        # would reproduce the FULL set's prior rather than the point-in-time subset's.
        # Either way class balance would be confounded with time selection, so draw
        # the reference checkpoint's exact per-class counts instead.
        try:
            reference = json.loads(Path(args.sample_like).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(
                f"sentiment-refit: cannot read reference manifest {args.sample_like}.",
                file=sys.stderr,
            )
            return 2
        wanted = reference.get("class_distribution")
        reference_cutoff = reference.get("provenance", {}).get("label_cutoff")
        if reference_cutoff is None:
            print(
                f"sentiment-refit: {args.sample_like} is not a point-in-time "
                "manifest (no provenance.label_cutoff); --sample-like must "
                "reference the frozen checkpoint it controls for.",
                file=sys.stderr,
            )
            return 2
        if not wanted:
            print(
                f"sentiment-refit: {args.sample_like} has no class_distribution.",
                file=sys.stderr,
            )
            return 2
        if "sample_id" not in df.columns:
            print(
                "sentiment-refit: --sample-like requires a sample_id column "
                "to identify the drawn rows.",
                file=sys.stderr,
            )
            return 2
    frame = dataset.prepare_model_input(df, args.input_variant)
    if args.sample_like:
        available = frame["label"].value_counts().to_dict()
        short = {k: v for k, v in wanted.items() if v > available.get(k, 0)}
        if short:
            print(
                f"sentiment-refit: not enough labels to match {short} "
                "after deduplication.",
                file=sys.stderr,
            )
            return 2
        drawn = [
            frame.loc[frame["label"] == name].sample(
                n=int(count), random_state=args.sample_seed
            )
            for name, count in sorted(wanted.items())
        ]
        frame = pd.concat(drawn).sort_index().reset_index(drop=True)
        # Which rows were drawn is itself a random factor, so record the identity of
        # the draw. Without it a second seed cannot be told apart from this one, and
        # the control cannot be reproduced or compared across seeds.
        selected = sorted(str(value) for value in frame["sample_id"])
        subset_provenance = {
            "strategy": "class_matched_random",
            "reference_manifest": str(args.sample_like),
            "reference_manifest_sha256": hashlib.sha256(
                Path(args.sample_like).read_bytes()
            ).hexdigest(),
            "sample_seed": args.sample_seed,
            "selected_rows": len(selected),
            "selected_sample_id_sha256": hashlib.sha256(
                "\n".join(selected).encode("utf-8")
            ).hexdigest(),
            "point_in_time": False,
        }
        print(
            f"Refit restricted to {len(frame)} labels matching the class counts of "
            f"{args.sample_like} ({wanted}, seed {args.sample_seed}). This is a "
            "training-size control only; it draws from every date and is NOT "
            "point-in-time."
        )
        print("Subset id:", subset_provenance["selected_sample_id_sha256"][:16])
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
        allow_subset=bool(args.before_date or args.sample_like),
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
        label_cutoff=cutoff.date().isoformat() if args.before_date else None,
        subset=subset_provenance,
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


def _directory_fingerprint(path: Path) -> str | None:
    """Return a deterministic SHA-256 for the complete checkpoint directory."""
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_manifest_path(model_dir: Path) -> Path | None:
    """Find the manifest used by inference, matching the model loader's lookup."""
    for candidate in (model_dir / "manifest.json", model_dir.parent / "manifest.json"):
        if candidate.is_file():
            return candidate
    return None


def _validated_scored_news_manifest(news_path: Path, news: pd.DataFrame) -> dict:
    """Bind a forecast input to the contemporaneous score-news sidecar."""
    from stf.sentiment.dataset import file_fingerprint

    manifest_path = news_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(
            f"missing score-news manifest {manifest_path}; rerun score-news before forecast."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid score-news manifest {manifest_path}.") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid score-news manifest {manifest_path}.")
    if manifest.get("schema_version") != 2:
        raise ValueError(
            f"unsupported score-news manifest schema at {manifest_path}; rerun score-news."
        )

    output = manifest.get("output")
    checkpoint = manifest.get("checkpoint")
    inference = manifest.get("inference")
    score_input = manifest.get("input")
    if not all(
        isinstance(value, dict) for value in (output, checkpoint, inference, score_input)
    ):
        raise ValueError(f"score-news manifest {manifest_path} is missing required provenance.")

    identity_columns = {"ticker", "url", "published_at"}
    probability_columns = {"prob_negative", "prob_neutral", "prob_positive"}
    missing_columns = (identity_columns | probability_columns) - set(news.columns)
    if missing_columns:
        raise ValueError(
            f"score-news parquet lacks required columns: {sorted(missing_columns)}."
        )
    if news.empty:
        raise ValueError("score-news parquet has no rows.")
    try:
        probabilities = news.loc[:, sorted(probability_columns)].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(dtype="float64")
    except (TypeError, ValueError) as error:
        raise ValueError("score-news parquet has non-numeric probabilities.") from error
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.isclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-6).all()
    ):
        raise ValueError("score-news parquet has invalid probability vectors.")

    actual_hash = file_fingerprint(news_path)
    if output.get("sha256") != actual_hash:
        raise ValueError(
            f"score-news parquet hash does not match manifest {manifest_path}."
        )
    if output.get("rows") != int(len(news)):
        raise ValueError(
            f"score-news parquet row count does not match manifest {manifest_path}."
        )
    if not checkpoint.get("directory_sha256"):
        raise ValueError(f"score-news manifest {manifest_path} lacks checkpoint provenance.")
    if (
        not score_input.get("fingerprint")
        or inference.get("truncation_strategy") not in ("head", "tail", "head_tail")
        or not isinstance(inference.get("max_len"), int)
        or inference["max_len"] < 1
        or not isinstance(inference.get("batch_size"), int)
        or inference["batch_size"] < 1
        or not isinstance(inference.get("runtime"), dict)
    ):
        raise ValueError(f"score-news manifest {manifest_path} lacks inference provenance.")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_fingerprint(manifest_path),
        "output_sha256": actual_hash,
        "checkpoint_directory_sha256": checkpoint["directory_sha256"],
        "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256"),
        "checkpoint_manifest_path": checkpoint.get("manifest_path"),
        "inference": inference,
        "input": score_input,
    }


def cmd_score_news(args: argparse.Namespace) -> int:
    """Score crawler rows and persist audited probability provenance."""
    from stf.data.news import load_ticker_articles
    from stf.sentiment.dataset import (
        build_input_text,
        file_fingerprint,
        frame_fingerprint,
    )
    from stf.sentiment.model import (
        predict_proba,
        reproducibility_metadata,
        resolve_inference_config,
        resolve_input_variant,
    )

    model_dir = Path(args.model_dir)
    checkpoint_variant = resolve_input_variant(model_dir)
    if args.input_variant != checkpoint_variant:
        raise ValueError(
            f"score-news input variant {args.input_variant!r} does not match "
            f"checkpoint input variant {checkpoint_variant!r}."
        )

    articles = load_ticker_articles()
    if args.limit is not None:
        articles = articles.head(args.limit)
    if articles.empty:
        print("score-news: no ticker/article rows to score.", file=sys.stderr)
        return 1
    scored = build_input_text(
        articles, args.input_variant, context_chars=args.context_chars
    )
    scored_all = scored

    output_path = Path(args.output)
    existing = None
    if args.incremental and output_path.exists():
        existing = pd.read_parquet(output_path)
        scored_keys = set(zip(existing["ticker"], existing["url"]))
        fresh = ~pd.Series(
            list(zip(scored["ticker"], scored["url"])), index=scored.index
        ).isin(scored_keys)
        scored = scored.loc[fresh].reset_index(drop=True)
        print(
            f"[score-news] incremental: {int(fresh.sum())} new rows "
            f"({len(existing)} already scored)"
        )

    probability_columns = ["prob_negative", "prob_neutral", "prob_positive"]
    effective_strategy, effective_max_len = resolve_inference_config(
        model_dir, truncation_strategy=args.truncation_strategy
    )
    if len(scored):
        probs = predict_proba(
            scored["text"].tolist(),
            model_dir,
            batch_size=args.batch_size,
            truncation_strategy=effective_strategy,
            max_len=effective_max_len,
        )
        probs = np.asarray(probs, dtype="float64")
        if probs.shape != (len(scored), len(probability_columns)):
            raise RuntimeError(
                "score-news: checkpoint returned probabilities with an unexpected shape."
            )
        if (
            not np.isfinite(probs).all()
            or (probs < 0).any()
            or (probs > 1).any()
            or not np.isclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-6).all()
        ):
            raise RuntimeError(
                "score-news: checkpoint returned invalid probability vectors."
            )
        new_rows = pd.DataFrame(
            {
                "ticker": scored["ticker"].to_numpy(),
                "url": scored["url"].to_numpy(),
                "published_at": scored["published_at"].to_numpy(),
                "prob_negative": probs[:, 0],
                "prob_neutral": probs[:, 1],
                "prob_positive": probs[:, 2],
            }
        )
    else:
        new_rows = pd.DataFrame(
            columns=["ticker", "url", "published_at", *probability_columns]
        )

    out = (
        pd.concat([existing, new_rows], ignore_index=True)
        if existing is not None
        else new_rows
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    manifest_path = output_path.with_suffix(".manifest.json")
    refit_manifest_path = _checkpoint_manifest_path(model_dir)
    manifest = {
        "schema_version": 2,
        "output": {
            "path": str(output_path),
            "sha256": file_fingerprint(output_path),
            "rows": int(len(out)),
        },
        "probabilities": {
            "columns": probability_columns,
            "totals": {
                column: float(out[column].sum()) for column in probability_columns
            },
            "validation": {
                "finite": True,
                "nonnegative": True,
                "rows_summing_to_one": int(len(out)),
                "atol": 1e-6,
            },
        },
        "checkpoint": {
            "directory": str(model_dir),
            "directory_sha256": _directory_fingerprint(model_dir),
            "manifest_path": str(refit_manifest_path)
            if refit_manifest_path is not None
            else None,
            "manifest_sha256": file_fingerprint(refit_manifest_path)
            if refit_manifest_path is not None
            else None,
        },
        "inference": {
            "truncation_strategy": effective_strategy,
            "max_len": effective_max_len,
            "batch_size": args.batch_size,
            "runtime": reproducibility_metadata(),
        },
        "input": {
            "fingerprint": frame_fingerprint(
                scored_all, columns=("ticker", "url", "published_at", "text")
            ),
            "variant": args.input_variant,
            "context_chars": args.context_chars,
            "limit": args.limit,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[score-news] scored {len(out)} rows -> {output_path} "
        f"(manifest {manifest_path})"
    )
    return 0


def cmd_webapp(args: argparse.Namespace) -> int:
    """Serve the read-only dashboard over forecast artifacts."""
    import uvicorn

    uvicorn.run(
        "stf.webapp.app:app", host=args.host, port=args.port, log_level="info"
    )
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
        TemporalFusionClassifier,
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
    X_both = np.concatenate([X_price, X_sent], axis=2)
    net3 = TemporalFusionClassifier(
        X_both.shape[-1], hidden=8, num_heads=2, num_classes=len(TREND_LABELS)
    )
    history3 = fit_lstm(
        net3, X_both[train_mask2], y2[train_mask2], epochs=args.epochs, seed=args.seed
    )
    print(
        f"Forecast smoke: panel={len(labeled)} rows | "
        f"train={len(split.train)} val={len(split.val)} test={len(split.test)} | "
        f"sequences=train:{train_mask.sum()} val:{val_mask.sum()} test:{test_mask.sum()} | "
        f"price_lstm_final_loss={history['loss_history'][-1]:.6f} | "
        f"price_sentiment_lstm_final_loss={history2['loss_history'][-1]:.6f} | "
        f"tft_final_loss={history3['loss_history'][-1]:.6f}"
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
    from stf.forecasting.experiment import (
        ForecastConfig,
        first_test_observation_date,
        frame_hash,
        run_experiment,
    )

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
        try:
            score_manifest = _validated_scored_news_manifest(news_path, news)
        except ValueError as error:
            print(f"forecast: {error}", file=sys.stderr)
            return 2
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
            "score_manifest": score_manifest,
        }

    panel = assemble(prices, news)
    # Both bounds filter target_date, because the label is the outcome and the
    # declared study window is a window over outcomes. Price history deliberately
    # reaches back before the window so rolling features are defined on the first
    # evaluated session; bounding the low side on observation_date instead would
    # discard the first in-window outcome of every ticker and silently shift every
    # walk-forward split.
    panel_start = args.panel_start or config.DATE_START
    panel_end = args.panel_end or config.DATE_END
    target = pd.to_datetime(panel["target_date"])
    panel = panel[target >= pd.Timestamp(panel_start)].reset_index(drop=True)
    panel = panel[
        pd.to_datetime(panel["target_date"]) <= pd.Timestamp(panel_end)
    ].reset_index(drop=True)
    provenance["panel_start"] = panel_start
    provenance["panel_end"] = panel_end
    if "alignment_report" in panel.attrs:
        # This report covers every scored link handed to assemble, including any
        # published outside the study window, so it cannot describe the evaluated
        # panel on its own.
        provenance["alignment_report"] = panel.attrs["alignment_report"]
    if news is not None and not news.empty:
        from stf.forecasting.calendar import (
            align_news_to_sessions,
            alignment_report,
            to_local,
        )

        window_end = panel_end
        published = to_local(news["published_at"])
        # Half-open on the final day: an article published on window_end after the
        # 15:00 cutoff is still published inside the study window even though it
        # anchors to the next session. A 23:59:59 upper bound would drop the last
        # second when a timestamp carries fractional seconds.
        lower = pd.Timestamp(panel_start, tz=config.TIMEZONE)
        upper = pd.Timestamp(window_end, tz=config.TIMEZONE) + pd.Timedelta(days=1)
        in_window = news[
            ((published >= lower) & (published < upper)).to_numpy()
        ].reset_index(drop=True)
        aligned = align_news_to_sessions(in_window, prices)
        observation = pd.to_datetime(aligned["observation_date"], errors="coerce")
        anchored_outside = int((observation > pd.Timestamp(window_end)).sum())
        provenance["alignment_report_in_window"] = {
            "window": [panel_start, window_end],
            "articles": int(in_window["url"].nunique()),
            "links": int(len(in_window)),
            **alignment_report(aligned),
            # Published inside the window but rolled forward onto a session past
            # its end, so the panel cannot carry them.
            "anchored_after_window": anchored_outside,
        }
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
        target_mode=args.target,
        horizon=args.horizon,
    )
    if provenance["news_sentiment"] is not None:
        # Point-in-time verdict: the scoring checkpoint must have been trained
        # only on labels published before the first test observation date. The
        # first test observation comes from the same walk-forward call the
        # experiment uses, so the verdict cannot drift from the scored split.
        first_test_obs = None
        try:
            first_test_obs = first_test_observation_date(panel, cfg)
        except (ValueError, KeyError):
            first_test_obs = None
        checkpoint = provenance["news_sentiment"]["score_manifest"]
        manifest_path = checkpoint.get("checkpoint_manifest_path")
        label_cutoff = None
        config_selection_size = None
        reason = "manifest_missing"
        if manifest_path and Path(manifest_path).is_file():
            expected = checkpoint.get("checkpoint_manifest_sha256")
            try:
                raw = Path(manifest_path).read_bytes()
            except OSError:
                raw = None
                reason = "manifest_unreadable"
            if raw is None:
                pass
            elif not expected:
                reason = "hash_missing"
            elif hashlib.sha256(raw).hexdigest() != expected:
                reason = "hash_mismatch"
            else:
                try:
                    manifest_doc = json.loads(raw)
                    label_cutoff = manifest_doc.get("provenance", {}).get("label_cutoff")
                    config_selection_size = (
                        manifest_doc.get("selection", {})
                        .get("evaluation_reference", {})
                        .get("data_size")
                    )
                    reason = "no_label_cutoff" if label_cutoff is None else "verified"
                except (json.JSONDecodeError, AttributeError):
                    reason = "manifest_unreadable"
        point_in_time = bool(
            label_cutoff
            and first_test_obs
            and pd.Timestamp(label_cutoff).date() <= pd.Timestamp(first_test_obs).date()
        )
        if reason == "verified" and not point_in_time:
            reason = "no_test_window" if not first_test_obs else "cutoff_after_test_start"
        provenance["news_sentiment"]["label_cutoff"] = label_cutoff
        provenance["news_sentiment"]["first_test_observation_date"] = first_test_obs
        provenance["news_sentiment"]["point_in_time"] = point_in_time
        provenance["news_sentiment"]["point_in_time_reason"] = reason
        # The verdict certifies the *training labels* only: the checkpoint's
        # hyperparameters were selected on a CV whose data_size may exceed the
        # frozen subset, so config selection is not claimed point-in-time.
        provenance["news_sentiment"]["point_in_time_scope"] = "training_labels"
        provenance["news_sentiment"]["config_selection_data_size"] = config_selection_size
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

def cmd_forecast_refit(args: argparse.Namespace) -> int:
    """Fit one ladder arm on the full panel and persist it for daily inference."""
    from stf.forecasting import assemble
    from stf.forecasting.experiment import ForecastConfig, frame_hash
    from stf.forecasting.serve import ARM_FAMILIES, refit_arm

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
    if args.news_sentiment is not None:
        news_path = Path(args.news_sentiment)
        if not news_path.exists():
            print(f"forecast-refit: missing {news_path}", file=sys.stderr)
            return 2
        news = pd.read_parquet(news_path)
        try:
            score_manifest = _validated_scored_news_manifest(news_path, news)
        except ValueError as error:
            print(f"forecast-refit: {error}", file=sys.stderr)
            return 2
        provenance["news_sentiment"] = {
            "mode": "real",
            "source_path": str(news_path),
            "source_hash": frame_hash(news),
            "rows": int(len(news)),
            "score_manifest": score_manifest,
        }

    family, use_sentiment = ARM_FAMILIES[args.arm]
    if use_sentiment and news is None:
        print(
            f"forecast-refit: arm {args.arm!r} needs --news-sentiment.",
            file=sys.stderr,
        )
        return 2

    panel = assemble(prices, news)
    if "alignment_report" in panel.attrs:
        provenance["alignment_report"] = panel.attrs["alignment_report"]
    # Serving fits on every available row, which may extend past the configured
    # study window; record the actual fitted range, not the config constant.
    target_dates = pd.to_datetime(panel["target_date"]).dropna()
    provenance["panel_start"] = str(target_dates.min().date())
    provenance["panel_end"] = str(target_dates.max().date())
    provenance["panel_hash"] = frame_hash(panel)

    cfg = ForecastConfig(
        seq_len=args.seq_len,
        val_size=args.val_size,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        seeds=tuple(args.seeds),
    )
    manifest = refit_arm(
        panel, args.arm, cfg=cfg, output_dir=Path(args.model_dir), provenance=provenance
    )
    print(
        f"[forecast-refit] arm={args.arm} family={family} "
        f"train={manifest['train_rows']} val={manifest['val_rows']} "
        f"seeds={manifest['seeds']} -> {args.model_dir}"
    )
    return 0


def cmd_forecast_predict(args: argparse.Namespace) -> int:
    """Score the latest session per ticker with a refit arm."""
    from stf.forecasting import assemble
    from stf.forecasting.serve import load_arm, predict_latest

    arm = load_arm(Path(args.model_dir))
    prices = _load_prices()

    news = None
    if arm.use_sentiment:
        if args.news_sentiment is None:
            print(
                f"forecast-predict: arm {arm.name!r} needs --news-sentiment; "
                "run score-news first.",
                file=sys.stderr,
            )
            return 2
        news_path = Path(args.news_sentiment)
        if not news_path.exists():
            print(f"forecast-predict: missing {news_path}", file=sys.stderr)
            return 2
        news = pd.read_parquet(news_path)
        try:
            _validated_scored_news_manifest(news_path, news)
        except ValueError as error:
            print(f"forecast-predict: {error}", file=sys.stderr)
            return 2

    panel = assemble(prices, news)
    predictions = predict_latest(panel, arm)

    # Stamp issuance evidence: a resolved row only counts as a live forecast when
    # it was provably written before the target session traded.
    from stf.sentiment.dataset import file_fingerprint

    predictions["issued_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest_path = Path(args.model_dir) / "manifest.json"
    predictions["checkpoint_manifest_sha256"] = (
        file_fingerprint(manifest_path) if manifest_path.is_file() else None
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_date = pd.to_datetime(predictions["observation_date"]).max().date().isoformat()
    dated_path = out_dir / f"predictions_{obs_date}.parquet"
    if dated_path.exists():
        # Issued predictions are an audit trail: re-running for the same session
        # must reproduce them bit-for-bit (ignoring the issuance stamp) or stop,
        # never silently rewrite history.
        existing = pd.read_parquet(dated_path)
        shared = [c for c in predictions.columns if c in existing.columns]
        comparable = [c for c in shared if c not in ("issued_at",)]
        same = len(existing) == len(predictions) and existing[
            comparable
        ].reset_index(drop=True).equals(
            predictions[comparable].reset_index(drop=True)
        )
        if not same:
            print(
                f"forecast-predict: {dated_path} already holds different "
                "predictions; refusing to overwrite issued output. Delete it "
                "explicitly if the earlier issuance was invalid.",
                file=sys.stderr,
            )
            return 3
    else:
        predictions.to_parquet(dated_path, index=False)
    predictions.to_parquet(out_dir / "latest.parquet", index=False)

    print(f"[forecast-predict] arm={arm.name} date={obs_date} tickers={len(predictions)}")
    for _, row in predictions.iterrows():
        print(
            f"  {row['ticker']:<6} {row['y_pred']:<5} "
            f"UP={row['prob_up']:.3f} FLAT={row['prob_flat']:.3f} "
            f"DOWN={row['prob_down']:.3f} news={'yes' if row['has_news'] else 'no'}"
        )
    print(f"[forecast-predict] -> {dated_path}")
    return 0


def cmd_forecast_resolve(args: argparse.Namespace) -> int:
    """Join stored daily predictions with realized labels for monitoring."""
    from stf.forecasting import assemble
    from stf.forecasting.serve import load_arm, resolve_predictions

    arm = load_arm(Path(args.model_dir))
    live_dir = Path(args.predictions_dir)
    files = sorted(live_dir.glob("predictions_*.parquet"))
    if not files:
        print(f"forecast-resolve: no predictions under {live_dir}", file=sys.stderr)
        return 1
    predictions = pd.concat(
        [pd.read_parquet(path) for path in files], ignore_index=True
    )

    news = None
    if arm.use_sentiment and args.news_sentiment is not None:
        news_path = Path(args.news_sentiment)
        if not news_path.exists():
            print(f"forecast-resolve: missing {news_path}", file=sys.stderr)
            return 2
        news = pd.read_parquet(news_path)
    panel = assemble(_load_prices(), news)

    resolved = resolve_predictions(predictions, panel, arm.thresholds)
    out_path = live_dir / "resolved.parquet"
    resolved.to_parquet(out_path, index=False)

    scored = resolved.dropna(subset=["y_true"])
    pending = int(resolved["y_true"].isna().sum())
    prospective = scored[scored["prospective"].astype(bool)]
    print(
        f"[forecast-resolve] rows={len(resolved)} resolved={len(scored)} "
        f"(prospective={len(prospective)} replayed={len(scored) - len(prospective)}) "
        f"pending={pending} -> {out_path}"
    )
    if len(prospective):
        acc = float(prospective["correct"].astype(bool).mean())
        print(
            f"[forecast-resolve] live accuracy={acc:.4f} over "
            f"{len(prospective)} prospective rows"
        )
    elif len(scored):
        print(
            "[forecast-resolve] no prospective rows yet; resolved rows are "
            "replays and do not count toward the live track record"
        )
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
    """Measure polarity information against a paired neutral-news control."""
    from stf.forecasting.experiment import compare_information_gain

    report = compare_information_gain(
        Path(args.real), Path(args.control), arm=args.arm, price_arm=args.price_arm
    )
    # Traceability: the report must name its inputs and their point-in-time
    # verdicts, or it cannot be tied back to a specific pair of runs.
    for label, run_dir in (("real", args.real), ("control", args.control)):
        try:
            prov = json.loads(
                (Path(run_dir) / "forecast_results.json").read_text(encoding="utf-8")
            ).get("provenance", {})
            news_prov = prov.get("news_sentiment") or {}
            report[f"{label}_run"] = {
                "dir": str(run_dir),
                "point_in_time": news_prov.get("point_in_time"),
                "point_in_time_reason": news_prov.get("point_in_time_reason"),
                "panel_start": prov.get("panel_start"),
                "panel_end": prov.get("panel_end"),
            }
        except (OSError, json.JSONDecodeError):
            report[f"{label}_run"] = {"dir": str(run_dir)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"[forecast-compare] {args.arm} vs {args.price_arm}")
    for metric, effect in report["effects"].items():
        win = effect["information_gain_window_bootstrap"]
        date = effect["information_gain_date_bootstrap"]
        print(f"  {metric}:")
        print(f"      price-only                 {effect['price_only']:.4f}")
        print(f"      two-branch neutral prior   {effect['two_branch_neutral_prior']:.4f}")
        print(f"      two-branch real sentiment  {effect['two_branch_real_sentiment']:.4f}")
        print(
            "      architecture + news presence/volume "
            f"{effect['architecture_and_news_presence_volume_effect']:+.4f}"
        )
        print(
            f"      information gain           {effect['information_gain']:+.4f}  "
            f"window-CI [{win['low']:+.4f}, {win['high']:+.4f}]  "
            f"date-CI [{date['low']:+.4f}, {date['high']:+.4f}]"
        )
        print(f"      naive delta (conflated)    {effect['naive_delta']:+.4f}")
    print(f"[forecast-compare] -> {output}")
    return 0


def cmd_forecast_backtest(args: argparse.Namespace) -> int:
    """Score a stored arm as an idealized close-to-close long/short diagnostic."""
    from stf.forecasting.backtest import backtest_arm

    run_dir = Path(args.run)
    predictions = pd.read_csv(run_dir / "forecast_predictions.csv")
    report = backtest_arm(
        predictions,
        _load_prices(),
        arm=args.arm,
        cost_bps=args.cost_bps,
        permutations=args.permutations,
    )
    # A backtest that cannot be tied to the run it scored is an anecdote.
    results_path = run_dir / "forecast_results.json"
    if results_path.exists():
        record = json.loads(results_path.read_text(encoding="utf-8"))
        news_prov = (record.get("provenance") or {}).get("news_sentiment") or {}
        report["run"] = {
            "dir": str(run_dir),
            "target_mode": (record.get("config") or {}).get("target_mode"),
            "point_in_time": news_prov.get("point_in_time"),
            "prices_hash": (record.get("provenance") or {}).get("prices_hash"),
        }

    output = Path(args.output) if args.output else run_dir / f"backtest_{args.arm}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    net, gross = report["net"], report["gross"]
    rebalanced, hold = report["equal_weight_rebalanced"], report["buy_and_hold"]
    boot = report["net_bootstrap"]["mean_daily_return"]
    print(f"[forecast-backtest] {args.arm} @ {args.cost_bps:.0f} bps, {net['sessions']} sessions")
    print(f"  NOT TRADABLE: {report['execution']}")
    print(
        f"  gross  ann.return {gross['annualized_return']:+.2%}  "
        f"Sharpe {gross['annualized_sharpe']:+.2f}"
    )
    print(
        f"  net    ann.return {net['annualized_return']:+.2%}  "
        f"Sharpe {net['annualized_sharpe']:+.2f}  "
        f"hit {net['hit_rate']:.3f}  maxDD {net['max_drawdown']:.2%}"
    )
    print(
        f"         mean daily {net['mean_daily_return']:+.5f} "
        f"CI [{boot['low']:+.5f}, {boot['high']:+.5f}] "
        f"draws<=0 {boot['fraction_of_draws_le_zero']:.3f}"
    )
    print(
        f"  exposure  gross {report['mean_gross_exposure']:.2f}  "
        f"|net| {report['mean_abs_net_exposure']:.3f}  "
        f"one-sided sessions {report['one_sided_fraction']:.1%}  "
        f"turnover {report['mean_turnover']:.2f}"
    )
    print(
        f"  bench  equal-weight rebalanced {rebalanced['annualized_return']:+.2%} "
        f"(Sharpe {rebalanced['annualized_sharpe']:+.2f})  "
        f"buy-and-hold {hold['annualized_return']:+.2%} "
        f"(Sharpe {hold['annualized_sharpe']:+.2f})"
    )
    reg = report["gross_vs_market"]
    print(
        f"  alpha  gross vs basket {reg['alpha_daily']:+.5f}/day "
        f"(t {reg['alpha_t_statistic']:+.2f}, beta {reg['beta']:+.3f})  "
        f"break-even cost {report['breakeven_cost_bps']:.1f} bps"
    )
    null = report.get("permutation_null_gross_sharpe")
    if null:
        print(
            f"  null   shuffled labels, {null['draws']} draws: mean Sharpe "
            f"{null['mean_sharpe']:+.2f} sd {null['sd_sharpe']:.2f} "
            f"95% [{null['low']:+.2f}, {null['high']:+.2f}]"
        )
    print(f"[forecast-backtest] -> {output}")
    return 0


def cmd_forecast_ic(args: argparse.Namespace) -> int:
    """Measure what the sentiment score knows before any model is fitted."""
    from stf.forecasting import assemble
    from stf.forecasting.signal_ic import (
        build_return_targets,
        cross_sectional_ic,
        information_coefficients,
    )

    news = pd.read_parquet(args.news_sentiment)
    prices = _load_prices()
    panel = assemble(prices, news)
    panel = panel[panel["observation_date"] <= pd.Timestamp(args.panel_end)]
    targets = build_return_targets(
        panel[["ticker", "observation_date", "close"]].drop_duplicates()
    )

    report = information_coefficients(panel, targets, samples=args.bootstrap_samples)
    report["cross_sectional"] = cross_sectional_ic(panel, targets)
    report["news_sentiment"] = str(args.news_sentiment)
    report["panel_end"] = str(args.panel_end)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"[forecast-ic] {report['sessions']} sessions, news rows only")
    print(f"  {'signal':24s} {'target':28s} {'n':>6s} {'IC':>8s} {'95% CI':>20s} {'p(<=0)':>7s}")
    for row in report["rows"]:
        print(
            f"  {row['signal']:24s} {row['target']:28s} {row['n']:6d} {row['ic']:+8.4f} "
            f"[{row['low']:+.4f},{row['high']:+.4f}] {row['one_sided_p_le_zero']:7.3f}"
        )
    cs = report["cross_sectional"]
    print(
        f"  cross-sectional daily IC vs {cs['target']}: {cs['mean_ic']:+.4f} "
        f"(se {cs['standard_error']:.4f}, t {cs['t_statistic']:+.2f}, {cs['sessions']} sessions)"
    )
    print(f"[forecast-ic] -> {output}")
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
    p_prices.add_argument(
        "--end",
        default=config.DATE_END,
        help="last session date YYYY-MM-DD (default: study window end)",
    )
    p_prices.set_defaults(func=cmd_prices)

    p_news = sub.add_parser("news", help="crawl news (timestamp + title + body)")
    p_news.add_argument(
        "--limit-urls", type=int, default=None, help="only take the first N articles"
    )
    p_news.add_argument(
        "--end",
        default=config.DATE_END,
        help="crawl listings up to YYYY-MM-DD (default: study window end)",
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
    p_refit.add_argument(
        "--before-date",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "train only on labels published strictly before this date "
            "(point-in-time checkpoint); recorded in the manifest"
        ),
    )
    p_refit.add_argument(
        "--sample-like",
        default=None,
        metavar="MANIFEST",
        help=(
            "draw the exact per-class label counts recorded in another refit "
            "manifest, ignoring publication date; size- and class-matched control "
            "for a --before-date run (mutually exclusive with it)"
        ),
    )
    p_refit.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="seed for --sample-like (default: 42)",
    )
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
        help="verified score-news parquet with observed sentiment probabilities",
    )
    news_input.add_argument(
        "--neutral-news-sentiment",
        default=None,
        help=(
            "verified score-news parquet; preserve article timing and volume, "
            "force neutral probabilities"
        ),
    )
    p_forecast.add_argument("--output", default="outputs/forecast", help="artifact directory")
    p_forecast.add_argument(
        "--panel-end",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "drop panel rows whose target date is after this date "
            f"(default: the study window end, {config.DATE_END})"
        ),
    )
    p_forecast.add_argument(
        "--panel-start",
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "drop panel rows whose target date is before this date "
            f"(default: the study window start, {config.DATE_START})"
        ),
    )
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
    p_forecast.add_argument(
        "--target",
        choices=("raw", "excess"),
        default="raw",
        help=(
            "'raw' labels the next session's own return; 'excess' labels it net of the "
            "equal-weighted cross-section of the same session, so the classes measure "
            "relative performance instead of the shared market move"
        ),
    )
    p_forecast.add_argument(
        "--horizon",
        type=int,
        default=1,
        help=(
            "sessions ahead the label looks; >1 overlaps consecutive labels, so the "
            "effective sample shrinks by roughly this factor and the intervals narrow "
            "more than the evidence warrants"
        ),
    )
    p_forecast.set_defaults(func=cmd_forecast)

    p_refit_fc = sub.add_parser(
        "forecast-refit",
        help="fit one arm on the full panel and persist it for daily inference",
    )
    p_refit_fc.add_argument("--arm", required=True, choices=sorted(ARM_FAMILIES))
    p_refit_fc.add_argument(
        "--news-sentiment",
        default=None,
        help="verified score-news parquet; required for *_sentiment arms",
    )
    p_refit_fc.add_argument(
        "--model-dir", required=True, help="output directory for the frozen arm"
    )
    p_refit_fc.add_argument("--seq-len", type=int, default=5)
    p_refit_fc.add_argument("--val-size", type=int, default=60)
    p_refit_fc.add_argument("--hidden", type=int, default=32)
    p_refit_fc.add_argument("--epochs", type=int, default=30)
    p_refit_fc.add_argument("--batch-size", type=int, default=128)
    p_refit_fc.add_argument("--patience", type=int, default=5)
    p_refit_fc.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44]
    )
    p_refit_fc.set_defaults(func=cmd_forecast_refit)

    p_predict = sub.add_parser(
        "forecast-predict",
        help="score the latest session per ticker with a refit arm",
    )
    p_predict.add_argument("--model-dir", required=True)
    p_predict.add_argument(
        "--news-sentiment",
        default=None,
        help="verified score-news parquet; required for *_sentiment arms",
    )
    p_predict.add_argument(
        "--output-dir",
        default="outputs/live",
        help="directory for dated + latest prediction parquets",
    )
    p_predict.set_defaults(func=cmd_forecast_predict)

    p_resolve = sub.add_parser(
        "forecast-resolve",
        help="join stored daily predictions with realized labels",
    )
    p_resolve.add_argument("--model-dir", required=True)
    p_resolve.add_argument(
        "--predictions-dir", default="outputs/live"
    )
    p_resolve.add_argument(
        "--news-sentiment",
        default=None,
        help="score-news parquet matching the arm's sentiment features",
    )
    p_resolve.set_defaults(func=cmd_forecast_resolve)


    p_cmp = sub.add_parser(
        "forecast-compare",
        help="measure paired polarity information against a neutral-news control",
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

    p_bt = sub.add_parser(
        "forecast-backtest",
        help="price one arm's stored predictions as a cash-neutral long/short book",
    )
    p_bt.add_argument("--run", required=True, help="run directory holding forecast_predictions.csv")
    p_bt.add_argument("--arm", default="lstm_price_sentiment")
    p_bt.add_argument(
        "--cost-bps",
        type=float,
        default=20.0,
        help="cost charged per unit of turnover; 20 bps is a HOSE-realistic floor",
    )
    p_bt.add_argument("--output", default=None, help="defaults to <run>/backtest_<arm>.json")
    p_bt.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="label shuffles inside each session for the null Sharpe; 0 disables",
    )
    p_bt.set_defaults(func=cmd_forecast_backtest)

    p_ic = sub.add_parser(
        "forecast-ic",
        help="rank correlation of the sentiment score with realized returns, model-free",
    )
    p_ic.add_argument("--news-sentiment", required=True, help="parquet written by score-news")
    p_ic.add_argument("--panel-end", default="2025-12-31", help="last observation date to include")
    p_ic.add_argument("--bootstrap-samples", type=int, default=2000)
    p_ic.add_argument("--output", default="outputs/signal_ic.json")
    p_ic.set_defaults(func=cmd_forecast_ic)

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
    p_score.add_argument(
        "--incremental",
        action="store_true",
        help="append only articles not already scored in --output",
    )
    p_score.set_defaults(func=cmd_score_news)

    p_web = sub.add_parser(
        "webapp",
        help="serve the read-only forecast dashboard over outputs/ artifacts",
    )
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.set_defaults(func=cmd_webapp)

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
