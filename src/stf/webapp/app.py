"""FastAPI surface exposing forecast experiment artifacts to the dashboard.

The API is read-only: every endpoint derives from files under ``outputs/`` and
never mutates them. Run directories are discovered by the presence of
``forecast_results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from stf import config

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = REPO_ROOT / "web" / "dist"

app = FastAPI(title="stock-trend-forecasting", docs_url="/api/docs")


def _outputs_root() -> Path:
    return Path(config.ROOT) / "outputs"

def _run_dirs() -> dict[str, Path]:
    root = _outputs_root()
    if not root.is_dir():
        return {}
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "forecast_results.json").is_file()
    }


def _run_dir(name: str) -> Path:
    runs = _run_dirs()
    if name not in runs:
        raise HTTPException(status_code=404, detail=f"unknown run {name!r}")
    return runs[name]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500, detail=f"cannot read artifact {path.name}"
        ) from error


def _read_csv_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"missing artifact {path.name}")
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


def _normalize_news_sentiment(news: dict) -> dict:
    """Map pre-rename artifact keys onto the canonical schema.

    Runs produced before the provenance rename store ``path``/``hash`` and no
    ``mode``; a present path implies real scored news.
    """
    normalized = dict(news)
    if "source_path" not in normalized and "path" in normalized:
        normalized["source_path"] = normalized.pop("path")
    else:
        normalized.pop("path", None)
    if "source_hash" not in normalized and "hash" in normalized:
        normalized["source_hash"] = normalized.pop("hash")
    else:
        normalized.pop("hash", None)
    if normalized.get("mode") is None and normalized.get("source_path"):
        normalized["mode"] = "real"
    return normalized


def _normalize_information_gain(payload: dict) -> dict:
    """Rename the pre-rename effect key to the canonical one."""
    effects = payload.get("effects")
    if isinstance(effects, dict):
        for effect in effects.values():
            if isinstance(effect, dict) and "architecture_effect" in effect:
                effect.setdefault(
                    "architecture_and_news_presence_volume_effect",
                    effect.pop("architecture_effect"),
                )
    return payload



@app.get("/api/runs")
def list_runs() -> dict:
    """List forecast run directories with headline provenance."""
    runs = []
    for name, path in _run_dirs().items():
        results = _read_json(path / "forecast_results.json")
        provenance = results.get("provenance") or {}
        news = _normalize_news_sentiment(provenance.get("news_sentiment") or {})
        runs.append(
            {
                "name": name,
                "panel_rows": results.get("panel_rows"),
                "mode": news.get("mode"),
                "date_start": provenance.get("date_start"),
                "date_end": provenance.get("date_end"),
                "has_information_gain": (path / "information_gain.json").is_file(),
            }
        )
    return {"runs": runs}


@app.get("/api/runs/{name}/summary")
def run_summary(name: str) -> dict:
    """Config, per-arm summary metrics, ablation and stratified aggregates."""
    results = _read_json(_run_dir(name) / "forecast_results.json")
    provenance = results.get("provenance") or {}
    if isinstance(provenance.get("news_sentiment"), dict):
        provenance = dict(provenance)
        provenance["news_sentiment"] = _normalize_news_sentiment(
            provenance["news_sentiment"]
        )
    return {
        "name": name,
        "config": results.get("config"),
        "chance_level": results.get("chance_level"),
        "panel_rows": results.get("panel_rows"),
        "panel_tickers": results.get("panel_tickers"),
        "summary": results.get("summary"),
        "ablation": results.get("ablation"),
        "stratified_by_news": results.get("stratified_by_news"),
        "environment": results.get("environment"),
        "provenance": provenance or None,
    }


@app.get("/api/runs/{name}/metrics")
def run_metrics(name: str) -> dict:
    """Per-arm metric table with bootstrap intervals."""
    return {"rows": _read_csv_records(_run_dir(name) / "forecast_metrics.csv")}


@app.get("/api/runs/{name}/predictions")
def run_predictions(
    name: str,
    arm: str | None = Query(default=None),
    ticker: str | None = Query(default=None),
    window: int | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Paginated per-row predictions, filterable by arm/ticker/window."""
    rows = _read_csv_records(_run_dir(name) / "forecast_predictions.csv")
    if arm is not None:
        rows = [row for row in rows if row.get("arm") == arm]
    if ticker is not None:
        rows = [row for row in rows if row.get("ticker") == ticker]
    if window is not None:
        rows = [row for row in rows if row.get("window") == window]
    return {"total": len(rows), "rows": rows[offset : offset + limit]}


@app.get("/api/runs/{name}/stratified")
def run_stratified(name: str) -> dict:
    """Per-window metrics split by news presence."""
    return {"rows": _read_csv_records(_run_dir(name) / "forecast_stratified.csv")}


@app.get("/api/runs/{name}/information-gain")
def run_information_gain(name: str) -> dict:
    """Paired decomposition of the sentiment contribution for one run."""
    path = _run_dir(name) / "information_gain.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="run has no information_gain.json")
    return _normalize_information_gain(_read_json(path))


def _live_dir() -> Path:
    return _outputs_root() / "live"


def _live_arm_dirs() -> dict[str, Path]:
    root = _live_dir()
    if not root.is_dir():
        return {}
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "latest.parquet").is_file()
    }


@app.get("/api/live/latest")
def live_latest() -> dict:
    """Latest per-ticker predictions produced by ``forecast-predict``, per arm."""
    arms = []
    for name, path in _live_arm_dirs().items():
        frame = pd.read_parquet(path / "latest.parquet")
        rows = json.loads(frame.to_json(orient="records", date_format="iso"))
        obs = pd.to_datetime(frame["observation_date"]).max()
        arms.append(
            {
                "arm": name,
                "observation_date": obs.date().isoformat(),
                "rows": rows,
            }
        )
    if not arms:
        raise HTTPException(status_code=404, detail="no live predictions yet")
    return {"arms": arms}


@app.get("/api/live/history")
def live_history() -> dict:
    """Resolved predictions with realized labels and per-date accuracy, per arm."""
    arms = []
    for name, path in _live_arm_dirs().items():
        resolved_path = path / "resolved.parquet"
        if not resolved_path.is_file():
            continue
        frame = pd.read_parquet(resolved_path)
        scored = frame.dropna(subset=["y_true"])
        # Only rows issued before their target session count as a live forecast;
        # replayed or pre-stamping rows are reported separately so the headline
        # number cannot silently mix the two.
        prospective = (
            scored[scored["prospective"].astype(bool)]
            if "prospective" in scored.columns
            else scored.iloc[0:0]
        )
        by_date = []
        if len(prospective):
            grouped = prospective.groupby(
                pd.to_datetime(prospective["target_date"]).dt.date, sort=True
            )
            for date, block in grouped:
                by_date.append(
                    {
                        "target_date": date.isoformat(),
                        "n": int(len(block)),
                        "accuracy": float(block["correct"].astype(bool).mean()),
                    }
                )
        arms.append(
            {
                "arm": name,
                "total": int(len(frame)),
                "resolved": int(len(scored)),
                "prospective_resolved": int(len(prospective)),
                "replayed_resolved": int(len(scored) - len(prospective)),
                "pending": int(frame["y_true"].isna().sum()),
                "accuracy": float(prospective["correct"].astype(bool).mean())
                if len(prospective)
                else None,
                "by_date": by_date,
            }
        )
    if not arms:
        raise HTTPException(status_code=404, detail="no resolved predictions yet")
    return {"arms": arms}


@app.get("/api/live/status")
def live_status() -> dict:
    """Freshness of the daily job and of each arm's latest issued prediction."""
    status_path = _live_dir() / "last_run.json"
    last_run = _read_json(status_path) if status_path.is_file() else None
    arms = []
    for name, path in _live_arm_dirs().items():
        latest_path = path / "latest.parquet"
        if not latest_path.is_file():
            continue
        frame = pd.read_parquet(latest_path)
        obs = pd.to_datetime(frame["observation_date"]).max()
        arms.append(
            {
                "arm": name,
                "observation_date": obs.date().isoformat(),
                "issued_at": (
                    str(frame["issued_at"].iloc[0])
                    if "issued_at" in frame.columns
                    else None
                ),
            }
        )
    return {"last_run": last_run, "arms": arms}


PRIMARY_LIVE_ARM = "lstm_price_sentiment"
# Matches ROLLING_WINDOW in stf.forecasting.sentiment_agg: the model reads a
# trailing 5-session sentiment window, so the news shown to explain a
# prediction covers the same span.
NEWS_LOOKBACK_SESSIONS = 5


@app.get("/api/live/today")
def live_today() -> dict:
    """User-facing view: next-session call per ticker with its driving news.

    Joins the primary arm's latest issued predictions with the last close, the
    issued class boundaries (as an expected price band), and the articles the
    sentiment features were computed from.
    """
    live = _live_dir() / PRIMARY_LIVE_ARM
    latest_path = live / "latest.parquet"
    if not latest_path.is_file():
        raise HTTPException(status_code=404, detail="no live predictions yet")
    predictions = pd.read_parquet(latest_path)

    manifest_path = Path(config.ROOT) / "models" / "forecast" / PRIMARY_LIVE_ARM / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    thresholds = manifest.get("thresholds")
    if thresholds is None and {"threshold_low", "threshold_high"} <= set(
        predictions.columns
    ):
        thresholds = [
            float(predictions["threshold_low"].iloc[0]),
            float(predictions["threshold_high"].iloc[0]),
        ]

    obs = pd.to_datetime(predictions["observation_date"]).max()

    # Last close per ticker for the expected price band.
    closes: dict[str, float] = {}
    for ticker in predictions["ticker"]:
        price_path = config.PRICES_DIR / f"{ticker}.parquet"
        if price_path.is_file():
            prices = pd.read_parquet(price_path)
            closes[ticker] = float(prices["close"].iloc[-1])

    # Articles inside the model's trailing sentiment window, joined with their
    # scored probabilities and titles.
    news_by_ticker: dict[str, list[dict]] = {t: [] for t in predictions["ticker"]}
    scored_path = (
        Path(config.ROOT) / "data" / "processed" / "news_sentiment_merged.parquet"
    )
    articles_path = config.ARTICLES_PQ
    if scored_path.is_file() and articles_path.is_file():
        scored = pd.read_parquet(scored_path)
        articles = pd.read_parquet(articles_path)[["url", "title"]]
        merged = scored.merge(articles, on="url", how="left")
        merged["published_at"] = pd.to_datetime(
            merged["published_at"], format="ISO8601", utc=True
        )
        cutoff = merged["published_at"].max() - pd.Timedelta(
            days=NEWS_LOOKBACK_SESSIONS + 2
        )
        recent = merged[merged["published_at"] >= cutoff]
        for ticker, group in recent.groupby("ticker"):
            if ticker not in news_by_ticker:
                continue
            group = group.assign(
                conviction=(group[["prob_negative", "prob_positive"]].max(axis=1))
            ).nlargest(5, "conviction")
            news_by_ticker[ticker] = [
                {
                    "title": row["title"] if isinstance(row["title"], str) else None,
                    "url": row["url"],
                    "published_at": row["published_at"].isoformat(),
                    "prob_negative": float(row["prob_negative"]),
                    "prob_neutral": float(row["prob_neutral"]),
                    "prob_positive": float(row["prob_positive"]),
                }
                for _, row in group.iterrows()
            ]

    tickers = []
    for _, row in predictions.iterrows():
        ticker = row["ticker"]
        close = closes.get(ticker)
        band = None
        if thresholds is not None and close is not None:
            band = {
                "low": round(close * (1 + thresholds[0]), 2),
                "high": round(close * (1 + thresholds[1]), 2),
            }
        tickers.append(
            {
                "ticker": ticker,
                "y_pred": row["y_pred"],
                "confidence": float(
                    row[["prob_down", "prob_flat", "prob_up"]].max()
                ),
                "prob_down": float(row["prob_down"]),
                "prob_flat": float(row["prob_flat"]),
                "prob_up": float(row["prob_up"]),
                "has_news": bool(row["has_news"]),
                "last_close": close,
                "expected_band": band,
                "news": news_by_ticker.get(ticker, []),
            }
        )

    return {
        "arm": PRIMARY_LIVE_ARM,
        "observation_date": obs.date().isoformat(),
        "issued_at": (
            str(predictions["issued_at"].iloc[0])
            if "issued_at" in predictions.columns
            else None
        ),
        "thresholds": thresholds,
        "tickers": tickers,
    }

def mount_frontend() -> None:
    """Serve the built dashboard when ``web/dist`` exists."""
    if not WEB_DIST.is_dir():
        return
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


mount_frontend()
