"""FastAPI surface exposing forecast experiment artifacts.

The API is read-only: every endpoint derives from files under ``outputs/`` and
never mutates them. Run directories are discovered by the presence of
``forecast_results.json``.

It is meant to be public (behind a Cloudflare Tunnel), so :func:`create_app`
layers the hardening from :mod:`stf.webapp.security` on top of the routes and
every endpoint declares a Pydantic response model (:mod:`stf.webapp.schemas`),
which is what ``/openapi.json`` and the committed ``openapi.json`` describe.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from stf import config
from stf.webapp import schemas
from stf.webapp.security import (
    CORS_ALLOWED_METHODS,
    CORS_EXPOSED_HEADERS,
    ErrorGuardMiddleware,
    EventsLimitMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    SecuritySettings,
    TokenBucketLimiter,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = REPO_ROOT / "web" / "dist"

API_TITLE = "stock-trend-forecasting"
API_VERSION = "1.0.0"

# Upper bounds on caller-controlled query/path parameters.
MAX_NAME_LENGTH = 128
MAX_FILTER_LENGTH = 64
MAX_PAGE_SIZE = 5000
MAX_OFFSET = 1_000_000
MAX_WINDOW = 1000

router = APIRouter(
    responses={
        429: {
            "model": schemas.ErrorResponse,
            "description": "Rate limit exceeded; see the Retry-After header (seconds).",
            "headers": {"Retry-After": {"schema": {"type": "integer"}}},
        }
    }
)

# Every handler builds exactly the JSON it returns, so the response models are
# applied with exclude_unset: a key an artifact does not carry is not invented.
_MODELED = {"response_model_exclude_unset": True}


def _not_found(description: str) -> dict:
    return {404: {"model": schemas.ErrorResponse, "description": description}}


RunName = Annotated[str, PathParam(max_length=MAX_NAME_LENGTH)]


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
        # The file name stays in the log; the client gets the same opaque body
        # as any other unhandled error.
        logger.error("cannot read artifact %s: %s", path.name, error)
        raise HTTPException(status_code=500, detail="internal server error") from error


def _read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"missing artifact {path.name}")
    return pd.read_csv(path)


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def _read_csv_records(path: Path) -> list[dict]:
    return _records(_read_csv_frame(path))


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


@router.get(
    "/api/runs",
    response_model=schemas.RunsResponse,
    summary="List forecast runs",
    **_MODELED,
)
def list_runs() -> dict[str, Any]:
    """List forecast run directories with headline provenance."""
    runs = []
    for name, path in _run_dirs().items():
        results = _read_json(path / "forecast_results.json")
        provenance = results.get("provenance") or {}
        news = _normalize_news_sentiment(provenance.get("news_sentiment") or {})
        config = results.get("config") or {}
        runs.append(
            {
                "name": name,
                "panel_rows": results.get("panel_rows"),
                "mode": news.get("mode"),
                # Runs differ in what they predict, not just in their news source.
                # Without these two fields an excess-return or multi-session run is
                # indistinguishable from the headline next-session run in the list.
                "target_mode": config.get("target_mode", "raw"),
                "horizon": config.get("horizon", 1),
                "point_in_time": news.get("point_in_time"),
                "date_start": provenance.get("date_start"),
                "date_end": provenance.get("date_end"),
                "has_information_gain": (path / "information_gain.json").is_file(),
            }
        )
    return {"runs": runs}


@router.get(
    "/api/runs/{name}/summary",
    response_model=schemas.RunSummary,
    responses=_not_found("Unknown run."),
    **_MODELED,
)
def run_summary(name: RunName) -> dict[str, Any]:
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


@router.get(
    "/api/runs/{name}/metrics",
    response_model=schemas.MetricsResponse,
    responses=_not_found("Unknown run, or the run has no forecast_metrics.csv."),
    **_MODELED,
)
def run_metrics(name: RunName) -> dict[str, Any]:
    """Per-arm metric table with bootstrap intervals."""
    return {"rows": _read_csv_records(_run_dir(name) / "forecast_metrics.csv")}


@router.get(
    "/api/runs/{name}/predictions",
    response_model=schemas.PredictionsResponse,
    responses=_not_found("Unknown run, or the run has no forecast_predictions.csv."),
    **_MODELED,
)
def run_predictions(
    name: RunName,
    arm: str | None = Query(default=None, max_length=MAX_FILTER_LENGTH),
    ticker: str | None = Query(default=None, max_length=MAX_FILTER_LENGTH),
    window: int | None = Query(default=None, ge=0, le=MAX_WINDOW),
    limit: int = Query(default=500, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0, le=MAX_OFFSET),
) -> dict[str, Any]:
    """Paginated per-row predictions, filterable by arm/ticker/window."""
    frame = _read_csv_frame(_run_dir(name) / "forecast_predictions.csv")
    for column, value in (("arm", arm), ("ticker", ticker), ("window", window)):
        if value is not None:
            frame = frame[frame[column] == value] if column in frame else frame.iloc[:0]
    return {
        "total": len(frame),
        "rows": _records(frame.iloc[offset : offset + limit]),
    }


@router.get(
    "/api/runs/{name}/stratified",
    response_model=schemas.StratifiedResponse,
    responses=_not_found("Unknown run, or the run has no forecast_stratified.csv."),
    **_MODELED,
)
def run_stratified(name: RunName) -> dict[str, Any]:
    """Per-window metrics split by news presence."""
    return {"rows": _read_csv_records(_run_dir(name) / "forecast_stratified.csv")}


@router.get(
    "/api/runs/{name}/information-gain",
    response_model=schemas.InformationGain,
    responses=_not_found("Unknown run, or the run has no information_gain.json."),
    **_MODELED,
)
def run_information_gain(name: RunName) -> dict[str, Any]:
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


@router.get(
    "/api/live/latest",
    response_model=schemas.LiveLatestResponse,
    responses=_not_found("No live predictions have been issued yet."),
    **_MODELED,
)
def live_latest() -> dict[str, Any]:
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


@router.get(
    "/api/live/history",
    response_model=schemas.LiveHistoryResponse,
    responses=_not_found("No resolved predictions yet."),
    **_MODELED,
)
def live_history() -> dict[str, Any]:
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


@router.get(
    "/api/live/status",
    response_model=schemas.LiveStatusResponse,
    **_MODELED,
)
def live_status() -> dict[str, Any]:
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


@router.get(
    "/api/live/today",
    response_model=schemas.LiveTodayResponse,
    responses=_not_found("No live predictions have been issued yet."),
    **_MODELED,
)
def live_today() -> dict[str, Any]:
    """User-facing view: next-session call per ticker with related news.

    Joins the primary arm's latest issued predictions with the last close, the
    issued class boundaries (terciles of all labeled historical returns -- NOT a
    predicted price interval; magnitude forecasting is not implemented), and the
    articles that fed the sentiment features. The articles are context the model
    read, not proven causes: no attribution method is applied.
    """
    live = _live_dir() / PRIMARY_LIVE_ARM
    latest_path = live / "latest.parquet"
    if not latest_path.is_file():
        raise HTTPException(status_code=404, detail="no live predictions yet")
    predictions = pd.read_parquet(latest_path)

    # Prefer the boundaries stamped into the issued rows; the manifest is only a
    # fallback for predictions written before stamping existed, so a later refit
    # cannot change what an old prediction displays.
    manifest_path = (
        Path(config.ROOT) / "models" / "forecast" / PRIMARY_LIVE_ARM / "manifest.json"
    )
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    manifest_thresholds = manifest.get("thresholds")

    obs = pd.to_datetime(predictions["observation_date"]).max()

    # Close at the observation session per ticker (not the latest close on disk:
    # prices fetched after issuance must not move the displayed band).
    closes: dict[str, float] = {}
    sessions: dict[str, np.ndarray] = {}
    for ticker in predictions["ticker"]:
        price_path = config.PRICES_DIR / f"{ticker}.parquet"
        if not price_path.is_file():
            continue
        prices = pd.read_parquet(price_path)
        prices["_session"] = pd.to_datetime(prices["time"]).dt.normalize()
        sessions[ticker] = np.sort(prices["_session"].unique())
        at_obs = prices[prices["_session"] <= obs.normalize()]
        if len(at_obs):
            closes[ticker] = float(at_obs["close"].iloc[-1])

    # Articles that actually fed the prediction: anchored to one of the last
    # NEWS_LOOKBACK_SESSIONS sessions ending at the observation date, using the
    # same session alignment the model's features were built with. News filed
    # after the session cutoff anchors to the NEXT session and is excluded.
    news_by_ticker: dict[str, list[dict]] = {t: [] for t in predictions["ticker"]}
    scored_path = (
        Path(config.ROOT) / "data" / "processed" / "news_sentiment_merged.parquet"
    )
    articles_path = config.ARTICLES_PQ
    if scored_path.is_file() and articles_path.is_file() and sessions:
        from stf.forecasting import calendar as cal

        scored = pd.read_parquet(scored_path)
        articles = pd.read_parquet(articles_path)[["url", "title"]]
        merged = scored.merge(articles, on="url", how="left")
        aligned = cal.align_news_to_sessions(merged, sessions)
        mapped = aligned[aligned["mapping_status"].isin(cal.MAPPED_STATUSES)]
        for ticker, group in mapped.groupby("ticker"):
            if ticker not in news_by_ticker or ticker not in sessions:
                continue
            window_sessions = sessions[ticker][sessions[ticker] <= obs.normalize()][
                -NEWS_LOOKBACK_SESSIONS:
            ]
            group = group[group["observation_date"].isin(window_sessions)]
            group = group.assign(
                conviction=(group[["prob_negative", "prob_positive"]].max(axis=1))
            ).nlargest(5, "conviction")
            news_by_ticker[ticker] = [
                {
                    "title": row["title"] if isinstance(row["title"], str) else None,
                    "url": row["url"],
                    "published_at": str(row["published_at"]),
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
        if {"threshold_low", "threshold_high"} <= set(predictions.columns):
            low_t = row["threshold_low"]
            high_t = row["threshold_high"]
            thresholds = (
                [float(low_t), float(high_t)]
                if pd.notna(low_t) and pd.notna(high_t)
                else manifest_thresholds
            )
        else:
            thresholds = manifest_thresholds
        band = None
        if thresholds is not None and close is not None:
            band = {
                "low": round(close * (1 + thresholds[0]), 2),
                "high": round(close * (1 + thresholds[1]), 2),
            }
        # A ticker whose observation predates the panel's latest session is a
        # stale prediction (e.g. its price fetch failed), not a call for the
        # next session.
        row_obs = pd.to_datetime(row["observation_date"])
        tickers.append(
            {
                "ticker": ticker,
                "y_pred": row["y_pred"],
                "confidence": float(row[["prob_down", "prob_flat", "prob_up"]].max()),
                "prob_down": float(row["prob_down"]),
                "prob_flat": float(row["prob_flat"]),
                "prob_up": float(row["prob_up"]),
                "has_news": bool(row["has_news"]),
                "last_close": close,
                "flat_band": band,
                "stale": bool(row_obs < obs),
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
        "tickers": tickers,
    }


def mount_frontend(target: FastAPI) -> None:
    """Serve the built dashboard when ``web/dist`` exists."""
    if not WEB_DIST.is_dir():
        return
    target.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def create_app(env: Mapping[str, str] | None = None) -> FastAPI:
    """Build the API with hardening configured from ``env`` (default: process env).

    Middleware order, outermost first: security headers, trusted hosts, CORS,
    error guard, rate limit, events limits. Headers therefore reach every
    response (preflights, 400s and 500s included), CORS headers reach 429s and
    500s so the browser portal can read them, and preflights are never rate
    counted.
    """
    settings = SecuritySettings.from_env(os.environ if env is None else env)
    application = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=(
            "Read-only forecast artifacts: experiment runs and live daily predictions."
        ),
        docs_url="/api/docs",
    )
    application.include_router(router)
    mount_frontend(application)

    application.add_middleware(
        EventsLimitMiddleware,
        handshake=TokenBucketLimiter(settings.events_connect_per_minute, 60.0),
        max_streams_per_ip=settings.events_max_per_ip,
        trust_cf_headers=settings.trust_cf_headers,
    )
    if settings.rate_limit > 0:
        application.add_middleware(
            RateLimitMiddleware,
            limiter=TokenBucketLimiter(
                settings.rate_limit, settings.rate_window_seconds
            ),
            trust_cf_headers=settings.trust_cf_headers,
        )
    application.add_middleware(ErrorGuardMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=False,
        allow_methods=CORS_ALLOWED_METHODS,
        allow_headers=[],
        expose_headers=CORS_EXPOSED_HEADERS,
    )
    if settings.allowed_hosts:
        application.add_middleware(
            TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts
        )
    application.add_middleware(SecurityHeadersMiddleware)
    return application


def render_openapi() -> str:
    """The OpenAPI document as committed to ``openapi.json``.

    Independent of the environment (middleware is not part of the schema) and
    key-sorted so the text is stable across runs.
    """
    document = create_app({}).openapi()
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


app = create_app()
