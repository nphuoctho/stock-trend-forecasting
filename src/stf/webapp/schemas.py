"""Pydantic response models: the OpenAPI contract of the read-only API.

The artifacts under ``outputs/`` are written by several generations of the
pipeline, so the models describe the keys that are real and allow extras
instead of dropping data. Endpoints serialise with ``exclude_unset`` (see
``stf.webapp.app``), so a key absent from an artifact stays absent on the wire
and the JSON is byte-for-byte what the handlers built; a model never injects
``null`` defaults.

Fields declared ``X | None`` without a default are always present (possibly
null); fields with a ``= None`` default may be absent from older artifacts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Loose(BaseModel):
    """Base for artifact-derived models: unknown keys pass through unchanged."""

    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    detail: str


NewsMode = Literal["real", "neutral_prior"]


# --- runs -------------------------------------------------------------------


class RunInfo(Loose):
    name: str
    panel_rows: int | None
    mode: NewsMode | None
    date_start: str | None
    date_end: str | None
    has_information_gain: bool


class RunsResponse(Loose):
    runs: list[RunInfo]


class MeanStd(Loose):
    mean: float | None
    std: float | None
    n: int


class BootstrapCI(Loose):
    mean: float | None
    low: float | None
    high: float | None
    one_sided_p_le_zero: float | None = None
    n_blocks: int | None = None
    n_window_strata: int | None = None


class AblationMetric(Loose):
    per_window: list[float | None]
    window_bootstrap: BootstrapCI
    date_block_bootstrap: BootstrapCI


class StratumStat(Loose):
    n_mean: float | None = None
    macro_f1_mean: float | None = None
    balanced_accuracy_mean: float | None = None
    accuracy_mean: float | None = None


class NewsStrata(Loose):
    has_news: StratumStat | None = None
    no_news: StratumStat | None = None


class NewsSentimentProvenance(Loose):
    mode: NewsMode | None = None
    source_path: str | None = None
    source_hash: str | None = None
    feature_hash: str | None = None
    rows: int | None = None


class NewsCoverage(Loose):
    panel_rows: int | None = None
    rows_with_news: int | None = None
    fraction: float | None = None


class Provenance(Loose):
    tickers: list[str] | None = None
    date_start: str | None = None
    date_end: str | None = None
    session_cutoff: str | None = None
    timezone: str | None = None
    prices_hash: str | None = None
    news_sentiment: NewsSentimentProvenance | None = None
    panel_start: str | None = None
    panel_end: str | None = None
    alignment_report: dict[str, int | float] | None = None
    alignment_report_in_window: dict[str, Any] | None = None
    panel_hash: str | None = None
    news_coverage: NewsCoverage | None = None


class RunSummary(Loose):
    name: str
    config: dict[str, Any] | None
    chance_level: float | None
    panel_rows: int | None
    panel_tickers: list[str] | None
    # arm -> metric -> mean/std/n
    summary: dict[str, dict[str, MeanStd]] | None
    # "<arm>__minus__<arm>" -> metric -> paired bootstrap
    ablation: dict[str, dict[str, AblationMetric]] | None
    # arm -> has_news / no_news aggregates
    stratified_by_news: dict[str, NewsStrata] | None
    environment: dict[str, Any] | None
    provenance: Provenance | None


# --- per-run CSV tables ------------------------------------------------------


class MetricsRow(Loose):
    """One arm's metric row; the metric columns vary by pipeline version."""

    arm: str


class MetricsResponse(Loose):
    rows: list[MetricsRow]


class Prediction(Loose):
    # Declared in the CSV's column order so the JSON key order is unchanged.
    ticker: str
    observation_date: str | None = None
    target_date: str | None = None
    window: int
    seed: int | None = None
    arm: str
    y_true: str
    y_pred: str
    has_news: int | None = None


class PredictionsResponse(Loose):
    # Row count after filtering, before pagination.
    total: int
    rows: list[Prediction]


class StratifiedRow(Loose):
    window: int
    seed: int | None = None
    arm: str
    stratum: str
    n: int | None = None
    macro_f1: float | None = None
    balanced_accuracy: float | None = None
    accuracy: float | None = None


class StratifiedResponse(Loose):
    rows: list[StratifiedRow]


class InformationGainEffect(Loose):
    price_only: float | None = None
    two_branch_neutral_prior: float | None = None
    two_branch_real_sentiment: float | None = None
    architecture_and_news_presence_volume_effect: float | None = None
    information_gain: float | None = None
    naive_delta: float | None = None
    information_gain_per_window: list[float | None] | None = None
    information_gain_window_bootstrap: BootstrapCI | None = None
    information_gain_date_bootstrap: BootstrapCI | None = None


class InformationGain(Loose):
    arm: str
    price_arm: str
    config: dict[str, Any] | None = None
    # metric name -> effect decomposition
    effects: dict[str, InformationGainEffect]


# --- live serving ------------------------------------------------------------


class LivePrediction(Loose):
    ticker: str
    observation_date: str
    has_news: int
    prob_down: float
    prob_flat: float
    prob_up: float
    y_pred: str
    arm: str


class LiveArmLatest(Loose):
    arm: str
    observation_date: str
    rows: list[LivePrediction]


class LiveLatestResponse(Loose):
    arms: list[LiveArmLatest]


class LiveDateAccuracy(Loose):
    target_date: str
    n: int
    accuracy: float


class LiveArmHistory(Loose):
    arm: str
    total: int
    resolved: int
    prospective_resolved: int
    replayed_resolved: int
    pending: int
    # Over prospective rows only; null until one resolves.
    accuracy: float | None
    by_date: list[LiveDateAccuracy]


class LiveHistoryResponse(Loose):
    arms: list[LiveArmHistory]


class LastRun(Loose):
    finished_at: str
    ok: bool
    exit_code: int
    failed_step: str | None


class LiveArmStatus(Loose):
    arm: str
    observation_date: str
    issued_at: str | None


class LiveStatusResponse(Loose):
    last_run: LastRun | None
    arms: list[LiveArmStatus]


class LiveTodayNews(Loose):
    title: str | None
    url: str
    published_at: str
    prob_negative: float
    prob_neutral: float
    prob_positive: float


class FlatBand(Loose):
    low: float
    high: float


class LiveTodayTicker(Loose):
    ticker: str
    y_pred: Literal["UP", "FLAT", "DOWN"]
    confidence: float
    prob_down: float
    prob_flat: float
    prob_up: float
    has_news: bool
    last_close: float | None
    # Issued class boundaries applied to last_close; not a price forecast.
    flat_band: FlatBand | None
    # Observation predates the panel's latest session.
    stale: bool
    news: list[LiveTodayNews]


class LiveTodayResponse(Loose):
    arm: str
    observation_date: str
    issued_at: str | None
    tickers: list[LiveTodayTicker]
