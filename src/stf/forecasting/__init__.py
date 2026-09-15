"""Point-in-time forecasting feature and model core.

Pipeline: align news to trading sessions -> aggregate daily sentiment -> build causal
price features and next-session targets -> assemble the panel -> fit train-only trend
thresholds -> chronological / walk-forward splits -> baseline and LSTM models.

The heavy torch model core lives in :mod:`stf.forecasting.models` and is imported lazily
so lightweight feature work does not require torch to be importable.
"""

from __future__ import annotations

from stf.forecasting.calendar import (
    MAPPED_STATUSES,
    PROB_COLS,
    align_news_to_sessions,
    alignment_report,
    session_as_of,
    trading_sessions,
)
from stf.forecasting.features import (
    FeatureScaler,
    MA_WINDOW,
    VOL_WINDOW,
    price_feature_columns,
    price_features,
)
from stf.forecasting.labels import (
    ID2TREND,
    TREND2ID,
    TREND_LABELS,
    add_target,
    apply_labels,
    fit_thresholds,
    label_panel,
)
from stf.forecasting.panel import assemble, build_panel, panel_columns
from stf.forecasting.sentiment_agg import SENTIMENT_COLUMNS, daily_sentiment
from stf.forecasting.split import (
    TimeSplit,
    chronological_split,
    walk_forward_windows,
)

__all__ = [
    "MAPPED_STATUSES",
    "PROB_COLS",
    "align_news_to_sessions",
    "alignment_report",
    "session_as_of",
    "trading_sessions",
    "FeatureScaler",
    "price_feature_columns",
    "price_features",
    "SENTIMENT_COLUMNS",
    "daily_sentiment",
    "ID2TREND",
    "TREND2ID",
    "TREND_LABELS",
    "add_target",
    "apply_labels",
    "fit_thresholds",
    "label_panel",
    "assemble",
    "build_panel",
    "panel_columns",
    "TimeSplit",
    "chronological_split",
    "walk_forward_windows",
]


def __getattr__(name: str):
    """Lazily expose the torch model core (imported on first access)."""
    model_exports = {
        "PriceLSTM",
        "PriceSentimentLSTM",
        "MajorityBaseline",
        "RandomBaseline",
        "make_sequences",
        "make_two_branch_sequences",
        "fit_lstm",
        "set_seed",
        "NUM_TREND_CLASSES",
    }
    if name in model_exports:
        from stf.forecasting import models

        return getattr(models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
