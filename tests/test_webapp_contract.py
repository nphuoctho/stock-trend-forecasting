"""The response models must declare every key the handlers build.

The models allow extra keys (artifacts written by older pipelines carry columns
nobody declared) and endpoints serialise with ``exclude_unset``, so a handler
that adds a key still produces valid output while ``openapi.json`` silently
lacks it. This test builds every endpoint's response from a complete fixture
and fails on any key the model does not declare.

The fixture's artifact-derived rows carry only declared columns: what is pinned
here is the handler-built structure, not the open-ended artifact passthrough.

uv run pytest tests/test_webapp_contract.py -q
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel

from stf import config
from stf.webapp import app as webapp_module
from stf.webapp import schemas

RUN = "run_a"
PRIMARY_ARM = webapp_module.PRIMARY_LIVE_ARM

ENDPOINTS = {
    "/api/runs": "/api/runs",
    "/api/runs/{name}/summary": f"/api/runs/{RUN}/summary",
    "/api/runs/{name}/metrics": f"/api/runs/{RUN}/metrics",
    "/api/runs/{name}/predictions": f"/api/runs/{RUN}/predictions",
    "/api/runs/{name}/stratified": f"/api/runs/{RUN}/stratified",
    "/api/runs/{name}/information-gain": f"/api/runs/{RUN}/information-gain",
    "/api/live/latest": "/api/live/latest",
    "/api/live/history": "/api/live/history",
    "/api/live/status": "/api/live/status",
    "/api/live/today": "/api/live/today",
}


def undeclared_keys(value: Any, path: str = "$") -> list[str]:
    """Paths of keys that validated only because the model allows extras."""
    found: list[str] = []
    if isinstance(value, BaseModel):
        found += [f"{path}.{key}" for key in value.model_extra or {}]
        for name in type(value).model_fields:
            found += undeclared_keys(getattr(value, name), f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += undeclared_keys(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            found += undeclared_keys(item, f"{path}.{key}")
    return found


def _bootstrap() -> dict:
    return {
        "mean": 0.1,
        "low": 0.0,
        "high": 0.2,
        "one_sided_p_le_zero": 0.05,
        "n_blocks": 4,
        "n_window_strata": 2,
    }


@pytest.fixture()
def complete_outputs(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / RUN
    run_dir.mkdir(parents=True)
    (run_dir / "forecast_results.json").write_text(
        json.dumps(
            {
                "config": {"seq_len": 5, "target_mode": "excess", "horizon": 2},
                "chance_level": 0.33,
                "panel_rows": 10,
                "panel_tickers": ["FPT"],
                "summary": {
                    "majority": {"macro_f1": {"mean": 0.1, "std": 0.0, "n": 1}}
                },
                "ablation": {
                    "a__minus__b": {
                        "macro_f1": {
                            "per_window": [0.1, None],
                            "window_bootstrap": _bootstrap(),
                            "date_block_bootstrap": _bootstrap(),
                        }
                    }
                },
                "stratified_by_news": {
                    "majority": {
                        "has_news": {"n_mean": 3.0, "macro_f1_mean": 0.2},
                        "no_news": {"n_mean": 7.0, "macro_f1_mean": 0.1},
                    }
                },
                "environment": {"python": "3.12"},
                "provenance": {
                    "tickers": ["FPT"],
                    "date_start": "2020-01-01",
                    "date_end": "2025-12-31",
                    "session_cutoff": "15:00",
                    "timezone": "Asia/Ho_Chi_Minh",
                    "prices_hash": "abc",
                    "news_sentiment": {
                        "mode": "real",
                        "source_path": "x.parquet",
                        "source_hash": "def",
                        "feature_hash": "ghi",
                        "rows": 3,
                    },
                    "panel_start": "2020-01-01",
                    "panel_end": "2025-12-31",
                    "alignment_report": {"mapped": 3},
                    "alignment_report_in_window": {"mapped": 3},
                    "panel_hash": "jkl",
                    "news_coverage": {
                        "panel_rows": 10,
                        "rows_with_news": 3,
                        "fraction": 0.3,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "information_gain.json").write_text(
        json.dumps(
            {
                "arm": "b",
                "price_arm": "a",
                "config": {"seq_len": 5},
                "effects": {
                    "macro_f1": {
                        "price_only": 0.1,
                        "two_branch_neutral_prior": 0.1,
                        "two_branch_real_sentiment": 0.2,
                        "architecture_and_news_presence_volume_effect": 0.0,
                        "information_gain": 0.1,
                        "naive_delta": 0.1,
                        "information_gain_per_window": [0.1],
                        "information_gain_window_bootstrap": _bootstrap(),
                        "information_gain_date_bootstrap": _bootstrap(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"arm": ["majority"]}).to_csv(
        run_dir / "forecast_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "observation_date": ["2025-01-02"],
            "target_date": ["2025-01-03"],
            "window": [1],
            "seed": [7],
            "arm": ["majority"],
            "y_true": ["UP"],
            "y_pred": ["UP"],
            "has_news": [1],
        }
    ).to_csv(run_dir / "forecast_predictions.csv", index=False)
    pd.DataFrame(
        {
            "window": [1],
            "seed": [7],
            "arm": ["majority"],
            "stratum": ["has_news"],
            "n": [3],
            "macro_f1": [0.2],
            "balanced_accuracy": [0.3],
            "accuracy": [0.4],
        }
    ).to_csv(run_dir / "forecast_stratified.csv", index=False)

    live = tmp_path / "outputs" / "live"
    arm_dir = live / PRIMARY_ARM
    arm_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "observation_date": pd.to_datetime(["2026-09-23"]),
            "has_news": [1],
            "prob_down": [0.2],
            "prob_flat": [0.3],
            "prob_up": [0.5],
            "y_pred": ["UP"],
            "arm": [PRIMARY_ARM],
        }
    ).to_parquet(arm_dir / "latest.parquet", index=False)
    pd.DataFrame(
        {
            "ticker": ["FPT", "VNM"],
            "observation_date": pd.to_datetime(["2026-09-18", "2026-09-21"]),
            "target_date": pd.to_datetime(["2026-09-21", "2026-09-22"]),
            "y_pred": ["UP", "UP"],
            "y_true": ["UP", "DOWN"],
            "correct": [True, False],
            "prospective": [True, False],
        }
    ).to_parquet(arm_dir / "resolved.parquet", index=False)
    (live / "last_run.json").write_text(
        json.dumps(
            {
                "finished_at": "2026-09-23T08:35:00+00:00",
                "ok": True,
                "exit_code": 0,
                "failed_step": None,
            }
        ),
        encoding="utf-8",
    )

    model_dir = tmp_path / "models" / "forecast" / PRIMARY_ARM
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps({"thresholds": [-0.01, 0.01]}), encoding="utf-8"
    )
    prices_dir = tmp_path / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["FPT"] * 3,
            "time": pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23"]),
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.0, 101.0, 102.0],
            "volume": [1] * 3,
        }
    ).to_parquet(prices_dir / "FPT.parquet", index=False)
    news_dir = tmp_path / "data" / "raw" / "news"
    news_dir.mkdir(parents=True)
    pd.DataFrame({"url": ["u1"], "title": ["tin"]}).to_parquet(
        news_dir / "articles.parquet", index=False
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "url": ["u1"],
            "published_at": ["2026-09-23T10:00:00+07:00"],
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    ).to_parquet(processed / "news_sentiment_merged.parquet", index=False)

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "PRICES_DIR", prices_dir)
    monkeypatch.setattr(config, "ARTICLES_PQ", news_dir / "articles.parquet")
    return tmp_path


def test_fixture_covers_every_api_route():
    routes = {
        route.path
        for route in webapp_module.router.routes
        if isinstance(route, APIRoute) and route.path.startswith("/api/")
    }
    assert routes == set(ENDPOINTS)


@pytest.mark.parametrize("route_path", sorted(ENDPOINTS))
def test_models_declare_every_key_the_handler_emits(complete_outputs, route_path):
    route = next(
        r
        for r in webapp_module.router.routes
        if isinstance(r, APIRoute) and r.path == route_path
    )
    response = TestClient(webapp_module.create_app({})).get(ENDPOINTS[route_path])
    assert response.status_code == 200, response.text

    model = route.response_model.model_validate(response.json())
    assert undeclared_keys(model) == []


def test_a_key_the_model_does_not_declare_is_reported():
    entry = {
        "name": "run_a",
        "panel_rows": 1,
        "mode": None,
        "target_mode": "raw",
        "horizon": 1,
        "point_in_time": None,
        "date_start": None,
        "date_end": None,
        "has_information_gain": False,
        "added_by_a_handler": True,
    }
    model = schemas.RunsResponse.model_validate({"runs": [entry]})
    assert undeclared_keys(model) == ["$.runs[0].added_by_a_handler"]
