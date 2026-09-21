"""Webapp API contract tests - run offline, no network access.

uv run pytest tests/test_webapp.py -q
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stf.webapp import app as webapp_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "forecast_results.json").write_text(
        json.dumps(
            {
                "config": {"seq_len": 5},
                "panel_rows": 10,
                "summary": {"majority": {"macro_f1": {"mean": 0.1, "std": 0.0, "n": 1}}},
                "provenance": {
                    "date_start": "2020-01-01",
                    "date_end": "2025-12-31",
                    "news_sentiment": {"mode": "real", "rows": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame({"arm": ["majority"], "macro_f1_mean": [0.1]}).to_csv(
        run_dir / "forecast_metrics.csv", index=False
    )
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "arm": ["majority"],
            "window": [1],
            "y_true": ["UP"],
            "y_pred": ["UP"],
        }
    ).to_csv(run_dir / "forecast_predictions.csv", index=False)
    pd.DataFrame({"window": [1], "arm": ["majority"], "stratum": ["has_news"]}).to_csv(
        run_dir / "forecast_stratified.csv", index=False
    )
    monkeypatch.setattr(webapp_module, "_outputs_root", lambda: tmp_path / "outputs")
    return TestClient(webapp_module.app)


def test_runs_listing_and_summary(client):
    runs = client.get("/api/runs").json()["runs"]
    assert [run["name"] for run in runs] == ["run_a"]
    assert runs[0]["mode"] == "real"

    summary = client.get("/api/runs/run_a/summary").json()
    assert summary["summary"]["majority"]["macro_f1"]["mean"] == 0.1
    assert summary["provenance"]["news_sentiment"]["rows"] == 3


def test_run_scoped_endpoints_and_filters(client):
    metrics = client.get("/api/runs/run_a/metrics").json()["rows"]
    assert metrics[0]["arm"] == "majority"

    predictions = client.get("/api/runs/run_a/predictions?arm=majority").json()
    assert predictions["total"] == 1
    assert client.get("/api/runs/run_a/predictions?arm=lstm_price").json()["total"] == 0

    assert client.get("/api/runs/run_a/stratified").json()["rows"][0]["stratum"] == "has_news"


def test_unknown_run_and_missing_information_gain_return_404(client):
    assert client.get("/api/runs/missing/summary").status_code == 404
    assert client.get("/api/runs/run_a/information-gain").status_code == 404
