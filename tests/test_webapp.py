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


def test_legacy_artifact_keys_are_normalized(client, tmp_path, monkeypatch):
    """Pre-rename runs expose canonical keys through the API."""
    run_dir = tmp_path / "outputs" / "legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "forecast_results.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "news_sentiment": {
                        "path": "data/processed/news_sentiment.parquet",
                        "hash": "abc123",
                        "rows": 5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "information_gain.json").write_text(
        json.dumps(
            {
                "arm": "lstm_price_sentiment",
                "price_arm": "lstm_price",
                "effects": {
                    "macro_f1": {
                        "architecture_effect": 0.01,
                        "information_gain": 0.02,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runs = {run["name"]: run for run in client.get("/api/runs").json()["runs"]}
    assert runs["legacy"]["mode"] == "real"

    news = client.get("/api/runs/legacy/summary").json()["provenance"]["news_sentiment"]
    assert news["source_path"] == "data/processed/news_sentiment.parquet"
    assert news["source_hash"] == "abc123"
    assert "path" not in news and "hash" not in news

    effect = client.get("/api/runs/legacy/information-gain").json()["effects"]["macro_f1"]
    assert effect["architecture_and_news_presence_volume_effect"] == 0.01
    assert "architecture_effect" not in effect
