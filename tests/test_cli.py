"""CLI contract tests that stay local and deterministic."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stf.cli import main


def test_forecast_smoke_command_reports_partitioned_sequences(capsys):
    assert main(["forecast-smoke", "--n", "20", "--epochs", "1"]) == 0
    output = capsys.readouterr().out
    assert "sequences=train:" in output
    assert "val:" in output
    assert "test:" in output


def test_forecast_smoke_rejects_too_short_input(capsys):
    assert main(["forecast-smoke", "--n", "11"]) == 2
    assert "--n must be at least 12" in capsys.readouterr().err


def test_forecast_smoke_trains_both_price_and_fused_models(capsys):
    assert main(["forecast-smoke", "--n", "20", "--epochs", "1"]) == 0
    output = capsys.readouterr().out
    assert "price_lstm_final_loss=" in output
    assert "price_sentiment_lstm_final_loss=" in output


def test_score_news_scores_articles_without_a_real_model(monkeypatch, tmp_path):
    from stf.data import news as news_module
    from stf.sentiment import model as model_module

    articles = pd.DataFrame(
        {
            "ticker": ["FPT", "VNM"],
            "url": ["https://vietstock.vn/a.htm", "https://vietstock.vn/b.htm"],
            "published_at": [
                "2021-01-01T10:00:00+07:00",
                "2021-01-02T10:00:00+07:00",
            ],
            "title": ["Tin A", "Tin B"],
            "body": ["Nội dung A", "Nội dung B"],
        }
    )
    monkeypatch.setattr(news_module, "load_ticker_articles", lambda: articles)

    captured = {}

    def fake_predict_proba(
        texts, model_dir, *, batch_size=32, truncation_strategy=None, max_len=None
    ):
        captured["texts"] = list(texts)
        captured["model_dir"] = model_dir
        captured["truncation_strategy"] = truncation_strategy
        captured["max_len"] = max_len
        return np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])[: len(texts)]

    monkeypatch.setattr(model_module, "predict_proba", fake_predict_proba)

    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "manifest.json").write_text(
        json.dumps({"selection": {"input_variant": "title"}}), encoding="utf-8"
    )
    output = tmp_path / "scored.parquet"
    rc = main(
        [
            "score-news",
            "--model-dir",
            str(model_dir),
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert captured["texts"] == ["Tin A", "Tin B"]
    result = pd.read_parquet(output)
    assert result["ticker"].tolist() == ["FPT", "VNM"]
    assert result["prob_positive"].tolist() == pytest.approx([0.1, 0.7])
    manifest = pd.read_json(output.with_suffix(".manifest.json"), typ="series")
    assert manifest["output"]["rows"] == 2
    assert manifest["probabilities"]["totals"]["prob_neutral"] == pytest.approx(0.4)
    assert manifest["checkpoint"]["directory_sha256"]
    assert manifest["schema_version"] == 2
    assert manifest["inference"]["truncation_strategy"] == "head"
    assert manifest["inference"]["max_len"] == 256
    assert manifest["input"]["variant"] == "title"
    assert manifest["input"]["fingerprint"]
    assert manifest["input"]["limit"] is None

    limited_output = tmp_path / "limited.parquet"
    assert (
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--limit",
                "1",
                "--output",
                str(limited_output),
            ]
        )
        == 0
    )
    limited_manifest = pd.read_json(
        limited_output.with_suffix(".manifest.json"), typ="series"
    )
    assert len(pd.read_parquet(limited_output)) == 1
    assert limited_manifest["input"]["limit"] == 1

    articles = articles.iloc[::-1].reset_index(drop=True)
    reordered_output = tmp_path / "reordered.parquet"
    assert (
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(reordered_output),
            ]
        )
        == 0
    )
    reordered_manifest = pd.read_json(
        reordered_output.with_suffix(".manifest.json"), typ="series"
    )
    assert reordered_manifest["input"]["fingerprint"] == manifest["input"]["fingerprint"]

    articles.loc[0, "title"] = "Tin A đã sửa"
    changed_output = tmp_path / "changed.parquet"
    assert (
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(changed_output),
            ]
        )
        == 0
    )
    changed_manifest = pd.read_json(
        changed_output.with_suffix(".manifest.json"), typ="series"
    )
    assert changed_manifest["input"]["fingerprint"] != manifest["input"]["fingerprint"]

    with pytest.raises(ValueError, match="does not match checkpoint input variant"):
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--input-variant",
                "title_context",
                "--output",
                str(tmp_path / "mismatched.parquet"),
            ]
        )


    monkeypatch.setattr(
        model_module,
        "predict_proba",
        lambda *_args, **_kwargs: np.array([[1.0, 0.1, -0.1], [0.0, 0.0, 1.0]]),
    )
    invalid_output = tmp_path / "invalid.parquet"
    with pytest.raises(RuntimeError, match="invalid probability vectors"):
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(invalid_output),
            ]
        )
    assert not invalid_output.exists()


def test_forecast_binds_verified_score_manifest(monkeypatch, tmp_path, capsys):
    from stf import cli as cli_module
    from stf import forecasting as forecasting_module
    from stf.forecasting import experiment as experiment_module
    from stf.sentiment.dataset import file_fingerprint

    news_path = tmp_path / "scored.parquet"
    news = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "url": ["https://vietstock.vn/a.htm"],
            "published_at": ["2021-01-01T10:00:00+07:00"],
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    )
    news.to_parquet(news_path, index=False)
    manifest_path = news_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "output": {"sha256": file_fingerprint(news_path), "rows": 1},
                "checkpoint": {"directory_sha256": "checkpoint-v1"},
                "inference": {
                    "truncation_strategy": "head_tail",
                    "max_len": 256,
                    "batch_size": 32,
                    "runtime": {},
                },
                "input": {"fingerprint": "inputs-v1"},
            }
        ),
        encoding="utf-8",
    )
    prices = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "time": [pd.Timestamp("2021-01-01")],
            "close": [100.0],
        }
    )
    panel = pd.DataFrame({"has_news": [1]})
    captured = {}
    monkeypatch.setattr(cli_module, "_load_prices", lambda: prices)
    monkeypatch.setattr(forecasting_module, "assemble", lambda *_args: panel)

    def fake_run_experiment(panel, *, cfg, output_dir, provenance):
        captured["provenance"] = provenance
        return {
            "panel_rows": len(panel),
            "windows": [],
            "chance_level": 1 / 3,
            "summary": {},
            "ablation": {},
        }

    monkeypatch.setattr(experiment_module, "run_experiment", fake_run_experiment)
    assert (
        main(
            [
                "forecast",
                "--news-sentiment",
                str(news_path),
                "--output",
                str(tmp_path / "forecast"),
            ]
        )
        == 0
    )
    score_manifest = captured["provenance"]["news_sentiment"]["score_manifest"]
    assert score_manifest["manifest_sha256"] == file_fingerprint(manifest_path)
    assert score_manifest["checkpoint_directory_sha256"] == "checkpoint-v1"
    assert score_manifest["inference"]["truncation_strategy"] == "head_tail"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    news.loc[0, "prob_negative"] = 1.2
    news.to_parquet(news_path, index=False)
    manifest["output"]["sha256"] = file_fingerprint(news_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert main(["forecast", "--news-sentiment", str(news_path)]) == 2
    assert "invalid probability vectors" in capsys.readouterr().err

    news.loc[0, "prob_negative"] = 0.1
    news.loc[0, "url"] = "https://vietstock.vn/b.htm"
    news.to_parquet(news_path, index=False)
    assert main(["forecast", "--news-sentiment", str(news_path)]) == 2
    assert "parquet hash does not match manifest" in capsys.readouterr().err
