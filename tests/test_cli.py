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

    monkeypatch.setattr(
        model_module,
        "predict_proba",
        lambda *_args, **_kwargs: np.array(
            [[0.5000011, 0.5, 0.0], [0.5000011, 0.5, 0.0]]
        ),
    )
    over_tolerance_output = tmp_path / "over-tolerance.parquet"
    with pytest.raises(RuntimeError, match="invalid probability vectors"):
        main(
            [
                "score-news",
                "--model-dir",
                str(model_dir),
                "--output",
                str(over_tolerance_output),
            ]
        )
    assert not invalid_output.exists()
    assert not over_tolerance_output.exists()


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
    panel = pd.DataFrame(
        {
            "has_news": [1],
            "observation_date": [pd.Timestamp("2021-01-01")],
            "target_date": [pd.Timestamp("2021-01-04")],
        }
    )
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

    manifest_path.write_text("[]", encoding="utf-8")
    assert main(["forecast", "--news-sentiment", str(news_path)]) == 2
    assert "invalid score-news manifest" in capsys.readouterr().err


def _point_in_time_verdict(monkeypatch, tmp_path, *, label_cutoff, tamper=False):
    """Run `forecast` against a checkpoint with `label_cutoff` and return its verdict."""
    from stf import cli as cli_module
    from stf import forecasting as forecasting_module
    from stf.forecasting import experiment as experiment_module
    from stf.sentiment.dataset import file_fingerprint

    checkpoint_manifest = tmp_path / "checkpoint-manifest.json"
    checkpoint_manifest.write_text(
        json.dumps({"provenance": {"label_cutoff": label_cutoff}}), encoding="utf-8"
    )
    news_path = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "ticker": ["FPT"],
            "url": ["https://vietstock.vn/a.htm"],
            "published_at": ["2021-01-01T10:00:00+07:00"],
            "prob_negative": [0.1],
            "prob_neutral": [0.2],
            "prob_positive": [0.7],
        }
    ).to_parquet(news_path, index=False)
    recorded_hash = file_fingerprint(checkpoint_manifest)
    if tamper:
        checkpoint_manifest.write_text(
            json.dumps({"provenance": {"label_cutoff": "1999-01-01"}}), encoding="utf-8"
        )
    news_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "output": {"sha256": file_fingerprint(news_path), "rows": 1},
                "checkpoint": {
                    "directory_sha256": "checkpoint-v1",
                    "manifest_path": str(checkpoint_manifest),
                    "manifest_sha256": recorded_hash,
                },
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

    panel = pd.DataFrame(
        {
            "has_news": [1, 1],
            "observation_date": pd.to_datetime(["2024-10-20", "2024-10-21"]),
            "target_date": pd.to_datetime(["2024-10-21", "2024-10-22"]),
        }
    )
    captured = {}
    monkeypatch.setattr(
        cli_module,
        "_load_prices",
        lambda: pd.DataFrame(
            {"ticker": ["FPT"], "time": [pd.Timestamp("2024-10-21")], "close": [100.0]}
        ),
    )
    monkeypatch.setattr(forecasting_module, "assemble", lambda *_args: panel)
    monkeypatch.setattr(
        experiment_module,
        "first_test_observation_date",
        lambda *_a, **_k: "2024-10-21",
    )

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
    return captured["provenance"]["news_sentiment"]


@pytest.mark.parametrize(
    "label_cutoff, point_in_time",
    [
        ("2024-10-20", True),  # a day before the first test observation
        ("2024-10-21", True),  # exactly on it: the filter is strictly-before, so valid
        ("2024-10-22", False),  # a day after: the checkpoint saw a test-period label
    ],
)
def test_point_in_time_verdict_at_the_cutoff_boundary(
    monkeypatch, tmp_path, label_cutoff, point_in_time
):
    """Equality is the tight admissible case, and one day past it is not.

    `sentiment-refit --before-date` keeps labels strictly before the cutoff, so a
    cutoff equal to the first test observation proves no test-period label was seen.
    An off-by-one here either voids a valid out-of-sample claim or, worse, endorses
    a leaked one.
    """
    verdict = _point_in_time_verdict(monkeypatch, tmp_path, label_cutoff=label_cutoff)

    assert verdict["first_test_observation_date"] == "2024-10-21"
    assert verdict["point_in_time"] is point_in_time
    assert verdict["point_in_time_reason"] == (
        "verified" if point_in_time else "cutoff_after_test_start"
    )


def test_point_in_time_refuses_a_checkpoint_manifest_edited_after_scoring(
    monkeypatch, tmp_path
):
    """A cutoff is only trusted when the manifest still hashes to what scoring saw."""
    verdict = _point_in_time_verdict(
        monkeypatch, tmp_path, label_cutoff="2024-10-21", tamper=True
    )

    assert verdict["point_in_time"] is False
    assert verdict["point_in_time_reason"] == "hash_mismatch"
    assert verdict["label_cutoff"] is None


def test_sentiment_refit_rejects_a_before_date_with_a_time_component(
    tmp_path, capsys
):
    """--before-date must be a calendar date, not a timestamp.

    The cutoff is recorded in the manifest as a date and verified against the
    first test observation date. Accepting '2024-10-21T15:00' would silently
    record a different boundary than the one the operator asked for.
    """
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "text": ["tin tốt", "tin xấu"],
            "label": ["POSITIVE", "NEGATIVE"],
            "published_at": [
                "2024-01-01T10:00:00+07:00",
                "2024-01-02T10:00:00+07:00",
            ],
        }
    ).to_csv(labels, index=False)

    assert (
        main(
            [
                "sentiment-refit",
                "--data",
                str(labels),
                "--cv-results",
                str(tmp_path / "cv.json"),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--seed",
                "42",
                "--input-variant",
                "title",
                "--truncation-strategy",
                "head_tail",
                "--class-weighting",
                "none",
                "--output",
                str(tmp_path / "out"),
                "--before-date",
                "2024-10-21T15:00",
            ]
        )
        == 2
    )
    assert "must be a calendar date" in capsys.readouterr().err


def test_forecast_predict_stamps_issuance_and_refuses_to_rewrite(
    monkeypatch, tmp_path, capsys
):
    """Issued predictions are an audit trail, not a cache.

    Every dated parquet carries when it was issued and which checkpoint produced
    it; a re-run that would change the content must stop instead of silently
    rewriting what was already published.
    """
    from stf import cli as cli_module
    from stf import forecasting as forecasting_module
    from stf.forecasting import serve as serve_module

    model_dir = tmp_path / "arm"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")

    arm = type("A", (), {"name": "logreg_price", "use_sentiment": False})()
    monkeypatch.setattr(serve_module, "load_arm", lambda *_a, **_k: arm)
    monkeypatch.setattr(
        cli_module, "_load_prices", lambda: pd.DataFrame({"ticker": ["FPT"]})
    )
    monkeypatch.setattr(
        forecasting_module, "assemble", lambda *_a: pd.DataFrame({"x": [1]})
    )

    frame = pd.DataFrame(
        {
            "ticker": ["FPT"],
            "observation_date": pd.to_datetime(["2026-09-21"]),
            "has_news": [0],
            "prob_down": [0.2],
            "prob_flat": [0.3],
            "prob_up": [0.5],
            "y_pred": ["UP"],
            "arm": ["logreg_price"],
        }
    )
    monkeypatch.setattr(
        serve_module, "predict_latest", lambda *_a, **_k: frame.copy()
    )

    out_dir = tmp_path / "live"
    args = [
        "forecast-predict",
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(out_dir),
    ]
    assert main(args) == 0
    written = pd.read_parquet(out_dir / "predictions_2026-09-21.parquet")
    assert written["issued_at"].notna().all()
    assert written["checkpoint_manifest_sha256"].notna().all()

    # Same content re-run: idempotent, keeps the original issuance stamp.
    assert main(args) == 0
    reread = pd.read_parquet(out_dir / "predictions_2026-09-21.parquet")
    assert reread["issued_at"].equals(written["issued_at"])

    # Different content for the same session: refused, file untouched.
    changed = frame.copy()
    changed["y_pred"] = ["DOWN"]
    monkeypatch.setattr(
        serve_module, "predict_latest", lambda *_a, **_k: changed.copy()
    )
    assert main(args) == 2
    assert "refusing to overwrite" in capsys.readouterr().err
    after = pd.read_parquet(out_dir / "predictions_2026-09-21.parquet")
    assert after["y_pred"].tolist() == ["UP"]


def test_in_window_alignment_report_uses_a_half_open_final_day(monkeypatch, tmp_path):
    """The window covers all of its last day and none of the next one.

    An article published after the 15:00 cutoff on the final day still belongs to
    the study window even though it anchors to the next session; one published at
    midnight the following day does not. An inclusive upper bound admits that
    midnight row, and a 23:59:59 bound drops timestamps carrying fractional
    seconds in the last second.
    """
    from stf import cli as cli_module
    from stf import forecasting as forecasting_module
    from stf.forecasting import experiment as experiment_module
    from stf.sentiment.dataset import file_fingerprint

    published = [
        "2019-12-31T23:59:59.500000+07:00",  # before the window
        "2020-01-01T09:00:00+07:00",  # first instant of the window
        "2020-12-31T16:30:00+07:00",  # after the cutoff on the last day: in window
        "2020-12-31T23:59:59.500000+07:00",  # fractional second in the last second
        "2021-01-01T00:00:00+07:00",  # first instant after the window
    ]
    news_path = tmp_path / "scored.parquet"
    pd.DataFrame(
        {
            "ticker": ["FPT"] * len(published),
            "url": [f"https://vietstock.vn/{i}.htm" for i in range(len(published))],
            "published_at": published,
            "prob_negative": [0.1] * len(published),
            "prob_neutral": [0.2] * len(published),
            "prob_positive": [0.7] * len(published),
        }
    ).to_parquet(news_path, index=False)
    news_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "output": {"sha256": file_fingerprint(news_path), "rows": len(published)},
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

    sessions = pd.bdate_range("2019-12-02", "2021-01-29")
    prices = pd.DataFrame(
        {
            "ticker": "FPT",
            "time": sessions,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + np.arange(len(sessions), dtype=float),
            "volume": 1_000_000,
        }
    )
    captured = {}
    monkeypatch.setattr(cli_module, "_load_prices", lambda: prices)
    monkeypatch.setattr(
        forecasting_module,
        "assemble",
        lambda *_a: pd.DataFrame(
            {
                "has_news": [1],
                "observation_date": pd.to_datetime(["2020-06-01"]),
                "target_date": pd.to_datetime(["2020-06-02"]),
            }
        ),
    )
    monkeypatch.setattr(
        experiment_module,
        "run_experiment",
        lambda panel, *, cfg, output_dir, provenance: captured.update(
            provenance=provenance
        )
        or {
            "panel_rows": len(panel),
            "windows": [],
            "chance_level": 1 / 3,
            "summary": {},
            "ablation": {},
        },
    )

    assert (
        main(
            [
                "forecast",
                "--news-sentiment",
                str(news_path),
                "--panel-end",
                "2020-12-31",
                "--output",
                str(tmp_path / "forecast"),
            ]
        )
        == 0
    )

    report = captured["provenance"]["alignment_report_in_window"]
    assert report["window"] == ["2020-01-01", "2020-12-31"]
    assert report["links"] == 3  # the two boundary rows outside are excluded
    # Both 2020-12-31 articles are published after the 15:00 cutoff, so they roll
    # onto the first 2021 session: published in window, unusable by a 2020 panel.
    assert report["anchored_after_window"] == 2
