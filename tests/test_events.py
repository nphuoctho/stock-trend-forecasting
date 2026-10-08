"""Event log: idempotent emission, reconcile, and crash recovery.

The log under ``outputs/live`` is an index derived from the persisted artifacts,
so these tests assert on what a consumer of the log can observe: which events
exist after a rerun, a refused overwrite, or a SIGKILL between persisting a file
and appending its event.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stf import config, events
from stf.cli import main

ARM = "logreg_price"
SESSION = "2026-09-21"


def _live() -> Path:
    return events.live_dir()


def _logged() -> list[dict]:
    return events.EventLog(_live()).read_all()


def _frame(y_pred: str = "UP") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["FPT", "VNM"],
            "observation_date": pd.to_datetime([SESSION, SESSION]),
            "has_news": [0, 1],
            "prob_down": [0.2, 0.3],
            "prob_flat": [0.3, 0.3],
            "prob_up": [0.5, 0.4],
            "y_pred": [y_pred, "UP"],
            "arm": [ARM, ARM],
        }
    )


@pytest.fixture()
def predict(monkeypatch, tmp_path):
    """Run the real ``forecast-predict`` with the model and panel stubbed out."""
    from stf import cli as cli_module
    from stf import forecasting as forecasting_module
    from stf.forecasting import serve as serve_module

    model_dir = tmp_path / "arm"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
    arm = type("A", (), {"name": ARM, "use_sentiment": False})()
    current = {"frame": _frame()}
    monkeypatch.setattr(serve_module, "load_arm", lambda *_a, **_k: arm)
    monkeypatch.setattr(
        cli_module, "_load_prices", lambda: pd.DataFrame({"ticker": ["FPT"]})
    )
    monkeypatch.setattr(
        forecasting_module, "assemble", lambda *_a: pd.DataFrame({"x": [1]})
    )
    monkeypatch.setattr(
        serve_module, "predict_latest", lambda *_a, **_k: current["frame"].copy()
    )

    class Runner:
        output_dir = _live() / ARM
        dated = output_dir / f"predictions_{SESSION}.parquet"

        def run(self, frame: pd.DataFrame | None = None) -> int:
            if frame is not None:
                current["frame"] = frame
            return main(
                [
                    "forecast-predict",
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(self.output_dir),
                ]
            )

    return Runner()


def test_first_prediction_logs_one_signal_with_its_row_count(predict):
    assert predict.run() == 0

    [event] = _logged()
    assert event["id"] == 1
    assert event["type"] == "signal.issued"
    assert event["data"]["arm"] == ARM
    assert event["data"]["observation_date"] == SESSION
    assert event["data"]["n_predictions"] == 2
    assert event["data"]["issued_at"] == str(
        pd.read_parquet(predict.dated)["issued_at"].iloc[0]
    )
    assert event["key"].startswith(f"signal.issued:{ARM}:{SESSION}:")


def test_identical_rerun_in_the_same_day_adds_no_events(predict):
    assert predict.run() == 0
    before = _logged()

    assert predict.run() == 0
    assert predict.run() == 0

    assert _logged() == before


def test_refused_overwrite_keeps_the_file_and_adds_no_signal(predict, capsys):
    assert predict.run() == 0
    [issued] = _logged()
    kept = predict.dated.read_bytes()

    assert predict.run(_frame(y_pred="DOWN")) == 3
    assert "refusing to overwrite" in capsys.readouterr().err

    assert predict.dated.read_bytes() == kept
    assert _logged() == [issued]
    # Reconcile sees the kept file, whose event is already logged: still nothing new.
    assert events.reconcile() == []
    assert _logged() == [issued]


def test_prediction_outside_the_live_directory_is_not_indexed(
    predict, tmp_path
):
    predict.output_dir = tmp_path / "adhoc" / ARM
    assert predict.run() == 0
    assert (predict.output_dir / f"predictions_{SESSION}.parquet").is_file()
    assert _logged() == []


def test_reissued_file_with_new_content_gets_its_own_signal(predict):
    """Deleting an invalid issuance and reissuing is a new, distinct signal."""
    assert predict.run() == 0
    predict.dated.unlink()

    assert predict.run(_frame(y_pred="DOWN")) == 0

    first, second = _logged()
    assert first["key"] != second["key"]
    assert (first["data"]["arm"], second["data"]["arm"]) == (ARM, ARM)
    assert second["id"] == first["id"] + 1


# --- append semantics ------------------------------------------------------------


def test_append_is_idempotent_on_key_and_ids_are_sequential(tmp_path):
    log = events.EventLog(tmp_path)
    a = log.append("job.status", "k1", {"n": 1})
    b = log.append("job.status", "k2", {"n": 2})
    again = log.append("job.status", "k1", {"n": 99})

    assert (a["id"], b["id"]) == (1, 2)
    assert again == a
    assert [e["key"] for e in log.read_all()] == ["k1", "k2"]


def test_a_torn_final_line_is_dropped_before_the_next_append(tmp_path):
    log = events.EventLog(tmp_path)
    log.append("job.status", "k1", {})
    with open(log.path, "ab") as handle:
        handle.write(b'{"id": 2, "type": "job.sta')  # power loss mid-write

    assert [e["id"] for e in log.read_all()] == [1]
    new = log.append("job.status", "k2", {})

    assert new["id"] == 2
    assert [e["key"] for e in log.read_all()] == ["k1", "k2"]
    assert log.path.read_bytes().endswith(b"\n")


def test_concurrent_processes_never_share_an_id_or_duplicate_a_key(tmp_path):
    worker = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from stf.events import EventLog

        log = EventLog(Path(sys.argv[1]))
        for n in range(40):
            log.append("job.status", f"key-{n}", {"writer": sys.argv[2]})
        """
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", worker, str(tmp_path), str(i)])
        for i in range(4)
    ]
    assert [p.wait(timeout=60) for p in procs] == [0, 0, 0, 0]

    logged = events.EventLog(tmp_path).read_all()
    assert sorted(e["id"] for e in logged) == list(range(1, 41))
    assert sorted(e["key"] for e in logged) == sorted(f"key-{n}" for n in range(40))


# --- reconcile --------------------------------------------------------------------


def _write_prediction_file(arm: str, session: str, rows: int = 2) -> Path:
    path = _live() / arm / f"predictions_{session}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(rows)],
            "observation_date": pd.to_datetime([session] * rows),
            "issued_at": ["2026-01-01T00:00:00+00:00"] * rows,
        }
    ).to_parquet(path, index=False)
    return path


def test_reconcile_rebuilds_a_missing_log_from_artifacts_once():
    _write_prediction_file("lstm_price", "2026-09-22")
    _write_prediction_file("lstm_price", "2026-09-21")
    _write_prediction_file("tft_price", "2026-09-21", rows=3)

    first = events.reconcile()
    assert [(e["data"]["observation_date"], e["data"]["arm"]) for e in first] == [
        ("2026-09-21", "lstm_price"),
        ("2026-09-21", "tft_price"),
        ("2026-09-22", "lstm_price"),
    ]
    assert [e["id"] for e in first] == [1, 2, 3]
    assert first[1]["data"]["n_predictions"] == 3

    assert events.reconcile() == []
    assert events.reconcile() == []
    assert len(_logged()) == 3


def test_reconcile_invents_nothing_from_partial_or_unreadable_files():
    arm_dir = _live() / "lstm_price"
    arm_dir.mkdir(parents=True)
    (arm_dir / ".predictions_2026-09-21.parquet.123.tmp").write_bytes(b"PAR1 partial")
    (arm_dir / "latest.parquet").write_bytes(b"not a dated file")
    (arm_dir / "predictions_2026-09-21.parquet").write_bytes(b"truncated garbage")
    (_live() / "last_run.json").write_text("{half written", encoding="utf-8")

    assert events.reconcile() == []
    assert _logged() == []


def test_reconcile_is_a_noop_without_a_live_directory():
    assert not _live().exists()
    assert events.reconcile() == []
    assert not _live().exists()


def test_job_status_is_logged_with_the_failed_step_and_rederived_from_the_file():
    events.record_last_run(exit_code=2, failed_step="forecast-predict tft_price")

    [event] = _logged()
    assert event["type"] == "job.status"
    assert event["data"]["ok"] is False
    assert event["data"]["exit_code"] == 2
    assert event["data"]["failed_step"] == "forecast-predict tft_price"
    on_disk = json.loads((_live() / "last_run.json").read_text(encoding="utf-8"))
    assert on_disk["finished_at"] == event["data"]["finished_at"]
    assert event["key"] == f"job.status:{on_disk['finished_at']}"

    # Lose the log: the status file alone yields exactly the same event key.
    events.EventLog(_live()).path.unlink()
    [rebuilt] = events.reconcile()
    assert rebuilt["key"] == event["key"]
    assert events.reconcile() == []


def test_successful_job_status_has_no_failed_step():
    events.record_last_run(exit_code=0, failed_step="ignored on success")

    [event] = _logged()
    assert event["data"] == {
        "ok": True,
        "exit_code": 0,
        "failed_step": None,
        "finished_at": event["data"]["finished_at"],
    }


def test_events_reconcile_command_reports_what_it_appended(capsys):
    _write_prediction_file("lstm_price", "2026-09-21")

    assert main(["events-reconcile"]) == 0
    assert "appended 1 event(s)" in capsys.readouterr().out
    assert main(["events-reconcile"]) == 0
    assert "appended 0 event(s)" in capsys.readouterr().out


# --- score-news -------------------------------------------------------------------


@pytest.fixture()
def score(monkeypatch, tmp_path):
    """Run the real ``score-news`` over a growing article list with a fake model."""
    from stf.data import news as news_module
    from stf.sentiment import model as model_module

    articles: list[dict] = []

    def load():
        return pd.DataFrame(
            articles,
            columns=["ticker", "url", "published_at", "title", "body"],
        )

    def fake_predict_proba(texts, model_dir, **_kwargs):
        return np.tile([0.1, 0.2, 0.7], (len(texts), 1))

    monkeypatch.setattr(news_module, "load_ticker_articles", load)
    monkeypatch.setattr(model_module, "predict_proba", fake_predict_proba)
    model_dir = tmp_path / "fake-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "manifest.json").write_text(
        json.dumps({"selection": {"input_variant": "title"}}), encoding="utf-8"
    )

    class Runner:
        output = tmp_path / "scored.parquet"

        def add(self, ticker: str, name: str) -> None:
            articles.append(
                {
                    "ticker": ticker,
                    "url": f"https://vietstock.vn/{name}.htm",
                    "published_at": "2021-01-01T10:00:00+07:00",
                    "title": f"Tin {name}",
                    "body": "nội dung",
                }
            )

        def run(self, *flags: str) -> int:
            return main(
                ["score-news", "--model-dir", str(model_dir), "--output",
                 str(self.output), *flags]
            )

    return Runner()


def test_each_incremental_batch_logs_one_news_event_and_empty_batches_none(score):
    score.add("FPT", "a")
    score.add("VNM", "b")
    assert score.run("--incremental") == 0
    [first] = _logged()
    assert first["type"] == "news.ingested"
    assert first["data"] == {
        "count": 2,
        "total_scored": 2,
        "by_ticker": {"FPT": 1, "VNM": 1},
    }

    assert score.run("--incremental") == 0  # nothing new was scored
    assert _logged() == [first]

    score.add("FPT", "c")
    assert score.run("--incremental") == 0
    second = _logged()[1]
    assert second["data"] == {"count": 1, "total_scored": 3, "by_ticker": {"FPT": 1}}
    assert second["key"] != first["key"]
    assert events.reconcile() == []


def test_a_full_rescore_is_not_a_news_event(score):
    score.add("FPT", "a")
    assert score.run() == 0
    assert _logged() == []


# --- crash recovery ---------------------------------------------------------------

_CRASH_CHILD = textwrap.dedent(
    """
    import json, os, signal, sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from stf import config, events
    from stf import cli as cli_module

    kind, root, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    config.ROOT = root


    def die(*_args, **_kwargs):
        # Whole process group, like a power cut: no `finally`, no trap, no atexit.
        os.killpg(os.getpgid(0), signal.SIGKILL)


    if kind == "predict":
        from stf import forecasting as forecasting_module
        from stf.forecasting import serve as serve_module

        arm = type("A", (), {"name": "logreg_price", "use_sentiment": False})()
        serve_module.load_arm = lambda *_a, **_k: arm
        cli_module._load_prices = lambda: pd.DataFrame({"ticker": ["FPT"]})
        forecasting_module.assemble = lambda *_a: pd.DataFrame({"x": [1]})
        frame = pd.DataFrame(
            {
                "ticker": ["FPT", "VNM"],
                "observation_date": pd.to_datetime(["2026-09-21"] * 2),
                "has_news": [0, 1],
                "prob_down": [0.2, 0.3],
                "prob_flat": [0.3, 0.3],
                "prob_up": [0.5, 0.4],
                "y_pred": ["UP", "UP"],
                "arm": ["logreg_price"] * 2,
            }
        )
        serve_module.predict_latest = lambda *_a, **_k: frame.copy()
        argv = [
            "forecast-predict",
            "--model-dir", str(work / "arm"),
            "--output-dir", str(root / "outputs" / "live" / "logreg_price"),
        ]
        events.append_event = die  # persisted, then killed before the append
    else:
        from stf.data import news as news_module
        from stf.sentiment import model as model_module

        articles = pd.DataFrame(
            {
                "ticker": ["FPT", "VNM"],
                "url": ["https://vietstock.vn/a.htm", "https://vietstock.vn/b.htm"],
                "published_at": ["2021-01-01T10:00:00+07:00"] * 2,
                "title": ["Tin A", "Tin B"],
                "body": ["a", "b"],
            }
        )
        news_module.load_ticker_articles = lambda: articles
        model_module.predict_proba = lambda texts, *_a, **_k: np.tile(
            [0.1, 0.2, 0.7], (len(texts), 1)
        )
        argv = [
            "score-news",
            "--model-dir", str(work / "fake-model"),
            "--incremental",
            "--output", str(work / "scored.parquet"),
        ]
        if kind == "score-committed":
            events.append_event = die  # parquet committed, event never appended
        else:  # "score-uncommitted"
            os.replace = die  # ledger intent written, rename never happens
    sys.exit(cli_module.main(argv))
    """
)


def _crash(kind: str, tmp_path: Path) -> subprocess.CompletedProcess:
    work = tmp_path / "work"
    (work / "arm").mkdir(parents=True, exist_ok=True)
    (work / "arm" / "manifest.json").write_text("{}", encoding="utf-8")
    (work / "fake-model").mkdir(exist_ok=True)
    (work / "fake-model" / "config.json").write_text("{}", encoding="utf-8")
    (work / "fake-model" / "manifest.json").write_text(
        json.dumps({"selection": {"input_variant": "title"}}), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, "-c", _CRASH_CHILD, kind, str(config.ROOT), str(work)],
        capture_output=True,
        text=True,
        timeout=300,
        start_new_session=True,  # its own process group: killpg cannot reach pytest
    )


def test_sigkill_between_persisting_a_signal_and_logging_it_recovers_exactly_once(
    tmp_path,
):
    result = _crash("predict", tmp_path)

    assert result.returncode == -signal.SIGKILL, result.stderr
    dated = _live() / ARM / f"predictions_{SESSION}.parquet"
    assert dated.is_file() and (_live() / ARM / "latest.parquet").is_file()
    assert _logged() == []  # the crash really did skip the append

    [recovered] = events.reconcile()
    assert recovered["type"] == "signal.issued"
    assert recovered["data"]["n_predictions"] == 2
    assert events.reconcile() == []
    assert events.reconcile() == []
    assert [e["id"] for e in _logged()] == [1]


def test_sigkill_between_committing_news_and_logging_it_recovers_exactly_once(
    tmp_path,
):
    result = _crash("score-committed", tmp_path)

    assert result.returncode == -signal.SIGKILL, result.stderr
    assert len(pd.read_parquet(tmp_path / "work" / "scored.parquet")) == 2
    assert _logged() == []

    [recovered] = events.reconcile()
    assert recovered["type"] == "news.ingested"
    assert recovered["data"]["count"] == 2
    assert events.reconcile() == []
    assert len(_logged()) == 1


def test_sigkill_before_the_news_rename_invents_no_event(tmp_path, score):
    result = _crash("score-uncommitted", tmp_path)

    assert result.returncode == -signal.SIGKILL, result.stderr
    assert not (tmp_path / "work" / "scored.parquet").exists()
    assert (_live() / events.NEWS_LEDGER_FILE).is_file()  # the intent is on disk
    assert events.reconcile() == []
    assert _logged() == []

    # The rerun scores the same batch; the dead intent shares its key, so the
    # batch is still announced exactly once.
    score.output = tmp_path / "work" / "scored.parquet"
    score.add("FPT", "a")
    score.add("VNM", "b")
    assert score.run("--incremental") == 0
    assert [e["type"] for e in _logged()] == ["news.ingested"]
    assert events.reconcile() == []
    assert len(_logged()) == 1


def test_api_startup_recovers_the_unlogged_signal_without_duplicates(tmp_path):
    from stf.webapp import app as webapp_module

    assert _crash("predict", tmp_path).returncode == -signal.SIGKILL
    assert _logged() == []

    with TestClient(webapp_module.app):  # runs the lifespan: startup reconcile
        recovered = _logged()
    assert [e["type"] for e in recovered] == ["signal.issued"]

    with TestClient(webapp_module.app):  # a second start adds nothing
        pass
    assert _logged() == recovered
    assert events.reconcile() == []

