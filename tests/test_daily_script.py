"""``scripts/daily-forecast.sh`` end to end with a stubbed ``uv``.

The real bash script runs against a scratch copy of the repository. Only what
needs the network or a PhoBERT checkpoint is stubbed: ``prices`` and ``news`` are
skipped, and ``score-news`` uses a fake sentiment model. Everything else is the
real CLI: ``forecast-predict`` fits nothing but loads real refit arms and writes
real parquet files, and the event log, ledger and ``last_run.json`` are the
production code paths.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stf import config, events
from stf.forecasting import assemble
from stf.forecasting.experiment import ForecastConfig
from stf.forecasting.serve import refit_arm

REPO = Path(__file__).resolve().parents[1]

_UV_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    # Minimal `uv run python ...` stand-in: same argv shape the script uses.
    set -euo pipefail
    [ "$1" = run ] && shift
    [ "$1" = python ] && shift
    if [ "$1" = -m ] && [ "$2" = stf.cli ] && { [ "$3" = prices ] || [ "$3" = news ]; }; then
      echo "[stub uv] skipped network step: $3"
      if [ "$3" = prices ] && [ -n "${STUB_HANG_FILE:-}" ]; then
        touch "$STUB_HANG_FILE"  # tell the test the job is mid-run, then stall
        if [ -n "${STUB_RELEASE_FILE:-}" ]; then
          while [ ! -e "$STUB_RELEASE_FILE" ]; do sleep 0.1; done
          exit 1  # the step fails once the test lets it go
        fi
        exec sleep 600
      fi
      exit 0
    fi
    exec env PYTHONPATH="$STUB_REPO/src" "$STUB_PYTHON" "$STUB_DRIVER" "$@"
    """
)

_DRIVER = textwrap.dedent(
    """\
    import json, os, sys
    from pathlib import Path

    from stf import config

    root = Path(os.environ["STUB_REPO"])
    assert Path(config.ROOT) == root, f"stf resolved ROOT={config.ROOT}, not {root}"
    args = sys.argv[1:]
    if args[:2] == ["-m", "stf.cli"]:
        if args[2] == "score-news":
            import numpy as np
            import pandas as pd
            from stf.data import news as news_module
            from stf.sentiment import model as model_module

            articles = pd.DataFrame(json.loads((root / "fake_articles.json").read_text()))
            news_module.load_ticker_articles = lambda: articles
            model_module.predict_proba = lambda texts, *_a, **_k: np.tile(
                [0.1, 0.2, 0.7], (len(texts), 1)
            )
        from stf.cli import main

        sys.exit(main(args[2:]))
    if args == ["-"]:
        exec(compile(sys.stdin.read(), "<stdin>", "exec"), {"__name__": "__main__"})
        sys.exit(0)
    sys.exit(f"stub uv: unsupported arguments {args}")
    """
)

SESSIONS = 90


def _prices(ticker: str, offset: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=SESSIONS)
    close = pd.Series(
        100.0
        + offset
        + np.cumsum(np.where((np.arange(SESSIONS) + offset) % 3 == 0, 1.5, -0.4)),
        dtype="float64",
    )
    return pd.DataFrame(
        {
            "ticker": ticker,
            "time": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000 + np.arange(SESSIONS) * 10_000,
        }
    )


def _article(ticker: str, name: str, published: str) -> dict:
    return {
        "ticker": ticker,
        "url": f"https://vietstock.vn/{name}.htm",
        "published_at": published,
        "title": f"Tin {name}",
        "body": "nội dung",
    }


class JobRepo:
    """A scratch repository the real daily script can run in."""

    def __init__(self, root: Path):
        self.root = root
        self.live = root / "outputs" / "live"
        self.articles: list[dict] = []

    def add_article(self, ticker: str, name: str, published: str) -> None:
        self.articles.append(_article(ticker, name, published))
        (self.root / "fake_articles.json").write_text(
            json.dumps(self.articles), encoding="utf-8"
        )

    def _env(self, arms: str) -> dict[str, str]:
        stub = self.root / "stub"
        return {
            **os.environ,
            "PATH": f"{stub / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "STUB_REPO": str(self.root),
            "STUB_PYTHON": sys.executable,
            "STUB_DRIVER": str(stub / "driver.py"),
            "SENTIMENT_MODEL": str(self.root / "models" / "sentiment" / "fake"),
            "ARMS": arms,
        }

    def run(self, arms: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(self.root / "scripts" / "daily-forecast.sh")],
            env=self._env(arms),
            capture_output=True,
            text=True,
            timeout=900,
        )

    def start_midway(self, arms: str, release: Path | None = None) -> subprocess.Popen:
        """Start the job and return once it is stalled inside the ``prices`` step.

        With ``release``, the step fails (exit 1) as soon as that file exists;
        otherwise it stalls until the job is killed.
        """
        marker = self.root / "midway"
        env = {**self._env(arms), "STUB_HANG_FILE": str(marker)}
        if release is not None:
            env["STUB_RELEASE_FILE"] = str(release)
        process = subprocess.Popen(
            ["bash", str(self.root / "scripts" / "daily-forecast.sh")],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group: bash, tee, sleep, uv stub
        )
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
        return process

    def start_and_kill_midway(self, arms: str) -> int:
        """SIGKILL the whole process group mid-run: no trap EXIT, no cleanup."""
        process = self.start_midway(arms)
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=30)

    def events(self) -> list[dict]:
        return events.EventLog(self.live).read_all()


@pytest.fixture()
def job_repo(tmp_path, monkeypatch):
    if shutil.which("bash") is None:
        pytest.skip("bash is required")
    root = tmp_path / "job-repo"
    shutil.copytree(
        REPO / "src" / "stf",
        root / "src" / "stf",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "daily-forecast.sh", root / "scripts")
    (root / "stub" / "bin").mkdir(parents=True)
    uv = root / "stub" / "bin" / "uv"
    uv.write_text(_UV_STUB, encoding="utf-8")
    uv.chmod(0o755)
    (root / "stub" / "driver.py").write_text(_DRIVER, encoding="utf-8")

    prices_dir = root / "data" / "raw" / "prices"
    prices_dir.mkdir(parents=True)
    prices = []
    for offset, ticker in enumerate(config.TICKERS):
        frame = _prices(ticker, offset)
        frame.to_parquet(prices_dir / f"{ticker}.parquet", index=False)
        prices.append(frame)
    prices_all = pd.concat(prices, ignore_index=True)

    repo = JobRepo(root)
    for n, ticker in enumerate(config.TICKERS):
        repo.add_article(ticker, f"seed-{n}", f"2024-03-{1 + n:02d}T10:00:00+07:00")

    sentiment_dir = root / "models" / "sentiment" / "fake"
    sentiment_dir.mkdir(parents=True)
    (sentiment_dir / "config.json").write_text("{}", encoding="utf-8")
    (sentiment_dir / "manifest.json").write_text(
        json.dumps({"selection": {"input_variant": "title_context"}}), encoding="utf-8"
    )

    # Leave the scored-news file (with its manifest) as a previous daily run would
    # have: the real score-news over the seed articles, outside the job.
    scored = root / "data" / "processed" / "news_sentiment_merged.parquet"
    seeded = subprocess.run(
        [
            sys.executable, str(root / "stub" / "driver.py"), "-m", "stf.cli",
            "score-news", "--model-dir", str(sentiment_dir),
            "--input-variant", "title_context", "--context-chars", "400",
            "--output", str(scored),
        ],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "STUB_REPO": str(root)},
        capture_output=True,
        text=True,
    )
    assert seeded.returncode == 0, seeded.stderr
    assert not repo.live.exists()  # seeding is not a job run: nothing is logged

    cfg = ForecastConfig(val_size=10, seeds=(42,))
    refit_arm(
        assemble(prices_all),
        "logreg_price",
        cfg=cfg,
        output_dir=root / "models" / "forecast" / "logreg_price",
    )
    refit_arm(
        assemble(prices_all, pd.read_parquet(scored)),
        "logreg_price_sentiment",
        cfg=cfg,
        output_dir=root / "models" / "forecast" / "logreg_price_sentiment",
    )
    broken = root / "models" / "forecast" / "tft_price"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{}", encoding="utf-8")  # unloadable arm

    # Two articles arrive after the seeded scoring; the job must score and announce them.
    repo.add_article("FPT", "new-1", "2024-04-01T10:00:00+07:00")
    repo.add_article("VNM", "new-2", "2024-04-02T10:00:00+07:00")
    return repo


def _types(logged: list[dict]) -> list[str]:
    return [e["type"] for e in logged]


def test_failing_later_arm_leaves_events_for_persisted_arms_and_a_failed_status(
    job_repo,
):
    result = job_repo.run("logreg_price logreg_price_sentiment tft_price")

    assert result.returncode == 1, result.stdout + result.stderr
    logged = job_repo.events()
    assert _types(logged) == [
        "news.ingested",
        "signal.issued",
        "signal.issued",
        "job.status",
    ]
    assert [e["id"] for e in logged] == [1, 2, 3, 4]

    news, first, second, status = logged
    assert news["data"]["count"] == 2
    assert news["data"]["by_ticker"] == {"FPT": 1, "VNM": 1}
    assert (first["data"]["arm"], second["data"]["arm"]) == (
        "logreg_price",
        "logreg_price_sentiment",
    )
    assert {first["data"]["n_predictions"], second["data"]["n_predictions"]} == {
        len(config.TICKERS)
    }
    assert not any(
        e["data"].get("arm") == "tft_price" for e in logged if e["type"] == "signal.issued"
    )
    assert not (job_repo.live / "tft_price").exists()

    assert status["data"]["ok"] is False
    assert status["data"]["exit_code"] == 1
    assert status["data"]["failed_step"] == "forecast-predict tft_price"
    on_disk = json.loads((job_repo.live / "last_run.json").read_text("utf-8"))
    assert on_disk["failed_step"] == "forecast-predict tft_price"
    assert on_disk["finished_at"] == status["data"]["finished_at"]


def test_rerun_after_losing_the_log_rebuilds_it_without_duplicates(job_repo):
    arms = "logreg_price logreg_price_sentiment"
    first = job_repo.run(arms)
    assert first.returncode == 0, first.stdout + first.stderr
    before = job_repo.events()
    assert _types(before) == [
        "news.ingested",
        "signal.issued",
        "signal.issued",
        "job.status",
    ]
    assert before[-1]["data"]["ok"] is True

    # Lose the index (or a crash that never wrote it): the artifacts survive.
    (job_repo.live / events.EVENTS_FILE).unlink()

    second = job_repo.run(arms)
    assert second.returncode == 0, second.stdout + second.stderr
    after = job_repo.events()

    # The script's opening reconcile re-derived every committed event from files;
    # the identical rerun then added nothing but its own job status.
    assert sorted(_types(after)) == [
        "job.status",
        "job.status",
        "news.ingested",
        "signal.issued",
        "signal.issued",
    ]
    assert {e["key"] for e in before if e["type"] != "job.status"} == {
        e["key"] for e in after if e["type"] != "job.status"
    }
    assert len({e["key"] for e in after}) == len(after)


def test_job_killed_midway_is_repaired_by_the_next_runs_opening_reconcile(job_repo):
    arms = "logreg_price logreg_price_sentiment"
    assert job_repo.run(arms).returncode == 0
    (job_repo.live / events.EVENTS_FILE).unlink()  # events never made it to the log

    # The next run dies by SIGKILL before doing any work: its EXIT trap never runs,
    # so only the reconcile at the very start of the script can have repaired the log.
    assert job_repo.start_and_kill_midway(arms) == -signal.SIGKILL

    repaired = job_repo.events()
    assert sorted(_types(repaired)) == [
        "job.status",
        "news.ingested",
        "signal.issued",
        "signal.issued",
    ]
    # Nothing was invented for the run that died: its status file is the old one.
    [status] = [e for e in repaired if e["type"] == "job.status"]
    on_disk = json.loads((job_repo.live / "last_run.json").read_text("utf-8"))
    assert status["data"]["finished_at"] == on_disk["finished_at"]


def test_exit_trap_reconciles_what_the_log_lost_while_the_job_was_running(job_repo):
    arms = "logreg_price logreg_price_sentiment"
    assert job_repo.run(arms).returncode == 0

    release = job_repo.root / "release"
    process = job_repo.start_midway(arms, release)  # opening reconcile already ran
    (job_repo.live / events.EVENTS_FILE).unlink()  # the log loses everything
    release.touch()  # the prices step now fails, so the job exits through its trap
    assert process.wait(timeout=120) == 1

    logged = job_repo.events()
    assert sorted(_types(logged)) == [
        "job.status",
        "news.ingested",
        "signal.issued",
        "signal.issued",
    ]
    [status] = [e for e in logged if e["type"] == "job.status"]
    assert status["data"]["ok"] is False
    assert status["data"]["failed_step"] == "prices"
    assert status["data"]["exit_code"] == 1
    # The trap logged the new status first; its reconcile then restored the rest.
    assert status["id"] == 1
