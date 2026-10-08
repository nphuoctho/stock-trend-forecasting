"""Shared test isolation."""

from __future__ import annotations

import pytest

from stf import config


@pytest.fixture(autouse=True)
def _isolated_repo_root(tmp_path, monkeypatch):
    """Point ``config.ROOT`` at a scratch dir.

    ``outputs/live`` (event log, news ledger, last_run.json) hangs off
    ``config.ROOT``; without this a test that runs ``score-news --incremental`` or
    ``forecast-predict`` would append events to the real checkout's live log.
    Tests that need a specific root override it with their own monkeypatch.
    """
    monkeypatch.setattr(config, "ROOT", tmp_path / "repo-root")
