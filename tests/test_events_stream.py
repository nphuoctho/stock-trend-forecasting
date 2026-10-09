"""``GET /api/events`` against a real uvicorn process.

Starlette's TestClient buffers a response until the app finishes, so an endless
event stream can only be observed through a socket. Each server runs in its own
process with ``config.ROOT`` pointed at a scratch directory, and the tests append
to the log from the test process, exactly as the daily job would.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from stf import events
from stf.webapp.app import app, create_app

_LAUNCHER = textwrap.dedent(
    """
    import os
    from pathlib import Path

    import fastapi.routing
    import uvicorn

    from stf import config

    config.ROOT = Path(os.environ["STF_TEST_ROOT"])
    fastapi.routing._PING_INTERVAL = float(os.environ["STF_TEST_PING"])
    from stf.webapp import events_api

    events_api.POLL_SECONDS = 0.05
    from stf.webapp.app import app

    uvicorn.run(
        app, host="127.0.0.1", port=int(os.environ["STF_TEST_PORT"]), log_level="warning"
    )
    """
)


class Server:
    def __init__(self, root: Path, env: dict[str, str]):
        self.root = root
        self.live = root / "outputs" / "live"
        self.log = events.EventLog(self.live)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        self.process = subprocess.Popen(
            [sys.executable, "-c", _LAUNCHER],
            env={
                **os.environ,
                "STF_TEST_ROOT": str(root),
                "STF_TEST_PORT": str(self.port),
                "STF_TEST_PING": "0.3",
                # The per-IP limits belong to the hardening middleware and are
                # exercised on purpose below; everywhere else they must not
                # interfere, since every client here is 127.0.0.1.
                "STF_RATE_LIMIT": "0",
                "STF_EVENTS_CONNECT_PER_MINUTE": "100000",
                "STF_TRUST_CF_HEADERS": "0",
                "STF_EVENTS_MAX_PER_IP": "1000",
                **env,
            },
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 60
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"server died: {self.process.stderr.read()}")
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/events"

    def append(self, key: str, **data) -> dict:
        return self.log.append("job.status", key, {"ok": True, **data})

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)
        self.process.stderr.close()


class Stream:
    """A client connection whose frames are parsed on a background thread."""

    def __init__(self, url: str, headers: dict[str, str] | None = None):
        self.frames: queue.Queue = queue.Queue()
        self._client = httpx.Client(timeout=httpx.Timeout(15.0, read=None))
        self._context = self._client.stream("GET", url, headers=headers or {})
        self.response = self._context.__enter__()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        frame: dict[str, str] = {}
        try:
            for line in self.response.iter_lines():
                if line == "":
                    if frame:
                        self.frames.put(frame)
                    frame = {}
                elif line.startswith(":"):
                    self.frames.put({"comment": line[1:].strip()})
                else:
                    name, _, value = line.partition(":")
                    frame[name] = value.removeprefix(" ")
        except (httpx.HTTPError, OSError):
            pass
        finally:
            self.frames.put(None)  # end of stream

    def next_frame(self, timeout: float = 10.0) -> dict | None:
        return self.frames.get(timeout=timeout)

    def events(self, count: int, timeout: float = 10.0) -> list[dict]:
        """The next ``count`` event frames, skipping retry hints and comments."""
        found: list[dict] = []
        deadline = time.monotonic() + timeout
        while len(found) < count:
            frame = self.next_frame(max(0.05, deadline - time.monotonic()))
            assert frame is not None, f"stream ended after {found}"
            if "event" in frame:
                found.append(frame)
        return found

    def assert_quiet(self, seconds: float = 0.6) -> None:
        """No event frame arrives within ``seconds`` (comments are allowed)."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                frame = self.frames.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                return
            assert frame is None or "event" not in frame, f"unexpected {frame}"

    def close(self) -> None:
        self._context.__exit__(None, None, None)
        self._client.close()


@pytest.fixture()
def start_server(tmp_path):
    servers: list[Server] = []

    def start(**env: str) -> Server:
        server = Server(tmp_path / f"server-{len(servers)}", env)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


def _ids(frames: list[dict]) -> list[int]:
    return [int(frame["id"]) for frame in frames]


def test_stream_framing_headers_and_live_tail(start_server):
    server = start_server()
    server.append("before-connect")  # a new client is not replayed history

    stream = Stream(server.url)
    try:
        assert stream.response.status_code == 200
        assert stream.response.headers["content-type"].startswith("text/event-stream")
        assert stream.response.headers["cache-control"] == "no-cache"
        assert stream.response.headers["x-accel-buffering"] == "no"

        live = server.append("while-connected", step="x")
        [frame] = stream.events(1)
        assert frame["event"] == "job.status"
        assert frame["id"] == str(live["id"]) == "2"
        assert json.loads(frame["data"]) == live
    finally:
        stream.close()


def test_idle_stream_sends_heartbeat_comments_that_are_never_logged(start_server):
    server = start_server()
    stream = Stream(server.url)
    try:
        comments = []
        while len(comments) < 2:
            frame = stream.next_frame(timeout=5)
            assert frame is not None
            if "comment" in frame:
                comments.append(frame["comment"])
        assert comments == ["ping", "ping"]
        assert server.log.read_all() == []
    finally:
        stream.close()


def test_reconnect_with_last_event_id_receives_exactly_the_missed_events(start_server):
    server = start_server()
    first = Stream(server.url)
    try:
        for n in range(3):
            server.append(f"seen-{n}")
        seen = first.events(3)
        assert _ids(seen) == [1, 2, 3]
    finally:
        first.close()  # client drops mid-stream

    missed = [server.append(f"missed-{n}") for n in range(2)]

    second = Stream(server.url, {"Last-Event-ID": seen[-1]["id"]})
    try:
        replayed = second.events(2)
        assert _ids(replayed) == [m["id"] for m in missed] == [4, 5]
        second.assert_quiet()  # nothing twice, nothing extra

        server.append("after-reconnect")
        assert _ids(second.events(1)) == [6]
    finally:
        second.close()


def test_last_event_id_query_parameter_serves_clients_without_headers(start_server):
    server = start_server()
    for n in range(3):
        server.append(f"e{n}")

    stream = Stream(f"{server.url}?last_event_id=1")
    try:
        assert _ids(stream.events(2)) == [2, 3]
    finally:
        stream.close()

    both = Stream(f"{server.url}?last_event_id=1", {"Last-Event-ID": "2"})
    try:
        assert _ids(both.events(1)) == [3]  # the header wins
        both.assert_quiet()
    finally:
        both.close()


def test_caught_up_client_gets_nothing_replayed(start_server):
    server = start_server()
    server.append("only")

    stream = Stream(server.url, {"Last-Event-ID": "1"})
    try:
        stream.assert_quiet()
    finally:
        stream.close()


def _reset_after_connecting(stream: Stream) -> dict:
    [frame] = stream.events(1)
    assert frame["event"] == "reset"
    return frame


def test_id_from_the_future_yields_reset_then_tails_from_the_head(start_server):
    server = start_server()  # e.g. the log was deleted and rebuilt: ids restarted
    for n in range(2):
        server.append(f"e{n}")

    stream = Stream(server.url, {"Last-Event-ID": "500"})
    try:
        reset = _reset_after_connecting(stream)
        assert reset["id"] == "2"
        assert json.loads(reset["data"]) == {
            "id": 2,
            "type": "reset",
            "data": {"reason": "stale_last_event_id", "head_id": 2},
        }
        server.append("after-reset")
        assert _ids(stream.events(1)) == [3]
    finally:
        stream.close()


def test_id_older_than_the_oldest_retained_event_yields_reset(tmp_path, start_server):
    # An operator trimmed the head of the log: ids 1..9 can no longer be served.
    live = tmp_path / "server-0" / "outputs" / "live"
    live.mkdir(parents=True)
    lines = [
        {"id": n, "ts": "2026-01-01T00:00:00+00:00", "type": "job.status",
         "key": f"k{n}", "data": {}}
        for n in (10, 11)
    ]
    (live / events.EVENTS_FILE).write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    server = start_server()

    stale = Stream(server.url, {"Last-Event-ID": "3"})
    try:
        reset = _reset_after_connecting(stale)
        assert json.loads(reset["data"])["data"]["reason"] == "stale_last_event_id"
    finally:
        stale.close()

    edge = Stream(server.url, {"Last-Event-ID": "9"})  # first id - 1: fully servable
    try:
        assert _ids(edge.events(2)) == [10, 11]
    finally:
        edge.close()


def test_unparseable_last_event_id_yields_reset(start_server):
    server = start_server()
    server.append("e")

    stream = Stream(server.url, {"Last-Event-ID": "not-a-number"})
    try:
        reset = _reset_after_connecting(stream)
        assert json.loads(reset["data"])["data"]["reason"] == "invalid_last_event_id"
    finally:
        stream.close()


def test_stream_cap_returns_503_and_a_disconnect_frees_the_slot(start_server):
    server = start_server(STF_EVENTS_MAX_STREAMS="2")
    a, b = Stream(server.url), Stream(server.url)
    try:
        refused = httpx.get(server.url, timeout=5)
        assert refused.status_code == 503
        assert refused.headers["retry-after"] == "5"

        a.close()
        deadline = time.monotonic() + 10
        while True:
            retry = Stream(server.url)
            if retry.response.status_code == 200:
                break
            retry.close()
            assert time.monotonic() < deadline, "slot was never released"
            time.sleep(0.1)
        retry.close()
    finally:
        b.close()


def _client_header(address: str) -> dict[str, str]:
    return {"CF-Connecting-IP": address}


def test_handshakes_beyond_the_per_minute_limit_get_429_and_no_replay(start_server):
    # STF_TRUST_CF_HEADERS makes stf.webapp.security.client_ip see distinct clients
    # although every connection comes from loopback.
    server = start_server(
        STF_EVENTS_CONNECT_PER_MINUTE="3", STF_TRUST_CF_HEADERS="1"
    )
    server.append("one")
    server.append("two")
    greedy = _client_header("203.0.113.7")

    for _ in range(3):
        stream = Stream(server.url, {**greedy, "Last-Event-ID": "0"})
        try:
            assert _ids(stream.events(2)) == [1, 2]
        finally:
            stream.close()

    for query in ("", "?last_event_id=0"):
        rejected = httpx.get(
            server.url + query, headers={**greedy, "Last-Event-ID": "0"}, timeout=5
        )
        assert rejected.status_code == 429
        assert 1 <= int(rejected.headers["retry-after"]) <= 60
        assert rejected.headers["content-type"].startswith("application/json")
        # An answer to the refused handshake carries no frame: the log was not read.
        assert rejected.json() == {"detail": "too many stream connections"}
        assert "id:" not in rejected.text
        assert "event:" not in rejected.text

    other = Stream(server.url, {**_client_header("203.0.113.8"), "Last-Event-ID": "0"})
    try:
        assert _ids(other.events(2)) == [1, 2]
    finally:
        other.close()


def test_per_ip_concurrency_cap_refuses_without_replay_and_frees_on_disconnect(
    start_server,
):
    server = start_server(STF_EVENTS_MAX_PER_IP="1", STF_TRUST_CF_HEADERS="1")
    server.append("one")
    mine = _client_header("198.51.100.1")

    first = Stream(server.url, {**mine, "Last-Event-ID": "0"})
    try:
        assert _ids(first.events(1)) == [1]

        refused = httpx.get(
            server.url, headers={**mine, "Last-Event-ID": "0"}, timeout=5
        )
        assert refused.status_code == 429
        assert refused.headers["retry-after"] == "30"
        assert refused.json() == {"detail": "too many concurrent streams"}
        assert "event:" not in refused.text

        elsewhere = Stream(server.url, _client_header("198.51.100.2"))
        elsewhere.close()
    finally:
        first.close()

    # The disconnect reaches the app asynchronously; the slot is then free again.
    deadline = time.monotonic() + 10
    while True:
        again = Stream(server.url, {**mine, "Last-Event-ID": "0"})
        if again.response.status_code == 200:
            break
        again.close()
        assert time.monotonic() < deadline, "per-IP slot was never released"
        time.sleep(0.1)
    try:
        assert _ids(again.events(1)) == [1]
    finally:
        again.close()


def test_sigterm_with_an_open_stream_ends_it_and_exits_promptly(start_server):
    server = start_server()
    stream = Stream(server.url)
    try:
        server.append("e")
        stream.events(1)  # the stream is live, not merely connected

        server.process.send_signal(signal.SIGTERM)
        started = time.monotonic()
        # uvicorn re-raises the captured SIGTERM once it has shut down cleanly.
        code = server.process.wait(timeout=15)
        assert code in (0, -signal.SIGTERM), "server hung on an open stream"
        assert time.monotonic() - started < 10

        while stream.next_frame(timeout=5) is not None:  # drains to end-of-stream
            pass
    finally:
        stream.close()


def test_periodic_reconcile_publishes_an_artifact_persisted_without_an_event(
    start_server,
):
    server = start_server(STF_EVENTS_RECONCILE_SECONDS="0.3")
    stream = Stream(server.url)
    try:
        arm_dir = server.live / "lstm_price"
        arm_dir.mkdir(parents=True)
        pd.DataFrame(
            {"ticker": ["FPT"], "observation_date": pd.to_datetime(["2026-09-21"])}
        ).to_parquet(arm_dir / "predictions_2026-09-21.parquet", index=False)

        [frame] = stream.events(1, timeout=15)
        assert frame["event"] == "signal.issued"
        assert json.loads(frame["data"])["data"]["arm"] == "lstm_price"
        stream.assert_quiet(1.0)  # later reconcile passes add nothing
        assert len(server.log.read_all()) == 1
    finally:
        stream.close()


def test_event_payload_models_are_in_the_openapi_document():
    spec = app.openapi()

    assert "/api/events" in spec["paths"]
    schemas = spec["components"]["schemas"]
    for name in (
        "SignalIssuedData",
        "NewsIngestedData",
        "JobStatusData",
        "ResetData",
        "SignalIssuedEvent",
        "NewsIngestedEvent",
        "JobStatusEvent",
        "ResetEvent",
    ):
        assert name in schemas, name
    item = spec["paths"]["/api/events"]["get"]["responses"]["200"]["content"][
        "text/event-stream"
    ]["itemSchema"]
    refs = {
        option["$ref"].rsplit("/", 1)[1]
        for option in item["properties"]["data"]["contentSchema"]["oneOf"]
    }
    assert refs == {
        "SignalIssuedEvent",
        "NewsIngestedEvent",
        "JobStatusEvent",
        "ResetEvent",
    }
    assert schemas["SignalIssuedEvent"]["properties"]["data"] == {
        "$ref": "#/components/schemas/SignalIssuedData"
    }

    # Refusals happen before any stream starts, so their bodies are JSON errors,
    # not event-stream frames.
    responses = spec["paths"]["/api/events"]["get"]["responses"]
    for status in ("429", "503"):
        assert list(responses[status]["content"]) == ["application/json"], status
        assert responses[status]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }
        assert "Retry-After" in responses[status]["headers"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("STF_EVENTS_MAX_STREAMS", "0"),
        ("STF_EVENTS_MAX_STREAMS", "many"),
        ("STF_EVENTS_MAX_STREAMS", "2.5"),
        ("STF_EVENTS_RECONCILE_SECONDS", "-1"),
        ("STF_EVENTS_RECONCILE_SECONDS", "nan"),
        ("STF_EVENTS_RECONCILE_SECONDS", "soon"),
    ],
)
def test_invalid_events_settings_fail_at_startup_like_the_hardening_settings(
    key, value
):
    with pytest.raises(ValueError, match=key):
        create_app({key: value})
