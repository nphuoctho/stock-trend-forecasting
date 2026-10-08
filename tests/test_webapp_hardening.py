"""Consumer-visible behaviour of the public API hardening.

uv run pytest tests/test_webapp_hardening.py -q
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from stf.cli import main
from stf.webapp import app as webapp_module
from stf.webapp.security import (
    RateLimitMiddleware,
    TokenBucketLimiter,
    client_ip,
)

PORTAL = "https://portal.example.com"
PREVIEW_REGEX = r"https://stf-portal-[a-z0-9-]+\.vercel\.app"


@pytest.fixture()
def outputs(tmp_path, monkeypatch):
    run_dir = tmp_path / "outputs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "forecast_results.json").write_text(
        json.dumps(
            {"panel_rows": 10, "provenance": {"news_sentiment": {"mode": "real"}}}
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "ticker": ["FPT"] * 3,
            "arm": ["majority"] * 3,
            "window": [1, 1, 2],
            "y_true": ["UP"] * 3,
            "y_pred": ["UP"] * 3,
        }
    ).to_csv(run_dir / "forecast_predictions.csv", index=False)
    monkeypatch.setattr(webapp_module, "_outputs_root", lambda: tmp_path / "outputs")
    return tmp_path


def make_client(
    env: dict[str, str] | None = None, *, ip: str = "203.0.113.1"
) -> TestClient:
    # A huge general limit keeps unrelated tests from tripping the default.
    merged = {"STF_RATE_LIMIT": "100000", **(env or {})}
    return TestClient(
        webapp_module.create_app(merged),
        client=(ip, 50000),
        raise_server_exceptions=False,
    )


# --- CORS ---------------------------------------------------------------------


def test_no_cross_origin_access_by_default(outputs):
    response = make_client().get("/api/runs", headers={"Origin": PORTAL})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_allowed_origin_gets_cors_headers_and_others_do_not(outputs):
    client = make_client({"STF_CORS_ORIGINS": f"{PORTAL}, https://other.example.com/"})

    allowed = client.get("/api/runs", headers={"Origin": PORTAL})
    assert allowed.headers["access-control-allow-origin"] == PORTAL
    assert "access-control-allow-credentials" not in allowed.headers

    trailing_slash_entry = client.get(
        "/api/runs", headers={"Origin": "https://other.example.com"}
    )
    assert (
        trailing_slash_entry.headers["access-control-allow-origin"]
        == "https://other.example.com"
    )

    for origin in (
        "https://evil.example",
        PORTAL.replace("https", "http"),
        PORTAL + ".evil.example",
    ):
        denied = client.get("/api/runs", headers={"Origin": origin})
        assert "access-control-allow-origin" not in denied.headers, origin


def test_preflight_allows_only_read_methods_from_allowed_origin(outputs):
    client = make_client({"STF_CORS_ORIGINS": PORTAL})

    ok = client.options(
        "/api/runs",
        headers={"Origin": PORTAL, "Access-Control-Request-Method": "GET"},
    )
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == PORTAL
    assert set(ok.headers["access-control-allow-methods"].split(", ")) == {
        "GET",
        "HEAD",
        "OPTIONS",
    }

    write = client.options(
        "/api/runs",
        headers={"Origin": PORTAL, "Access-Control-Request-Method": "DELETE"},
    )
    assert write.status_code == 400

    foreign = client.options(
        "/api/runs",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in foreign.headers


def test_origin_regex_must_match_the_whole_origin(outputs):
    client = make_client({"STF_CORS_ORIGIN_REGEX": PREVIEW_REGEX})

    preview = "https://stf-portal-git-feature-x.vercel.app"
    assert (
        client.get("/api/runs", headers={"Origin": preview}).headers[
            "access-control-allow-origin"
        ]
        == preview
    )

    tricks = [
        preview + ".evil.example",  # suffix
        "https://evil.example/" + preview,  # prefix
        preview + "/",  # not a serialised origin
        "https://evil.example#" + preview,
    ]
    for origin in tricks:
        response = client.get("/api/runs", headers={"Origin": origin})
        assert "access-control-allow-origin" not in response.headers, origin


@pytest.mark.parametrize(
    "env",
    [
        {"STF_CORS_ORIGINS": "*"},
        {"STF_CORS_ORIGINS": "https://*.example.com"},
        {"STF_CORS_ORIGIN_REGEX": "(unclosed"},
    ],
)
def test_unsafe_cors_configuration_fails_at_startup(env):
    with pytest.raises(ValueError):
        webapp_module.create_app(env)


# --- rate limiting --------------------------------------------------------------


def test_exceeding_the_limit_returns_429_with_retry_after(outputs):
    client = make_client({"STF_RATE_LIMIT": "3", "STF_RATE_LIMIT_WINDOW": "60"})
    assert [client.get("/api/runs").status_code for _ in range(3)] == [200, 200, 200]

    limited = client.get("/api/runs")
    assert limited.status_code == 429
    assert 1 <= int(limited.headers["retry-after"]) <= 60
    assert limited.json() == {"detail": "rate limit exceeded"}


def test_clients_are_limited_independently(outputs):
    app = webapp_module.create_app({"STF_RATE_LIMIT": "2"})
    first = TestClient(app, client=("198.51.100.1", 1))
    second = TestClient(app, client=("198.51.100.2", 1))

    assert [first.get("/api/runs").status_code for _ in range(3)] == [200, 200, 429]
    assert second.get("/api/runs").status_code == 200


def test_cf_connecting_ip_is_ignored_unless_trusted(outputs):
    untrusted = make_client({"STF_RATE_LIMIT": "2"})
    codes = [
        untrusted.get(
            "/api/runs", headers={"CF-Connecting-IP": f"192.0.2.{i}"}
        ).status_code
        for i in range(3)
    ]
    assert codes == [200, 200, 429]

    trusted = make_client({"STF_RATE_LIMIT": "2", "STF_TRUST_CF_HEADERS": "1"})
    codes = [
        trusted.get("/api/runs", headers={"CF-Connecting-IP": "192.0.2.7"}).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    # Same socket peer, different Cloudflare-reported clients: separate budgets.
    assert (
        trusted.get("/api/runs", headers={"CF-Connecting-IP": "192.0.2.8"}).status_code
        == 200
    )
    # A garbage header cannot mint a fresh identity; it falls back to the peer.
    assert (
        trusted.get("/api/runs", headers={"CF-Connecting-IP": "not-an-ip"}).status_code
        == 200
    )
    assert (
        trusted.get("/api/runs", headers={"CF-Connecting-IP": "also-bad"}).status_code
        == 200
    )
    assert (
        trusted.get("/api/runs", headers={"CF-Connecting-IP": "x" * 40}).status_code
        == 429
    )


def test_client_ip_helper_follows_the_trust_switch():
    scope = {
        "client": ("10.0.0.5", 1234),
        "headers": [(b"cf-connecting-ip", b"198.51.100.9")],
    }
    assert client_ip(scope, trust_cf=False) == "10.0.0.5"
    assert client_ip(scope, trust_cf=True) == "198.51.100.9"
    assert client_ip({"client": None, "headers": []}, trust_cf=True) == "unknown"


def _scope(peer: str | None, cf: str | None = None) -> dict:
    headers = [(b"cf-connecting-ip", cf.encode())] if cf else []
    return {"client": (peer, 1234) if peer else None, "headers": headers}


@pytest.mark.parametrize("trust_cf", [False, True])
def test_ipv6_addresses_of_one_slash_64_share_a_bucket(trust_cf):
    def key(address: str) -> str:
        # The peer path when untrusted, the Cloudflare header path when trusted.
        scope = _scope("::1", address) if trust_cf else _scope(address)
        return client_ip(scope, trust_cf=trust_cf)

    same_subnet = [
        key("2001:db8:1:2::1"),
        key("2001:db8:1:2:aaaa:bbbb:cccc:dddd"),
        key("2001:0db8:0001:0002:ffff:ffff:ffff:ffff"),
    ]
    assert len(set(same_subnet)) == 1
    assert key("2001:db8:1:3::1") != same_subnet[0]
    assert key("2001:db8:2:2::1") != same_subnet[0]


@pytest.mark.parametrize("trust_cf", [False, True])
def test_ipv4_and_ipv4_mapped_addresses_are_keyed_by_the_full_address(trust_cf):
    def key(address: str) -> str:
        scope = _scope("::1", address) if trust_cf else _scope(address)
        return client_ip(scope, trust_cf=trust_cf)

    assert key("203.0.113.7") == "203.0.113.7"
    assert key("203.0.113.8") != key("203.0.113.7")
    assert key("::ffff:203.0.113.7") == "203.0.113.7"


def test_a_peer_that_is_not_an_ip_address_is_used_unchanged():
    assert client_ip(_scope("/run/stf.sock"), trust_cf=False) == "/run/stf.sock"


def test_the_limiter_treats_one_ipv6_subnet_as_one_client(outputs):
    application = webapp_module.create_app({"STF_RATE_LIMIT": "2"})
    codes = [
        TestClient(application, client=(f"2001:db8:1:2::{i}", 1))
        .get("/api/runs")
        .status_code
        for i in range(1, 6)
    ]
    assert codes == [200, 200, 429, 429, 429]
    elsewhere = TestClient(application, client=("2001:db8:1:3::1", 1))
    assert elsewhere.get("/api/runs").status_code == 200


def test_webapp_command_pins_proxy_headers_to_loopback(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    assert main(["webapp", "--host", "127.0.0.1", "--port", "8123"]) == 0

    [(args, kwargs)] = calls
    assert args == ("stf.webapp.app:app",)
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123
    # The trust boundary for X-Forwarded-For: only the local tunnel may set it.
    assert kwargs["proxy_headers"] is True
    assert kwargs["forwarded_allow_ips"] == "127.0.0.1,::1"


# --- token bucket ---------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_bucket_refills_at_limit_per_window_and_reports_the_wait():
    clock = FakeClock()
    limiter = TokenBucketLimiter(3, 60.0, clock=clock)  # one token per 20 s

    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire("a") == pytest.approx(20.0)

    clock.now = 10.0
    assert limiter.acquire("a") == pytest.approx(10.0)  # half a token has accrued
    clock.now = 20.0
    assert limiter.acquire("a") == 0.0
    assert limiter.acquire("a") == pytest.approx(20.0)

    clock.now = 1_000.0  # idle for far longer than a window: capped at the burst
    assert [limiter.acquire("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire("a") > 0


def test_retry_after_header_is_the_wait_rounded_up(outputs):
    clock = FakeClock()
    limiter = TokenBucketLimiter(4, 8.0, clock=clock)  # 0.5 token per second
    middleware = RateLimitMiddleware(webapp_module.create_app({}), limiter=limiter)
    client = TestClient(middleware, client=("203.0.113.1", 1))

    assert [client.get("/api/runs").status_code for _ in range(4)] == [200] * 4
    clock.now = 0.75  # 0.375 token back: 1.25 s to the next one, advertised as 2
    limited = client.get("/api/runs")
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "2"


def test_idle_keys_are_swept_after_a_full_window():
    clock = FakeClock()
    limiter = TokenBucketLimiter(3, 60.0, clock=clock)
    limiter.acquire("idle")
    clock.now = 30.0
    limiter.acquire("active")
    assert len(limiter) == 2

    clock.now = 60.0  # "idle" has now been quiet for a whole window
    limiter.acquire("active")
    assert len(limiter) == 1
    # Sweeping loses nothing: a returning client starts with a full bucket.
    assert [limiter.acquire("idle") for _ in range(3)] == [0.0, 0.0, 0.0]


def test_tracked_keys_never_exceed_max_keys_and_the_least_recent_is_dropped():
    limiter = TokenBucketLimiter(1, 60.0, max_keys=4, clock=FakeClock())
    for key in "abcd":
        limiter.acquire(key)
    limiter.acquire("a")  # "a" is now the most recently seen; "b" the oldest
    limiter.acquire("e")
    assert len(limiter) == 4

    # "b" was dropped, so it gets a fresh token; "a" still has none left.
    assert limiter.acquire("a") > 0
    assert limiter.acquire("b") == 0.0
    assert len(limiter) == 4


def test_cors_and_security_headers_reach_429_so_the_portal_can_read_it(outputs):
    client = make_client({"STF_RATE_LIMIT": "1", "STF_CORS_ORIGINS": PORTAL})
    client.get("/api/runs")
    limited = client.get("/api/runs", headers={"Origin": PORTAL})
    assert limited.status_code == 429
    assert limited.headers["access-control-allow-origin"] == PORTAL
    assert "retry-after" in limited.headers["access-control-expose-headers"].lower()
    assert limited.headers["x-content-type-options"] == "nosniff"


# --- SSE limits (test-only /api/events route; the real one ships separately) ----


def add_events_route(application) -> None:
    @application.get("/api/events")
    async def events(n: int = 1, hold: bool = False):
        async def body():
            for i in range(n):
                yield f"data: {i}\n\n"
            if hold:
                await asyncio.Event().wait()

        return StreamingResponse(body(), media_type="text/event-stream")


def events_app(env: dict[str, str] | None = None):
    application = webapp_module.create_app({"STF_RATE_LIMIT": "100000", **(env or {})})
    add_events_route(application)
    return application


def test_seventh_handshake_in_a_minute_is_rejected(outputs):
    client = TestClient(events_app(), client=("203.0.113.9", 1))
    assert [client.get("/api/events").status_code for _ in range(6)] == [200] * 6

    rejected = client.get("/api/events")
    assert rejected.status_code == 429
    assert 1 <= int(rejected.headers["retry-after"]) <= 10

    other = TestClient(client.app, client=("203.0.113.10", 1))
    assert other.get("/api/events").status_code == 200


class Stream:
    """A raw ASGI connection that stays open until ``disconnect()`` is called."""

    def __init__(self, application, ip: str, query: str = "hold=true") -> None:
        self._disconnected = asyncio.Event()
        self.status: int | None = None
        self.headers: dict[bytes, bytes] = {}
        self.chunks = 0
        self._started = asyncio.Event()
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/events",
            "raw_path": b"/api/events",
            "query_string": query.encode(),
            "headers": [(b"host", b"testserver")],
            "client": (ip, 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            await self._disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                self.status = message["status"]
                self.headers = dict(message["headers"])
                self._started.set()
            elif message["type"] == "http.response.body" and message.get("body"):
                self.chunks += 1

        self.task = asyncio.create_task(application(scope, receive, send))

    async def started(self) -> Stream:
        await asyncio.wait_for(self._started.wait(), timeout=5)
        return self

    async def finished(self) -> None:
        await asyncio.wait_for(self.task, timeout=5)

    async def disconnect(self) -> None:
        self._disconnected.set()
        await self.finished()


def test_fourth_concurrent_stream_from_one_ip_is_rejected(outputs):
    async def scenario():
        application = events_app()
        streams = [
            await Stream(application, "198.51.100.1").started() for _ in range(3)
        ]
        assert [s.status for s in streams] == [200, 200, 200]

        fourth = await Stream(application, "198.51.100.1").started()
        assert fourth.status == 429
        assert fourth.headers[b"retry-after"] == b"30"
        await fourth.finished()

        other_ip = await Stream(application, "198.51.100.2").started()
        assert other_ip.status == 200

        for stream in [*streams, other_ip]:
            await stream.disconnect()

    asyncio.run(scenario())


def test_slot_is_freed_when_the_client_disconnects(outputs):
    async def scenario():
        application = events_app(
            {"STF_EVENTS_MAX_PER_IP": "1", "STF_EVENTS_CONNECT_PER_MINUTE": "10"}
        )
        first = await Stream(application, "198.51.100.1").started()
        assert first.status == 200

        blocked = await Stream(application, "198.51.100.1").started()
        assert blocked.status == 429
        await blocked.finished()

        await first.disconnect()

        again = await Stream(application, "198.51.100.1").started()
        assert again.status == 200
        await again.disconnect()

    asyncio.run(scenario())


def test_slot_is_freed_when_the_stream_handler_raises(outputs):
    application = webapp_module.create_app(
        {
            "STF_RATE_LIMIT": "100000",
            "STF_EVENTS_MAX_PER_IP": "1",
            "STF_EVENTS_CONNECT_PER_MINUTE": "10",
        }
    )

    @application.get("/api/events")
    async def broken():
        raise RuntimeError("boom")

    client = TestClient(
        application, client=("203.0.113.5", 1), raise_server_exceptions=False
    )
    # A leaked slot would turn the second and third attempts into 429s.
    assert [client.get("/api/events").status_code for _ in range(3)] == [500, 500, 500]


def test_a_long_stream_costs_one_handshake_and_one_general_token(outputs):
    async def scenario():
        application = events_app(
            {"STF_EVENTS_CONNECT_PER_MINUTE": "2", "STF_RATE_LIMIT": "3"}
        )
        busy = await Stream(application, "198.51.100.1", "n=200").started()
        await busy.finished()
        assert busy.status == 200
        assert busy.chunks >= 200  # every event was delivered...

        # ...yet only one of the two handshake tokens and one of the three
        # general tokens were spent.
        second = await Stream(application, "198.51.100.1", "n=1").started()
        assert second.status == 200
        await second.finished()

        third = await Stream(application, "198.51.100.1", "n=1").started()
        assert third.status == 429  # handshake bucket (2) is spent; general (3) is too
        await third.finished()

    asyncio.run(scenario())


# --- headers, host allow-list, parameter bounds -----------------------------------


def test_security_headers_are_on_every_response(outputs):
    client = make_client({"STF_RATE_LIMIT": "100", "STF_CORS_ORIGINS": PORTAL})
    responses = [
        client.get("/api/runs"),
        client.get("/api/runs/missing/summary"),
        client.get("/api/runs/run_a/predictions?limit=0"),
        client.options(
            "/api/runs",
            headers={"Origin": PORTAL, "Access-Control-Request-Method": "GET"},
        ),
    ]
    for response in responses:
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-frame-options"] == "DENY"
    assert [r.status_code for r in responses] == [200, 404, 422, 200]
    # JSON answers are not cacheable by shared caches.
    assert responses[0].headers["cache-control"] == "no-store"


def test_allowed_hosts_rejects_other_host_headers(outputs):
    restricted = make_client({"STF_ALLOWED_HOSTS": "api.example.com"})
    assert (
        restricted.get("/api/runs", headers={"Host": "api.example.com"}).status_code
        == 200
    )
    rejected = restricted.get("/api/runs", headers={"Host": "evil.example"})
    assert rejected.status_code == 400
    assert rejected.headers["x-content-type-options"] == "nosniff"

    # Unset means any Host is accepted.
    assert (
        make_client().get("/api/runs", headers={"Host": "anything.example"}).status_code
        == 200
    )


@pytest.mark.parametrize(
    "query",
    [
        "limit=5001",
        "limit=0",
        "offset=1000001",
        "offset=-1",
        "window=1001",
        "arm=" + "a" * 65,
        "ticker=" + "T" * 65,
    ],
)
def test_query_parameters_are_bounded(outputs, query):
    assert make_client().get(f"/api/runs/run_a/predictions?{query}").status_code == 422


def test_query_parameter_limits_are_inclusive(outputs):
    client = make_client()
    response = client.get(
        "/api/runs/run_a/predictions?limit=5000&offset=1000000&window=1000"
    )
    assert response.status_code == 200
    assert response.json() == {"total": 0, "rows": []}


# --- error responses and run-name lookup ---------------------------------------------


def test_run_name_cannot_reach_outside_outputs(outputs):
    secret = outputs / "secret"
    secret.mkdir()
    (secret / "forecast_results.json").write_text("{}", encoding="utf-8")
    client = make_client()

    for name in (
        "..",
        "../secret",
        "..%2Fsecret",
        "%2e%2e%2fsecret",
        str(secret),
        "run_a/../../secret",
    ):
        response = client.get(f"/api/runs/{name}/summary")
        assert response.status_code == 404, name
        assert str(outputs) not in response.text

    assert client.get("/api/runs/" + "x" * 129 + "/summary").status_code == 422


def test_errors_never_leak_filesystem_paths_or_tracebacks(outputs):
    run = outputs / "outputs" / "run_a"
    (run / "forecast_results.json").write_text("{not json", encoding="utf-8")
    (run / "forecast_metrics.csv").write_bytes(b"\x00\xff\xfe,\n\x00\x00\x00\x00")
    (run / "forecast_stratified.csv").write_text("", encoding="utf-8")
    client = make_client({"STF_CORS_ORIGINS": PORTAL})

    urls = [
        "/api/runs",
        "/api/runs/run_a/summary",
        "/api/runs/run_a/metrics",
        "/api/runs/run_a/stratified",
        "/api/runs/run_a/information-gain",
        "/api/runs/run_a/predictions?limit=notanumber",
    ]
    for url in urls:
        response = client.get(url, headers={"Origin": PORTAL})
        assert response.status_code >= 400, url
        for needle in (
            str(outputs),
            "/home/",
            "Traceback",
            ".py",
            "forecast_results.json:",
        ):
            assert needle not in response.text, (url, needle)
        # Even a crash goes out with the portal's CORS and the hardening headers.
        assert response.headers["access-control-allow-origin"] == PORTAL
        assert response.headers["x-content-type-options"] == "nosniff"


def test_unreadable_artifact_is_logged_but_answered_with_the_opaque_500(
    outputs, caplog
):
    run = outputs / "outputs" / "run_a"
    (run / "forecast_results.json").write_text("{not json", encoding="utf-8")

    response = make_client().get("/api/runs/run_a/summary")

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "forecast_results.json" in caplog.text


# --- OpenAPI contract ----------------------------------------------------------------


def test_committed_openapi_is_current(tmp_path):
    """The portal generates its types from openapi.json; regenerate it when this fails:

    uv run python -m stf.cli openapi --output openapi.json
    """
    generated = tmp_path / "openapi.json"
    assert main(["openapi", "--output", str(generated)]) == 0

    committed = Path(__file__).resolve().parents[1] / "openapi.json"
    assert committed.read_text(encoding="utf-8") == generated.read_text(
        encoding="utf-8"
    )


def test_every_endpoint_documents_a_response_schema_and_its_errors():
    document = json.loads(webapp_module.render_openapi())
    api_paths = {
        path: item
        for path, item in document["paths"].items()
        if path.startswith("/api/")
    }
    assert len(api_paths) == 10

    for path, item in api_paths.items():
        responses = item["get"]["responses"]
        schema = responses["200"]["content"]["application/json"]["schema"]
        assert schema.get("$ref", "").startswith("#/components/schemas/"), path
        assert "429" in responses, path
        if path not in ("/api/runs", "/api/live/status"):
            assert "404" in responses, path
