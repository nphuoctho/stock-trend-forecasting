"""Public-exposure hardening for the read-only API.

The API runs on a personal machine behind a Cloudflare Tunnel, so it is public
but unauthenticated. Everything here is configured from environment variables
(parsed once, at startup, by :meth:`SecuritySettings.from_env`):

``STF_CORS_ORIGINS``             comma-separated exact origins allowed cross-origin
``STF_CORS_ORIGIN_REGEX``        regex an origin must *fully* match (Vercel previews)
``STF_RATE_LIMIT``               requests per window per client IP (default 120, 0 = off)
``STF_RATE_LIMIT_WINDOW``        window length in seconds (default 60)
``STF_EVENTS_CONNECT_PER_MINUTE`` stream handshakes per IP per minute (default 6)
``STF_EVENTS_MAX_PER_IP``        concurrent streams per IP (default 3)
``STF_TRUST_CF_HEADERS``         ``1`` = take the client IP from ``CF-Connecting-IP``
``STF_ALLOWED_HOSTS``            comma-separated Host header allow-list (unset = any)

Invalid values fail loudly at startup rather than silently widening access.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# The long-lived SSE stream (served by a separate change) gets two extra
# per-IP limits on top of the general request limiter.
EVENTS_PATH = "/api/events"
DEFAULT_EVENTS_CONNECT_PER_MINUTE = 6
DEFAULT_EVENTS_MAX_PER_IP = 3
EVENTS_CAP_RETRY_AFTER_SECONDS = 30

DEFAULT_RATE_LIMIT = 120
DEFAULT_RATE_WINDOW_SECONDS = 60.0
# Upper bound on tracked client keys; the least recently seen is dropped first.
MAX_TRACKED_CLIENTS = 10_000

CORS_ALLOWED_METHODS = ["GET", "HEAD", "OPTIONS"]
# Cross-origin scripts can only read Retry-After if it is exposed.
CORS_EXPOSED_HEADERS = ["Retry-After"]

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
# Artifacts change at most daily, but a stale answer after the daily job is a
# visible bug; JSON is never stored by shared caches unless a proxy opts in.
JSON_CACHE_CONTROL = "no-store"


def _split_csv(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _flag(raw: str | None) -> bool:
    return (raw or "").strip() == "1"


def _int_env(
    env: Mapping[str, str], key: str, default: int, *, minimum: int = 0
) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from error
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class SecuritySettings:
    cors_origins: list[str] = field(default_factory=list)
    cors_origin_regex: str | None = None
    rate_limit: int = DEFAULT_RATE_LIMIT
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS
    trust_cf_headers: bool = False
    events_connect_per_minute: int = DEFAULT_EVENTS_CONNECT_PER_MINUTE
    events_max_per_ip: int = DEFAULT_EVENTS_MAX_PER_IP
    allowed_hosts: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SecuritySettings:
        origins = []
        for origin in _split_csv(env.get("STF_CORS_ORIGINS")):
            origin = origin.rstrip("/")
            if "*" in origin:
                raise ValueError(
                    f"STF_CORS_ORIGINS must list exact origins, got {origin!r}"
                )
            origins.append(origin)

        regex = (env.get("STF_CORS_ORIGIN_REGEX") or "").strip() or None
        if regex is not None:
            try:
                re.compile(regex)
            except re.error as error:
                raise ValueError(
                    f"STF_CORS_ORIGIN_REGEX is invalid: {error}"
                ) from error

        window = _int_env(
            env, "STF_RATE_LIMIT_WINDOW", int(DEFAULT_RATE_WINDOW_SECONDS), minimum=1
        )

        return cls(
            cors_origins=origins,
            cors_origin_regex=regex,
            rate_limit=_int_env(env, "STF_RATE_LIMIT", DEFAULT_RATE_LIMIT),
            rate_window_seconds=float(window),
            events_connect_per_minute=_int_env(
                env,
                "STF_EVENTS_CONNECT_PER_MINUTE",
                DEFAULT_EVENTS_CONNECT_PER_MINUTE,
                minimum=1,
            ),
            events_max_per_ip=_int_env(
                env, "STF_EVENTS_MAX_PER_IP", DEFAULT_EVENTS_MAX_PER_IP, minimum=1
            ),
            trust_cf_headers=_flag(env.get("STF_TRUST_CF_HEADERS")),
            allowed_hosts=_split_csv(env.get("STF_ALLOWED_HOSTS")),
        )


class SecurityHeadersMiddleware:
    """Stamp hardening headers on every response, including errors and preflights."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or [])
                present = {key.lower() for key, _ in headers}
                for key, value in SECURITY_HEADERS.items():
                    headers.append((key.lower().encode(), value.encode()))
                if b"cache-control" not in present and any(
                    key.lower() == b"content-type"
                    and value.lower().startswith(b"application/json")
                    for key, value in headers
                ):
                    headers.append((b"cache-control", JSON_CACHE_CONTROL.encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class ErrorGuardMiddleware:
    """Turn unhandled exceptions into an opaque JSON 500.

    Starlette's own 500 handler sits outside user middleware, so without this
    an error response would carry neither CORS nor security headers. The
    exception (which may name filesystem paths) is logged, never sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("unhandled error on %s %s", scope["method"], scope["path"])
            if started:
                raise
            response = JSONResponse(
                {"detail": "internal server error"}, status_code=500
            )
            await response(scope, receive, send)


@dataclass
class _Bucket:
    tokens: float
    seen: float


class TokenBucketLimiter:
    """Per-key token bucket with bounded memory.

    A key idle for a full window has a full bucket again, so dropping it loses
    nothing. Keys are kept in recency order: idle ones are swept from the front
    and the table is hard-capped at ``max_keys`` on insert.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        max_keys: int = MAX_TRACKED_CLIENTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._rate = limit / window_seconds
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def __len__(self) -> int:
        return len(self._buckets)

    def acquire(self, key: str) -> float:
        """Spend one token; return 0.0 if allowed, else seconds until one is free."""
        now = self._clock()
        self._evict(now)
        bucket = self._buckets.get(key)
        if bucket is None:
            while len(self._buckets) >= self.max_keys:
                self._buckets.popitem(last=False)
            bucket = _Bucket(tokens=float(self.limit), seen=now)
            self._buckets[key] = bucket
        else:
            bucket.tokens = min(
                float(self.limit), bucket.tokens + (now - bucket.seen) * self._rate
            )
            bucket.seen = now
            self._buckets.move_to_end(key)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return 0.0
        return (1.0 - bucket.tokens) / self._rate

    def _evict(self, now: float) -> None:
        while self._buckets:
            key, bucket = next(iter(self._buckets.items()))
            if now - bucket.seen < self.window_seconds:
                break
            del self._buckets[key]


def _bucket_key(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    """The unit one subscriber controls: an IPv4 address or an IPv6 /64.

    A single IPv6 subscriber owns at least a /64, so limiting per full address
    would hand out a fresh budget for every address it cares to source from.
    IPv4-mapped IPv6 addresses count as the IPv4 address they carry.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Address(int(ip) >> 64 << 64))
    return str(ip)


def client_ip(scope: Scope, *, trust_cf: bool) -> str:
    """Rate-limit key of the client: its IPv4 address or IPv6 /64.

    The peer in ``scope["client"]`` is what uvicorn reports, and with proxy
    headers enabled uvicorn has already replaced it by the rightmost untrusted
    ``X-Forwarded-For`` entry when the TCP peer is loopback (the tunnel). With
    ``trust_cf`` (``STF_TRUST_CF_HEADERS=1``) a valid ``CF-Connecting-IP``
    takes precedence over that; an unparsable value falls back to the peer. A
    peer that is not an IP address (a unix socket path) is used as is.
    """
    if trust_cf:
        for key, value in scope.get("headers") or []:
            if key == b"cf-connecting-ip":
                try:
                    return _bucket_key(
                        ipaddress.ip_address(value.decode("latin-1").strip())
                    )
                except ValueError:
                    break
    client = scope.get("client")
    if not client:
        return "unknown"
    try:
        return _bucket_key(ipaddress.ip_address(client[0]))
    except ValueError:
        return client[0]


def _too_many_requests(detail: str, retry_after: int) -> JSONResponse:
    return JSONResponse(
        {"detail": detail},
        status_code=429,
        headers={"Retry-After": str(max(1, retry_after))},
    )


class RateLimitMiddleware:
    """Count every HTTP request once, on arrival, per client IP.

    Pure ASGI: the downstream response is never wrapped or buffered, so
    streaming bodies and disconnect detection are unaffected. A long-lived
    stream is one request and costs one token however many events it sends.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        limiter: TokenBucketLimiter,
        trust_cf_headers: bool = False,
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.trust_cf_headers = trust_cf_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        wait = self.limiter.acquire(client_ip(scope, trust_cf=self.trust_cf_headers))
        if wait > 0:
            response = _too_many_requests("rate limit exceeded", math.ceil(wait))
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class EventsLimitMiddleware:
    """Extra per-IP limits for the long-lived ``EVENTS_PATH`` stream.

    A handshake token bucket bounds how fast one IP may open streams, and a
    concurrency cap bounds how many it may hold at once. The concurrency slot is
    released in ``finally``, which runs when the response completes, the client
    disconnects, or the server cancels the request on shutdown.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        handshake: TokenBucketLimiter,
        max_streams_per_ip: int,
        trust_cf_headers: bool = False,
        path: str = EVENTS_PATH,
    ) -> None:
        self.app = app
        self.handshake = handshake
        self.max_streams_per_ip = max_streams_per_ip
        self.trust_cf_headers = trust_cf_headers
        self.path = path
        self._open: dict[str, int] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return
        ip = client_ip(scope, trust_cf=self.trust_cf_headers)
        wait = self.handshake.acquire(ip)
        if wait > 0:
            response = _too_many_requests(
                "too many stream connections", math.ceil(wait)
            )
            await response(scope, receive, send)
            return
        if self._open.get(ip, 0) >= self.max_streams_per_ip:
            response = _too_many_requests(
                "too many concurrent streams", EVENTS_CAP_RETRY_AFTER_SECONDS
            )
            await response(scope, receive, send)
            return
        self._open[ip] = self._open.get(ip, 0) + 1
        try:
            await self.app(scope, receive, send)
        finally:
            remaining = self._open[ip] - 1
            if remaining:
                self._open[ip] = remaining
            else:
                del self._open[ip]
