"""``GET /api/events``: Server-Sent Events over ``outputs/live/events.jsonl``.

The log is written by the daily job (a different process), so each stream polls
the file with its own :class:`stf.events.LogTail` (one ``stat`` per poll) instead
of relying on in-process pub/sub: no broker and no new dependency. Framing, the
15 s keep-alive comment and the ``Cache-Control``/``X-Accel-Buffering`` headers
come from FastAPI's native ``EventSourceResponse``.

Environment (parsed once when the app is built; an invalid value fails at startup,
like the hardening settings in :mod:`stf.webapp.security`):

``STF_EVENTS_MAX_STREAMS``         concurrent streams before 503 (default 50)
``STF_EVENTS_RECONCILE_SECONDS``   API-side reconcile period (default 300)

Per-IP handshake and concurrency limits for this route live in
:class:`stf.webapp.security.EventsLimitMiddleware`; only the global cap is here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import signal
import threading
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from stf import events
from stf.webapp import schemas
from stf.webapp.security import EVENTS_PATH, client_ip

LOGGER = logging.getLogger(__name__)

POLL_SECONDS = 0.5
RETRY_MILLISECONDS = 5000
DEFAULT_MAX_STREAMS = 50
DEFAULT_RECONCILE_SECONDS = 300.0


@dataclass(frozen=True)
class EventsSettings:
    max_streams: int = DEFAULT_MAX_STREAMS
    reconcile_seconds: float = DEFAULT_RECONCILE_SECONDS

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> EventsSettings:
        return cls(
            max_streams=int(
                _positive(env, "STF_EVENTS_MAX_STREAMS", DEFAULT_MAX_STREAMS, int)
            ),
            reconcile_seconds=float(
                _positive(
                    env,
                    "STF_EVENTS_RECONCILE_SECONDS",
                    DEFAULT_RECONCILE_SECONDS,
                    float,
                )
            ),
        )


def _positive(
    env: Mapping[str, str], key: str, default: float, cast: Callable[[str], float]
) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = cast(raw)
    except ValueError as error:
        raise ValueError(f"{key} must be a number, got {raw!r}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a positive number, got {raw!r}")
    return value


def _json_error(description: str) -> dict[str, Any]:
    """An OpenAPI response entry for a JSON error body on the event-stream route.

    Spelled out instead of ``{"model": ...}`` because FastAPI would label the body
    with the route's own ``text/event-stream`` media type, but every refusal here
    is a JSON ``{"detail": ...}`` produced before any stream starts.
    """
    ref = f"#/components/schemas/{schemas.ErrorResponse.__name__}"
    return {
        "description": description,
        "headers": {"Retry-After": {"schema": {"type": "integer"}}},
        "content": {"application/json": {"schema": {"$ref": ref}}},
    }


def _sse(event: dict[str, Any]) -> ServerSentEvent:
    return ServerSentEvent(data=event, event=event["type"], id=str(event["id"]))


def _reset(reason: str, head_id: int) -> ServerSentEvent:
    event = events.ResetEvent(
        id=head_id,
        type="reset",
        data=events.ResetData(reason=reason, head_id=head_id),
    )
    return _sse(event.model_dump(mode="json"))


def _chain_shutdown_signals(
    loop: asyncio.AbstractEventLoop, stop: asyncio.Event
) -> Callable[[], None]:
    """Set ``stop`` when SIGINT/SIGTERM arrive, then defer to the server's handler.

    uvicorn waits for every in-flight response before it runs the ASGI lifespan
    shutdown, so an open event stream would block a restart forever and the
    lifespan hook alone cannot end it. The server's own handler is called
    unchanged afterwards, so its shutdown behaviour is untouched. Outside the main
    thread (test clients) signals cannot be hooked and nothing is installed.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    installed: dict[int, tuple[Any, Any]] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)
        if not callable(previous):
            continue

        def handler(signum, frame, previous=previous):
            with contextlib.suppress(RuntimeError):  # loop already closed
                loop.call_soon_threadsafe(stop.set)
            previous(signum, frame)

        signal.signal(sig, handler)
        installed[sig] = (handler, previous)

    def restore() -> None:
        for sig, (handler, previous) in installed.items():
            if signal.getsignal(sig) is handler:
                signal.signal(sig, previous)

    return restore


class _Runtime:
    """Per-event-loop state: the shutdown flag and the open-stream count."""

    def __init__(self) -> None:
        self.stop = asyncio.Event()
        self.streams = 0


class EventsService:
    """The ``/api/events`` route plus the lifespan that keeps the log reconciled."""

    def __init__(self, live_dir: Callable[[], Path], settings: EventsSettings):
        self._live_dir = live_dir
        self._settings = settings
        self._runtime: _Runtime | None = None
        self.router = APIRouter()
        self._add_route()

    def _ensure_runtime(self) -> _Runtime:
        if self._runtime is None:
            self._runtime = _Runtime()
        return self._runtime

    # -- route ----------------------------------------------------------------

    def _add_route(self) -> None:
        service = self

        async def stream_slot(request: Request) -> AsyncIterator[None]:
            """Reserve one of the capped stream slots until the response ends.

            FastAPI unwinds a ``yield`` dependency after the streaming response
            finishes, was cancelled by a client disconnect, or was ended by
            shutdown, so the slot cannot leak on any of those paths.
            """
            runtime = service._ensure_runtime()
            if runtime.stop.is_set() or runtime.streams >= service._settings.max_streams:
                LOGGER.warning(
                    "events: refused a stream from %s (%d open, cap %d%s)",
                    client_ip(request.scope),
                    runtime.streams,
                    service._settings.max_streams,
                    ", shutting down" if runtime.stop.is_set() else "",
                )
                raise HTTPException(
                    status_code=503,
                    detail="too many open event streams",
                    headers={"Retry-After": "5"},
                )
            runtime.streams += 1
            try:
                yield
            finally:
                runtime.streams -= 1

        async def stream_events(
            last_event_id_header: Annotated[
                str | None,
                Header(
                    alias="Last-Event-ID",
                    description="Id of the last event the client received; "
                    "everything after it is replayed before the stream tails the log.",
                ),
            ] = None,
            last_event_id: Annotated[
                str | None,
                Query(
                    description="Same as the Last-Event-ID header, for clients "
                    "that cannot set headers. The header wins when both are sent."
                ),
            ] = None,
            _slot: None = Depends(stream_slot),
        ) -> AsyncIterable[events.StreamEvent]:
            requested = (
                last_event_id_header if last_event_id_header is not None else last_event_id
            )
            async with contextlib.aclosing(service._stream(requested)) as stream:
                async for item in stream:
                    yield item

        self.router.add_api_route(
            EVENTS_PATH,
            stream_events,
            methods=["GET"],
            response_class=EventSourceResponse,
            summary="Live signals, news and job status (Server-Sent Events)",
            description=(
                "`text/event-stream`. Each frame carries `id: <log id>`, "
                "`event: <type>` and one `data:` line holding the JSON event. "
                "A reconnect with `Last-Event-ID` replays the events it missed; "
                "an id the log cannot serve yields a `reset` event, after which "
                "the client must refetch its state. `: ping` comments keep the "
                "connection alive and carry no data."
            ),
            responses={
                429: _json_error(
                    "This client opened streams too fast or holds too many; "
                    "see the Retry-After header (seconds)."
                ),
                503: _json_error(
                    "The server is at its global stream cap or shutting down."
                ),
            },
        )

    # -- stream ---------------------------------------------------------------

    async def _stream(self, requested: str | None) -> AsyncIterator[ServerSentEvent]:
        runtime = self._ensure_runtime()
        tail = events.EventLog(self._live_dir()).tail()
        backlog, _ = tail.read()
        head = backlog[-1]["id"] if backlog else 0
        first = backlog[0]["id"] if backlog else None

        yield ServerSentEvent(retry=RETRY_MILLISECONDS)

        if requested is None:
            replay: list[dict[str, Any]] = []  # fresh client: tail from the head
        else:
            text = requested.strip()
            if not (text.isascii() and text.isdigit()):
                replay = []
                yield _reset("invalid_last_event_id", head)
            else:
                last = int(text)
                if last > head or (first is not None and last < first - 1):
                    replay = []
                    yield _reset("stale_last_event_id", head)
                else:
                    replay = [event for event in backlog if event["id"] > last]
        for event in replay:
            yield _sse(event)

        while not runtime.stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(runtime.stop.wait(), timeout=POLL_SECONDS)
            if runtime.stop.is_set():
                break
            fresh, replaced = tail.read()
            if replaced:
                yield _reset("log_replaced", fresh[-1]["id"] if fresh else 0)
                continue
            for event in fresh:
                yield _sse(event)

    # -- lifespan -------------------------------------------------------------

    async def _reconcile_once(self) -> None:
        try:
            appended = await asyncio.to_thread(events.reconcile, self._live_dir())
        except Exception:
            LOGGER.exception("events: reconcile failed")
            return
        if appended:
            LOGGER.info("events: reconcile recovered %d event(s)", len(appended))

    async def _reconcile_loop(self, runtime: _Runtime) -> None:
        period = self._settings.reconcile_seconds
        while not runtime.stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(runtime.stop.wait(), timeout=period)
            if runtime.stop.is_set():
                return
            await self._reconcile_once()

    @contextlib.asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        runtime = self._runtime = _Runtime()
        restore = _chain_shutdown_signals(asyncio.get_running_loop(), runtime.stop)
        await self._reconcile_once()
        reconciler = asyncio.create_task(self._reconcile_loop(runtime))
        try:
            yield
        finally:
            runtime.stop.set()
            reconciler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reconciler
            restore()
