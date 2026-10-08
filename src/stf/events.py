"""Append-only event log for the live API, derived from ``outputs/live`` artifacts.

The files under ``outputs/live/`` are the source of truth; ``events.jsonl`` is an
index of them that the SSE endpoint replays and tails. An event exists only for a
*newly persisted* result:

``signal.issued``
    one per arm per dated prediction file (key = arm + observation date + the
    sha256 of the persisted file).
``news.ingested``
    one per ``score-news --incremental`` batch that scored at least one new
    article (key = digest of the batch's ``(ticker, url)`` pairs).
``job.status``
    one per ``last_run.json`` (key = its ``finished_at``).

Every event key is a pure function of persisted artifacts, so :func:`reconcile`
can re-derive any committed-but-unlogged event after a crash (SIGKILL, power
loss) and appending an existing key is a no-op. Producers, the daily script and
the API all call the same code, so running it any number of times never
duplicates an event and never invents one.

Line format (one JSON object per line)::

    {"id": 7, "ts": "2026-10-08T08:31:02+00:00", "type": "signal.issued",
     "key": "signal.issued:lstm_price:2026-10-07:<sha256>", "data": {...}}

``id`` is assigned under an exclusive ``flock`` on ``events.lock`` so the daily
job and the API can append concurrently. ``news_batches.jsonl`` is the write-ahead
ledger that makes a news batch recoverable: ``score-news`` records the sha256 of
the parquet it is about to commit *before* the atomic rename, so a batch whose
recorded hash equals the file on disk is provably committed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import pyarrow.parquet as pq
from pydantic import BaseModel, Field

from stf import config

LOGGER = logging.getLogger(__name__)

EVENTS_FILE = "events.jsonl"
LOCK_FILE = "events.lock"
NEWS_LEDGER_FILE = "news_batches.jsonl"
LAST_RUN_FILE = "last_run.json"

_DATED_PREDICTIONS = re.compile(r"^predictions_(\d{4}-\d{2}-\d{2})\.parquet$")


def live_dir() -> Path:
    """Directory holding the live artifacts and the event log."""
    return Path(config.ROOT) / "outputs" / "live"


# --- Payload and wire models (exported to OpenAPI by the SSE route) -------------


class SignalIssuedData(BaseModel):
    """One arm's predictions for the next session were persisted."""

    arm: str
    observation_date: date = Field(
        description="Session the features were observed on; the forecast targets "
        "the next trading session after it."
    )
    n_predictions: int = Field(description="Rows (tickers) in the persisted file.")
    issued_at: str | None = Field(
        default=None, description="Issuance stamp written into the prediction file."
    )


class NewsIngestedData(BaseModel):
    """A batch of newly scored articles was persisted."""

    count: int = Field(description="Articles scored in this batch.")
    total_scored: int = Field(description="Rows in the scored-news file afterwards.")
    by_ticker: dict[str, int] = Field(description="Articles scored per ticker.")


class JobStatusData(BaseModel):
    """Outcome of one run of the daily job, as written to ``last_run.json``."""

    ok: bool
    exit_code: int
    failed_step: str | None = None
    finished_at: str


class ResetData(BaseModel):
    """The client's ``Last-Event-ID`` cannot be served; it must refetch state."""

    reason: Literal["stale_last_event_id", "invalid_last_event_id", "log_replaced"]
    head_id: int = Field(description="Id of the newest event; the stream resumes here.")


class _LoggedEvent(BaseModel):
    id: int = Field(description="Monotonic event id; also the SSE `id:` field.")
    ts: datetime
    key: str = Field(description="Deterministic idempotency key.")


class SignalIssuedEvent(_LoggedEvent):
    type: Literal["signal.issued"]
    data: SignalIssuedData


class NewsIngestedEvent(_LoggedEvent):
    type: Literal["news.ingested"]
    data: NewsIngestedData


class JobStatusEvent(_LoggedEvent):
    type: Literal["job.status"]
    data: JobStatusData


class ResetEvent(BaseModel):
    """Control event, never logged: sent in place of a replay that cannot be served."""

    id: int
    type: Literal["reset"]
    data: ResetData


StreamEvent = Annotated[
    Union[SignalIssuedEvent, NewsIngestedEvent, JobStatusEvent, ResetEvent],
    Field(discriminator="type"),
]

_PAYLOADS: dict[str, type[BaseModel]] = {
    "signal.issued": SignalIssuedData,
    "news.ingested": NewsIngestedData,
    "job.status": JobStatusData,
}


# --- Log ------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_lines(raw: bytes) -> Iterator[dict[str, Any]]:
    """Yield valid events from newline-terminated JSON lines, skipping garbage."""
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            LOGGER.warning("events: skipping unparseable log line")
            continue
        if (
            isinstance(event, dict)
            and isinstance(event.get("id"), int)
            and isinstance(event.get("type"), str)
            and isinstance(event.get("key"), str)
        ):
            yield event
        else:
            LOGGER.warning("events: skipping malformed log entry")


def _read_complete(path: Path) -> bytes:
    """Newline-terminated content of ``path``; the caller holds the lock.

    A torn tail means an appender died mid-write (power loss). That line was never
    acknowledged, and a new line glued to it would corrupt both, so it is cut off.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return b""
    complete = raw.rfind(b"\n") + 1
    if complete != len(raw):
        LOGGER.warning("events: truncating torn tail of %s", path)
        os.truncate(path, complete)
    return raw[:complete]


def _append_durably(path: Path, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


class EventLog:
    """``events.jsonl`` plus its ``flock`` sidecar inside one live directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.path = self.directory / EVENTS_FILE
        self.lock_path = self.directory / LOCK_FILE

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive cross-process lock shared by the log and the news ledger."""
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)  # closing releases the flock

    def read_all(self) -> list[dict[str, Any]]:
        """Every complete event currently in the log (lock-free)."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return []
        return list(_parse_lines(raw[: raw.rfind(b"\n") + 1]))

    def keys(self) -> set[str]:
        return {event["key"] for event in self.read_all()}

    def append(
        self, event_type: str, key: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Append one event, or return the existing event that already has ``key``.

        The scan, the id assignment and the write all happen under the exclusive
        lock, so concurrent appenders never share an id or write a key twice.
        """
        with self.locked():
            existing = list(_parse_lines(_read_complete(self.path)))
            for event in existing:
                if event["key"] == key:
                    return event
            event = {
                "id": max((e["id"] for e in existing), default=0) + 1,
                "ts": _utc_now(),
                "type": event_type,
                "key": key,
                "data": data,
            }
            _append_durably(self.path, json.dumps(event, separators=(",", ":")) + "\n")
            return event

    def tail(self) -> "LogTail":
        return LogTail(self.path)


class LogTail:
    """Incremental lock-free reader that only ever consumes complete lines."""

    def __init__(self, path: Path):
        self._path = path
        self._offset = 0
        self._inode: int | None = None

    def read(self) -> tuple[list[dict[str, Any]], bool]:
        """Return ``(new events, replaced)``.

        ``replaced`` is true when the file was deleted, recreated or shrunk since
        the last call; the returned events are then read from the start of the new
        file and the caller must treat its position as invalid.
        """
        try:
            stat = os.stat(self._path)
        except FileNotFoundError:
            replaced = self._inode is not None
            self._offset, self._inode = 0, None
            return [], replaced
        replaced = self._inode is not None and (
            stat.st_ino != self._inode or stat.st_size < self._offset
        )
        if replaced:
            self._offset = 0
        self._inode = stat.st_ino
        if stat.st_size == self._offset:
            return [], replaced
        with open(self._path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        complete = chunk.rfind(b"\n") + 1
        self._offset += complete
        return list(_parse_lines(chunk[:complete])), replaced


def append_event(
    directory: Path, event_type: str, key: str, data: BaseModel
) -> dict[str, Any]:
    """Append a typed event; idempotent on ``key``."""
    expected = _PAYLOADS[event_type]
    if not isinstance(data, expected):
        raise TypeError(f"{event_type} needs {expected.__name__}, got {type(data)}")
    return EventLog(directory).append(event_type, key, data.model_dump(mode="json"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- signal.issued ----------------------------------------------------------------


def _signal_identity(path: Path) -> tuple[str, str, str] | None:
    """``(arm, observation_date, key)`` for a dated prediction file, from path + bytes."""
    match = _DATED_PREDICTIONS.match(path.name)
    if match is None:
        return None
    arm, observed = path.parent.name, match.group(1)
    return arm, observed, f"signal.issued:{arm}:{observed}:{file_sha256(path)}"


def _signal_data(path: Path, arm: str, observed: str) -> SignalIssuedData:
    table = pq.read_table(path)
    issued_at = None
    if "issued_at" in table.column_names and table.num_rows:
        issued_at = str(table.column("issued_at")[0].as_py())
    return SignalIssuedData(
        arm=arm,
        observation_date=date.fromisoformat(observed),
        n_predictions=table.num_rows,
        issued_at=issued_at,
    )


def emit_signal_issued(
    path: Path, directory: Path | None = None
) -> dict[str, Any] | None:
    """Log the persisted prediction file at ``path`` (idempotent).

    Returns ``None`` without logging when ``path`` is not a dated prediction file
    directly under ``<live dir>/<arm>/``: ad-hoc output elsewhere is not part of
    the live index, and :func:`reconcile` could not re-derive it.
    """
    directory = Path(directory) if directory is not None else live_dir()
    path = Path(path)
    if path.resolve().parent.parent != directory.resolve():
        return None
    identity = _signal_identity(path)
    if identity is None:
        return None
    arm, observed, key = identity
    return append_event(
        directory, "signal.issued", key, _signal_data(path, arm, observed)
    )


# --- news.ingested ----------------------------------------------------------------


def news_batch_key(pairs: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for ticker, url in sorted(pairs):
        digest.update(f"{ticker}\t{url}\n".encode("utf-8"))
    return f"news.ingested:{digest.hexdigest()}"


def prepare_news_batch(
    directory: Path,
    *,
    key: str,
    data: NewsIngestedData,
    output: Path,
    output_sha256: str,
) -> None:
    """Write-ahead record: ``output`` will be atomically replaced by the file with
    ``output_sha256``. Must be durable *before* the rename."""
    record = {
        "key": key,
        "output": str(Path(output).resolve()),
        "output_sha256": output_sha256,
        "prepared_at": _utc_now(),
        "data": data.model_dump(mode="json"),
    }
    log = EventLog(directory)
    with log.locked():
        ledger = log.directory / NEWS_LEDGER_FILE
        _read_complete(ledger)
        _append_durably(ledger, json.dumps(record, separators=(",", ":")) + "\n")


def _committed_news_batches(
    directory: Path, logged: set[str]
) -> Iterator[tuple[str, NewsIngestedData]]:
    """Ledger batches whose recorded output hash equals the file now on disk."""
    path = Path(directory) / NEWS_LEDGER_FILE
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return
    hashes: dict[str, str | None] = {}
    for line in raw[: raw.rfind(b"\n") + 1].split(b"\n"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            key = record["key"]
            output = record["output"]
            expected = record["output_sha256"]
            data = NewsIngestedData.model_validate(record["data"])
        except (ValueError, KeyError, TypeError):
            LOGGER.warning("events: skipping malformed news ledger record")
            continue
        if key in logged:
            continue
        if output not in hashes:
            target = Path(output)
            hashes[output] = file_sha256(target) if target.is_file() else None
        if hashes[output] == expected:
            yield key, data


# --- job.status -------------------------------------------------------------------


def record_last_run(
    directory: Path | None = None,
    *,
    exit_code: int,
    failed_step: str | None,
) -> dict[str, Any]:
    """Write ``last_run.json`` atomically, then log the matching ``job.status``.

    The log append is best effort: the file is the source of truth and
    :func:`reconcile` re-derives the event, so a log failure must not mask the
    job's real exit status.
    """
    directory = Path(directory) if directory is not None else live_dir()
    directory.mkdir(parents=True, exist_ok=True)
    status = {
        "finished_at": _utc_now(),
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "failed_step": None if exit_code == 0 else failed_step,
    }
    _atomic_write_text(
        directory / LAST_RUN_FILE,
        json.dumps(status, ensure_ascii=False, indent=2),
    )
    try:
        append_event(
            directory,
            "job.status",
            f"job.status:{status['finished_at']}",
            JobStatusData(**status),
        )
    except Exception:
        LOGGER.exception("events: job.status append failed; reconcile will recover it")
    return status


def _job_status(directory: Path) -> tuple[str, JobStatusData] | None:
    try:
        payload = json.loads((Path(directory) / LAST_RUN_FILE).read_text("utf-8"))
        data = JobStatusData.model_validate(payload)
    except (OSError, ValueError):
        return None
    return f"job.status:{data.finished_at}", data


# --- reconcile --------------------------------------------------------------------


def reconcile(directory: Path | None = None) -> list[dict[str, Any]]:
    """Append every committed-but-unlogged event; return their log entries.

    Idempotent: each event key is derived from persisted artifacts, and
    appending a logged key is a no-op. Events are appended oldest first
    (prediction files by observation date, then news batches, then job status).
    """
    directory = Path(directory) if directory is not None else live_dir()
    if not directory.is_dir():
        return []
    logged = EventLog(directory).keys()
    appended: list[dict[str, Any]] = []

    def add(event_type: str, key: str, data: BaseModel) -> None:
        appended.append(append_event(directory, event_type, key, data))
        logged.add(key)

    candidates = []
    for path in directory.glob("*/predictions_*.parquet"):
        try:
            identity = _signal_identity(path)
        except OSError:
            continue  # vanished between glob and hash
        if identity is not None and identity[2] not in logged:
            candidates.append((identity[1], identity[0], identity[2], path))
    for observed, arm, key, path in sorted(candidates):
        try:
            data = _signal_data(path, arm, observed)
        except (OSError, ValueError):
            LOGGER.warning("events: unreadable prediction file %s; skipped", path)
            continue
        add("signal.issued", key, data)

    for key, data in list(_committed_news_batches(directory, logged)):
        add("news.ingested", key, data)

    job = _job_status(directory)
    if job is not None and job[0] not in logged:
        add("job.status", job[0], job[1])
    return appended
