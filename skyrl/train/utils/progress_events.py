"""Low-overhead, opt-in structured progress events for long-running jobs.

Set ``SKYRL_PROGRESS_EVENTS_PATH`` to an absolute JSONL path to enable the
writer. ``{hostname}`` and ``{pid}`` placeholders are supported and should be
used for multi-process or multi-node jobs so each process appends to its own
file. A missing setting is a strict no-op so existing jobs keep identical I/O
behavior. Progress reporting is observational and never raises into the
training path.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

PROGRESS_EVENT_SCHEMA_VERSION = 1
_PATH_ENV = "SKYRL_PROGRESS_EVENTS_PATH"
_RUN_ID_ENV = "SKYRL_PROGRESS_RUN_ID"
_lock = threading.Lock()
_disabled_paths: set[str] = set()
_prepared_paths: set[str] = set()
logger = logging.getLogger(__name__)
_UNSUPPORTED = object()
_MAX_SCALAR_STRING_LENGTH = 4096


def _json_scalar(value: Any) -> Any:
    """Return a bounded JSON scalar, or a sentinel for unsupported values."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSUPPORTED
    if isinstance(value, str):
        return value[:_MAX_SCALAR_STRING_LENGTH]
    return _UNSUPPORTED


def _safe_hostname() -> str:
    """Return a filename-safe host identifier."""

    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in socket.gethostname()
    )


def _resolve_event_path(raw_path: str) -> Path:
    resolved = raw_path.replace("{hostname}", _safe_hostname()).replace("{pid}", str(os.getpid()))
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{_PATH_ENV} must resolve to an absolute path")
    return path


def _reset_after_fork() -> None:
    """Do not inherit a possibly locked mutex or parent path cache."""

    global _lock
    _lock = threading.Lock()
    _disabled_paths.clear()
    _prepared_paths.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def emit_progress_event(event: str, /, **payload: Any) -> None:
    """Append one event when the opt-in event path is configured.

    Each event is serialized and appended with one ``os.write`` call under a
    process-local lock. The call does not perform network requests and does not
    retain model, prompt, response, or credential content.
    """

    raw_path = os.environ.get(_PATH_ENV)
    if not raw_path or raw_path in _disabled_paths:
        return
    try:
        path = _resolve_event_path(raw_path)
        record = {
            "schema_version": PROGRESS_EVENT_SCHEMA_VERSION,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "monotonic_ns": time.monotonic_ns(),
            "hostname": _safe_hostname(),
            "pid": os.getpid(),
            "event": event[:_MAX_SCALAR_STRING_LENGTH],
        }
        for key, value in payload.items():
            scalar = _json_scalar(value)
            if scalar is not _UNSUPPORTED:
                record[key] = scalar
        run_id = _json_scalar(os.environ.get(_RUN_ID_ENV))
        if run_id:
            record["run_id"] = run_id
        encoded = (json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        path_key = str(path)
        with _lock:
            if path_key not in _prepared_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                _prepared_paths.add(path_key)
            descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise OSError(f"short progress-event write: {written}/{len(encoded)} bytes")
            finally:
                os.close(descriptor)
    except Exception as exc:  # progress telemetry must not terminate paid work
        _disabled_paths.add(raw_path)
        logger.warning(
            "Disabling structured progress events after write failure at %s: %s: %s",
            raw_path,
            type(exc).__name__,
            exc,
        )


def reset_progress_event_writer_for_tests() -> None:
    """Clear paths disabled by an earlier test-only write failure."""

    with _lock:
        _disabled_paths.clear()
        _prepared_paths.clear()
