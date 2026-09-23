"""Where traces go.

A sink must never raise into the agent: a full disk is not an agent bug, and
crashing a production request to record telemetry is worse than losing the
telemetry. But losing it silently is the exact failure this library exists to
prevent, so the Tracer counts every drop and logs it. The counter is public so
a test and an operator can both see the loss.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Protocol

from toolproof.core.trace import Trace


class Sink(Protocol):
    """Anything that can accept a finished trace."""

    def write(self, trace: Trace) -> None:
        """Persist one trace. May raise; the Tracer catches and counts."""
        ...

    def close(self) -> None:
        """Release resources. Must be safe to call more than once."""
        ...


class MemorySink:
    """Keeps traces in a list. For tests and for in-process eval runs."""

    def __init__(self) -> None:
        self.traces: list[Trace] = []
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> None:
        with self._lock:
            self.traces.append(trace)

    def close(self) -> None:
        return None

    def clear(self) -> None:
        with self._lock:
            self.traces.clear()


class JsonlSink:
    """Appends one JSON object per line.

    Given a directory, writes ``traces.jsonl`` inside it. Given a path ending
    in ``.jsonl``, writes that file. Parent directories are created.

    Writes are line-atomic under a lock. The file is opened per write and
    closed again, so an abandoned run leaves no handle open and every completed
    trace is on disk the moment it finishes. That matters because a partial
    report is a documented outcome here, not an accident.

    Pass ``fsync=True`` to also force each line to the physical device. It is
    off by default because an fsync per trace dominates the tracer's own
    latency budget, and it protects against machine loss, not process loss.
    """

    def __init__(
        self, path: str | Path, *, filename: str = "traces.jsonl", fsync: bool = False
    ) -> None:
        target = Path(path)
        self.path = target if target.suffix == ".jsonl" else target / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> None:
        line = trace.model_dump_json()
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())

    def close(self) -> None:
        return None

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MultiSink:
    """Fans out to several sinks. One failing sink does not stop the others."""

    def __init__(self, *sinks: Sink) -> None:
        self.sinks = sinks

    def write(self, trace: Trace) -> None:
        errors: list[BaseException] = []
        for sink in self.sinks:
            try:
                sink.write(trace)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def close(self) -> None:
        for sink in self.sinks:
            with contextlib.suppress(Exception):
                sink.close()


class NullSink:
    """Discards everything. Explicit, so "no sink" is never accidental."""

    def write(self, trace: Trace) -> None:
        return None

    def close(self) -> None:
        return None


__all__ = ["JsonlSink", "MemorySink", "MultiSink", "NullSink", "Sink"]
