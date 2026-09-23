"""Record/replay cache for provider responses.

The key covers everything that could change a response, so editing a prompt
changes the key. That is the point: a replay test must not be able to pass
against a recording of the *previous* prompt, because then a prompt regression
would show as green.

In replay mode a miss is a hard error. A silent live call would make a free
deterministic run cost money and stop being deterministic, and the failure
would be invisible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

CacheMode: TypeAlias = Literal["live", "record", "replay"]

KEY_FIELDS: tuple[str, ...] = (
    "provider",
    "model",
    "resolved_model",
    "messages",
    "tool_schema_hash",
    "temperature",
    "top_p",
    "seed",
    "response_format",
    "cache_salt",
)
"""The documented key fields, and only these.

Restricting the key means an unrelated config edit does not invalidate every
recorded response, while any of these changing does.
"""


class CacheMissError(RuntimeError):
    """A replay-mode lookup found nothing, or found something unreadable."""


@dataclass(frozen=True)
class CacheEntry:
    """A recorded response plus the usage the provider reported with it."""

    response: Any
    usage: dict[str, Any] = field(default_factory=dict)
    resolved_model: str | None = None


def cache_key(fields: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the documented key fields."""
    selected = {name: fields.get(name) for name in KEY_FIELDS if name in fields}
    canonical = json.dumps(selected, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """A directory of recorded responses, keyed by content hash.

    Args:
        directory: Where entries live. Created on first write.
        mode: ``live`` never touches disk; ``record`` reads a hit and writes a
            miss; ``replay`` reads only and raises on a miss.
    """

    def __init__(self, directory: str | Path, *, mode: CacheMode = "live") -> None:
        if mode not in ("live", "record", "replay"):
            raise ValueError(f"mode must be live, record or replay, got {mode!r}")
        self.directory = Path(directory)
        self.mode = mode
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        """Sharded by the first two characters, so one directory stays small."""
        return self.directory / key[:2] / f"{key}.json"

    def get(self, fields: dict[str, Any]) -> CacheEntry | None:
        """Look up a recorded response.

        Returns ``None`` on a miss in ``live`` or ``record`` mode. Raises in
        ``replay`` mode, because replay never falls back to a provider.
        """
        if self.mode == "live":
            return None

        key = cache_key(fields)
        path = self.path_for(key)
        if not path.is_file():
            self.misses += 1
            if self.mode == "replay":
                raise CacheMissError(self._miss_message(key, fields))
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.misses += 1
            if self.mode == "replay":
                raise CacheMissError(
                    f"cache entry {key} is unreadable ({exc}). Replay must be exact, "
                    f"so a damaged entry cannot pass as a hit. Re-record it."
                ) from exc
            return None

        self.hits += 1
        return CacheEntry(
            response=payload.get("response"),
            usage=payload.get("usage") or {},
            resolved_model=payload.get("resolved_model"),
        )

    def put(
        self,
        fields: dict[str, Any],
        response: Any,
        *,
        usage: dict[str, Any] | None = None,
        resolved_model: str | None = None,
    ) -> str | None:
        """Record a response. A no-op in ``live`` mode."""
        if self.mode == "live":
            return None

        key = cache_key(fields)
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "key": key,
                    "response": response,
                    "usage": usage or {},
                    "resolved_model": resolved_model,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
            newline="\n",
        )
        return key

    def _miss_message(self, key: str, fields: dict[str, Any]) -> str:
        model = fields.get("model") or "unknown model"
        return (
            f"cache miss in replay mode for key {key} "
            f"(provider={fields.get('provider', 'unknown')}, model={model}). "
            f"Replay never calls a provider, so this is a hard error rather than a "
            f"silent live call. Re-record with mode='record', or check whether a "
            f"prompt changed: editing a prompt changes the key by design."
        )


__all__ = ["KEY_FIELDS", "CacheEntry", "CacheMissError", "CacheMode", "ResponseCache", "cache_key"]
