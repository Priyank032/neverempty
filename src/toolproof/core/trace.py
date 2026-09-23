"""Trace v1: the contract every downstream layer reads.

Nothing below the trace touches an agent, which is why an agent written in
another language is a first-class citizen rather than a special case. The
schema is generated from these models and committed, so a Node exporter can
validate its own output in its own test suite.

Two strictness rules pull in opposite directions, deliberately:

- Unknown keys in a *trace* are preserved. Other languages and later versions
  write fields this reader does not know, and dropping them silently would make
  the contract one-way.
- A ``schema_version`` above this reader's is rejected outright. Preserving
  unknown keys is forward compatibility; pretending to understand a newer
  schema is data corruption.
"""

from __future__ import annotations

import re
import uuid
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = 1
"""Trace schema version. Independent of the package version."""

ERROR_MESSAGE_CAP_BYTES = 2048
"""Error messages are capped so a stack-dumping target cannot bloat a report."""

TOOL_ARGS_CAP_BYTES = 8192
"""Tool arguments are capped for the same reason."""

_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_SPAN_ID = re.compile(r"^s\d{4,}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")

SpanKind: TypeAlias = Literal["llm", "tool", "node", "judge", "custom"]
SpanStatus: TypeAlias = Literal["ok", "empty", "error"]
TraceStatus: TypeAlias = Literal["ok", "target_error", "timeout", "budget_abort"]
RunMode: TypeAlias = Literal["live", "replay"]

AttributeValue: TypeAlias = str | int | float | bool | None
"""Span attributes are a flat map. Nesting would make cross-language readers
and OTel exporters disagree about how to flatten it."""


def truncate_utf8(text: str, cap_bytes: int, marker: str = " [truncated]") -> str:
    """Cap ``text`` at ``cap_bytes`` without splitting a codepoint.

    The marker is part of the budget, so the result always fits. Truncation is
    always visible: a silently shortened message reads as the whole message.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= cap_bytes:
        return text
    budget = cap_bytes - len(marker.encode("utf-8"))
    return encoded[: max(budget, 0)].decode("utf-8", errors="ignore") + marker


class _TraceModel(BaseModel):
    """Forward compatible: unknown keys are kept and re-serialised."""

    model_config = ConfigDict(extra="allow")


class Usage(_TraceModel):
    """Token counts, from provider usage fields only.

    Never a local tokenizer estimate: an estimate that looks like a measurement
    is the same class of bug as an error that looks like an empty result.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)

    @property
    def usage_missing(self) -> bool:
        """True when no count was reported at all.

        A reported ``0`` is a measurement and leaves this false. That
        distinction is the whole point of the flag.
        """
        return self.input_tokens is None and self.output_tokens is None

    @property
    def complete(self) -> bool:
        """True when both halves were reported, so a cost can be computed."""
        return self.input_tokens is not None and self.output_tokens is not None

    def model_post_init(self, _context: object, /) -> None:
        # Surface the flag in serialized output too: a Node reader should not
        # have to reimplement the rule from the null pattern.
        extra = self.__pydantic_extra__
        if extra is None:
            extra = {}
            object.__setattr__(self, "__pydantic_extra__", extra)
        extra["usage_missing"] = self.usage_missing


class Cost(_TraceModel):
    """Cost in USD, or an explicit null with the reason it is unknown.

    An unknown cost is never ``0``. A zero would be indistinguishable from a
    free call, and a report summing zeros would understate a run's real cost
    while looking perfectly healthy.
    """

    usd: float | None = Field(default=None, ge=0)
    pricing_version: str
    unknown_reason: str | None = None
    unknown_models: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _default_reason_only_when_unknown(cls, data: Any) -> Any:
        """Supply a reason for a null cost, but never for a measured one.

        A default on the field itself would reject ``Cost(usd=0.0)``, and a
        genuinely free call has to stay expressible: zero is only wrong when it
        stands in for missing.
        """
        if isinstance(data, dict) and data.get("usd") is None and "unknown_reason" not in data:
            data = {**data, "unknown_reason": "usage_missing"}
        return data

    @model_validator(mode="after")
    def _reason_must_agree_with_value(self) -> Cost:
        if self.usd is None and not self.unknown_reason:
            raise ValueError(
                "unknown_reason is required when usd is null: a missing cost must "
                "always say why it is missing"
            )
        if self.usd is not None and self.unknown_reason:
            raise ValueError(
                f"unknown_reason={self.unknown_reason!r} was given alongside a known "
                f"cost of {self.usd}; a cost is either known or explained, never both"
            )
        return self


class FinalOutput(_TraceModel):
    """What the agent produced. Every field is nullable and means it."""

    answer: str | None = None
    route: str | None = None
    structured: dict[str, Any] | None = None


class TraceError(_TraceModel):
    """How a run failed, with the message capped and the cap made visible."""

    kind: str
    message: str

    @field_validator("message")
    @classmethod
    def _cap(cls, value: str) -> str:
        return truncate_utf8(value, ERROR_MESSAGE_CAP_BYTES)


class Env(_TraceModel):
    """The reproducibility block.

    A report without the target's git sha, prompt hashes and the resolved model
    id is rejected by ``compare`` and by the README renderer. That check is
    only possible because these fields travel with the trace.
    """

    toolproof_version: str
    target_git_sha: str | None = None
    target_dirty: bool = False
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    resolved_models: list[str] = Field(default_factory=list)
    pricing_version: str
    python_version: str
    runtime: str | None = None
    concurrency: int = Field(default=1, ge=1)
    seed: int | None = None
    mode: RunMode = "live"
    fault_profile: str | None = None
    rate_limit_retries: int = Field(default=0, ge=0)

    @field_validator("target_git_sha")
    @classmethod
    def _sha_shape(cls, value: str | None) -> str | None:
        if value is not None and not _GIT_SHA.match(value):
            raise ValueError(f"target_git_sha must be hex, got {value!r}")
        return value

    @field_validator("prompt_hashes")
    @classmethod
    def _hashes_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            if not _SHA256_HEX.match(digest):
                raise ValueError(
                    f"prompt_hashes[{name!r}] must be lowercase SHA-256 hex, got {digest!r}"
                )
        return value


class Span(_TraceModel):
    """One unit of work. Attribute names are fixed; see the doc's tables."""

    span_id: str
    parent_id: str | None = None
    kind: SpanKind
    name: str
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=0)
    status: SpanStatus = "ok"
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)

    @field_validator("span_id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _SPAN_ID.match(value):
            raise ValueError(f"span_id must look like 's0001', got {value!r}")
        return value

    @field_validator("end_ns")
    @classmethod
    def _end_after_start(cls, value: int, info: ValidationInfo) -> int:
        start = info.data.get("start_ns")
        if isinstance(start, int) and value < start:
            raise ValueError(f"end_ns ({value}) precedes start_ns ({start})")
        return value

    @model_validator(mode="after")
    def _empty_is_a_tool_only_status(self) -> Span:
        if self.status == "empty" and self.kind != "tool":
            raise ValueError(
                f"status 'empty' is only meaningful on a tool span, not on a "
                f"{self.kind!r} span: an LLM call has no empty outcome"
            )
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


class Trace(_TraceModel):
    """One agent run, for one case and one repeat."""

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        json_schema_extra={"const": SCHEMA_VERSION},
    )
    trace_id: str
    case_id: str | None = None
    repeat: int = Field(default=0, ge=0)
    suite: str | None = None
    started_at: str
    duration_ms: int = Field(ge=0)
    spans: list[Span] = Field(default_factory=list)
    final_output: FinalOutput
    usage: Usage
    cost: Cost
    status: TraceStatus = "ok"
    error: TraceError | None = None
    env: Env
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def _version_is_readable(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"schema_version must be >= 1, got {value}")
        if value > SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {value} is newer than this reader understands "
                f"({SCHEMA_VERSION}). Upgrade toolproof to read this trace."
            )
        return value

    @field_validator("trace_id")
    @classmethod
    def _trace_id_is_a_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as exc:
            raise ValueError(f"trace_id must be a UUID, got {value!r}") from exc
        return value

    @field_validator("started_at")
    @classmethod
    def _timestamp_is_rfc3339_utc(cls, value: str) -> str:
        if not _RFC3339_UTC.match(value):
            raise ValueError(f"started_at must be RFC 3339 UTC ending in 'Z', got {value!r}")
        return value

    @field_validator("spans")
    @classmethod
    def _ordered_by_start(cls, value: list[Span]) -> list[Span]:
        """Ordered on write, so two runs of a suite produce diffable JSON."""
        return sorted(value, key=lambda span: (span.start_ns, span.span_id))


Attributes: TypeAlias = Annotated[dict[str, AttributeValue], Field(default_factory=dict)]


__all__ = [
    "ERROR_MESSAGE_CAP_BYTES",
    "SCHEMA_VERSION",
    "TOOL_ARGS_CAP_BYTES",
    "AttributeValue",
    "Cost",
    "Env",
    "FinalOutput",
    "RunMode",
    "Span",
    "SpanKind",
    "SpanStatus",
    "Trace",
    "TraceError",
    "TraceStatus",
    "Usage",
    "truncate_utf8",
]
