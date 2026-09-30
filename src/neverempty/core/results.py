"""The tagged union a tool returns: ``Ok``, ``Empty`` or ``Err``.

Three properties matter here, and each one is a bug this library exists to
prevent:

1. Failure is never representable as empty. ``Err`` is a distinct variant, and
   its model-facing rendering says so in words.
2. Falsiness is never inferred. ``0``, ``False`` and ``""`` are legitimate
   ``Ok`` values; only an explicit predicate can produce ``Empty``.
3. The rendering the model reads is part of the contract, not a detail. A
   perfectly typed error that serialises to ``[]`` in a tool message
   reproduces the bug in full.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeAliasType

T = TypeVar("T")

ErrorKind: TypeAlias = Literal[
    "timeout",
    "upstream",
    "validation",
    "permission",
    "rate_limit",
    "exception",
]
"""How a tool failed. Fixed set: scorers and the gate key on these."""

EMPTY_NOTE = "The query succeeded and returned no matching records."

ERROR_NOTE = (
    "The tool failed. You do not know whether matching data exists. Do not say that no data exists."
)
"""The sentence the whole library is built around.

An error must never render as something a model can read as absence. This note
is deliberately blunt, and the fault-injection suite measures whether it works
rather than assuming it does.
"""

TRUNCATED_NOTE = (
    "This result is incomplete: more matching records exist than are shown. "
    "Do not state a total count from this data."
)


class _Result(BaseModel):
    """Shared configuration. Results are frozen, so an ``Err`` can never be
    mutated into an ``Ok`` after the fact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fault_injected: bool = Field(
        default=False,
        exclude=True,
        description=(
            "True when a declared fault produced this result instead of the real "
            "dependency. Excluded from serialization: the tracer records it as the "
            "``tool.fault_injected`` span attribute, and it is never shown to a model."
        ),
    )

    def to_json(self) -> str:
        """Serialize for a sink or a trace. Round-trips exactly."""
        return self.model_dump_json()

    def to_model(self) -> str:
        """Render what the model actually reads in the tool message."""
        raise NotImplementedError  # pragma: no cover


class Ok(_Result, Generic[T]):
    """The tool ran and produced a value.

    The value may be falsy. ``Ok(value=0)`` and ``Ok(value=[])`` are ordinary
    successes; whether an empty collection means "no matches" is a question
    only the tool's author can answer, which is what ``empty_when`` is for.
    """

    status: Literal["ok"] = "ok"
    value: T
    truncated: bool = False
    """The result was capped. Covers the sibling bug to the headline one: 100
    rows out of a larger set, reported by the model as "there are 100 records"."""

    def to_model(self) -> str:
        payload: dict[str, Any] = {
            "status": "ok",
            "data": self.value,
            "truncated": self.truncated,
        }
        if self.truncated:
            payload["note"] = TRUNCATED_NOTE
        try:
            return json.dumps(payload, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            # No ``default=str``. Coercing an unserializable value to its repr
            # sent the model ``"<app.Row object at 0x7f...>"`` under
            # ``status: ok`` -- a non-result wearing a success label, which is
            # the bug this library exists to prevent, one layer down. A value
            # JSON cannot represent is a failure of the tool, so it renders as
            # one, and the note still forbids claiming that no data exists.
            return Err(
                kind="validation",
                message="tool result is not JSON-serializable",
                retryable=False,
                cause="TypeError",
            ).to_model()


class Empty(_Result):
    """The tool ran, succeeded, and there was genuinely nothing to return.

    This is a positive claim about the world, which is exactly why it must
    never be produced by inference from a falsy value.
    """

    status: Literal["empty"] = "empty"
    reason: str | None = None

    def to_model(self) -> str:
        payload: dict[str, Any] = {"status": "empty", "note": EMPTY_NOTE}
        if self.reason is not None:
            payload["reason"] = self.reason
        return json.dumps(payload, ensure_ascii=False)


class Err(_Result):
    """The tool failed. Nothing is known about whether data exists."""

    status: Literal["error"] = "error"
    kind: ErrorKind
    message: str
    retryable: bool
    cause: str | None = None
    """The exception class name, never the traceback text: tracebacks carry
    file paths and sometimes argument values."""

    def to_model(self) -> str:
        """Render status, kind and the refusal note, and nothing else.

        ``message`` and ``cause`` stay in the trace for humans. Putting
        upstream error text into a prompt leaks internal detail and gives the
        model something to paraphrase into a claim about the data.
        """
        return json.dumps(
            {"status": "error", "kind": self.kind, "note": ERROR_NOTE},
            ensure_ascii=False,
        )


ToolResult = TypeAliasType("ToolResult", Ok[T] | Empty | Err, type_params=(T,))
"""What every ``@tool`` returns. Subscript it with the ``Ok`` payload type::

    async def query(...) -> ToolResult[list[dict]]: ...

``status`` is the discriminator. A plain ``TypeAlias`` would not work here: the
union eagerly evaluates and loses its type parameter, so ``ToolResult[int]``
would fail at runtime. ``typing_extensions`` is a hard pydantic dependency, so
this costs nothing at install time.

Where a pydantic model embeds a result, annotate the field with
:data:`AnyToolResult` so pydantic builds a tagged union rather than trying each
variant in turn.
"""

AnyToolResult: TypeAlias = Annotated[Ok[Any] | Empty | Err, Field(discriminator="status")]
"""``ToolResult`` as a pydantic field annotation, tagged on ``status``.

``Annotated`` erases the type parameter, so this is a separate name from
:data:`ToolResult` rather than the same one."""


__all__ = [
    "EMPTY_NOTE",
    "ERROR_NOTE",
    "TRUNCATED_NOTE",
    "AnyToolResult",
    "Empty",
    "Err",
    "ErrorKind",
    "Ok",
    "ToolResult",
]
