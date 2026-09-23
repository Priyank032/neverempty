"""Case v1: one dataset line.

Every field in ``expect`` is optional, so one schema covers routing-only
suites, full trajectories and generated consistency suites. A scorer whose
expectation is absent reports "not applicable", never "pass" — which is only
possible because absent and empty are distinguishable here.

Strictness is the opposite of the trace's: an unknown top-level key is a
validation error. A typo in eval data is a silent killer, because a misspelled
expectation is an absent one, and an absent expectation cannot fail.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from toolproof.core.faults import FaultSpec

CASE_SCHEMA_VERSION = 1

_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_SUITE = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")

Split: TypeAlias = Literal["dev", "test"]
ArgMatch: TypeAlias = Literal["exact", "normalized", "set", "numeric", "regex", "present", "date"]
"""Argument match modes.

There is deliberately no ``judge`` mode: if an argument needs semantic
matching, the dataset is underspecified, and a judge would hide that.
"""

FactMatch: TypeAlias = Literal["contains", "regex", "judge"]
ToolCallMode: TypeAlias = Literal["first", "set", "sequence"]
FaultKind: TypeAlias = Literal["timeout", "upstream", "rate_limit", "empty", "truncated"]
LabelMethod: TypeAlias = Literal["human", "llm_drafted_human_verified", "generated_from_rules"]


class _CaseModel(BaseModel):
    """Strict by default: unknown keys are authoring bugs, not extensions."""

    model_config = ConfigDict(extra="forbid")


class Message(_CaseModel):
    """One chat turn."""

    role: str = Field(min_length=1)
    content: str


class CaseInput(_CaseModel):
    """Exactly one of ``messages`` (chat agents) or ``payload`` (everything else)."""

    messages: list[Message] | None = None
    payload: dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, value: list[Message] | None) -> list[Message] | None:
        if value is not None and not value:
            raise ValueError("messages must not be empty when present")
        return value

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> CaseInput:
        present = [name for name in ("messages", "payload") if getattr(self, name) is not None]
        if len(present) != 1:
            raise ValueError(
                f"input must carry exactly one of 'messages' or 'payload', got "
                f"{present or 'neither'}"
            )
        return self


class RouteExpectation(_CaseModel):
    """The branch the agent should take.

    Labels are free strings. The library never knows a target's branch names,
    so adding one later is a dataset edit; the target adapter asserts at import
    time that every branch exists in the compiled graph, which is where a typo
    can actually be caught.
    """

    label: str = Field(min_length=1)
    acceptable: list[str] = Field(default_factory=list)
    """Alternatives that count for lenient accuracy but not strict."""

    @model_validator(mode="after")
    def _acceptable_excludes_the_label(self) -> RouteExpectation:
        if self.label in self.acceptable:
            raise ValueError(
                f"acceptable must not contain the label {self.label!r}: strict and "
                f"lenient accuracy would then be the same number"
            )
        return self


class ArgExpectation(_CaseModel):
    """One expected argument, under one match mode."""

    value: Any = None
    match: ArgMatch = "exact"
    tol: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> ArgExpectation:
        if self.tol is not None and self.match != "numeric":
            raise ValueError(f"tol is only meaningful for match='numeric', not {self.match!r}")
        if self.match == "regex":
            _compile_or_raise(str(self.value), field="regex")
        return self


class ExpectedCall(_CaseModel):
    """One expected tool call."""

    tool: str = Field(min_length=1)
    args: dict[str, ArgExpectation] = Field(default_factory=dict)


class ToolCallExpectation(_CaseModel):
    """Which tools should be called, and how strictly the order matters."""

    mode: ToolCallMode
    calls: list[ExpectedCall] = Field(min_length=1)


class Fact(_CaseModel):
    """A claim the answer should (or must not) contain."""

    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    match: FactMatch
    evidence_key: str | None = None

    @model_validator(mode="after")
    def _regex_compiles(self) -> Fact:
        if self.match == "regex":
            _compile_or_raise(self.statement, field="regex")
        return self


class RuleStep(_CaseModel):
    """One criterion from a rule engine's trace."""

    criterion: str = Field(min_length=1)
    result: bool | None = None
    missing_field: str | None = None


class ItemExpectation(_CaseModel):
    """Per-item ground truth for generated suites.

    ``rule_result`` is a true three-way value: ``None`` means the rule could
    not evaluate, which is a different claim from "ineligible" and is the case
    most likely to catch an agent overclaiming.
    """

    item_id: str = Field(min_length=1)
    rule_result: bool | None = None
    rule_trace: list[RuleStep] = Field(default_factory=list)


class FaultDecl(_CaseModel):
    """A fault to inject for this case."""

    tool: str = Field(min_length=1)
    kind: FaultKind
    after_calls: int = Field(default=0, ge=0)


class Expect(_CaseModel):
    """What the case asserts. Every key optional, at least one present.

    ``None`` means "not asked" and an empty list means "asked for none". A
    scorer keys on that difference to report not-applicable rather than pass.
    """

    route: RouteExpectation | None = None
    tool_calls: ToolCallExpectation | None = None
    forbidden_tools: list[str] | None = None
    facts: list[Fact] | None = None
    forbidden_claims: list[Fact] | None = None
    items: list[ItemExpectation] | None = None

    @model_validator(mode="after")
    def _at_least_one_expectation(self) -> Expect:
        if not any(
            getattr(self, name) is not None
            for name in (
                "route",
                "tool_calls",
                "forbidden_tools",
                "facts",
                "forbidden_claims",
                "items",
            )
        ):
            raise ValueError(
                "expect must declare at least one expectation: a case that expects "
                "nothing can never fail, so it measures nothing"
            )
        _unique_ids(self.facts, "facts", "id")
        _unique_ids(self.forbidden_claims, "forbidden_claims", "id")
        _unique_ids(self.items, "items", "item_id")
        return self


class Provenance(_CaseModel):
    """Where this case's ground truth came from.

    Required on the test split: a published number has to be able to say who
    decided the right answer, and when.
    """

    method: LabelMethod
    labeller: str | None = None
    labelled_at: str | None = None
    note: str | None = None
    source_commit: str | None = None
    scheme_file_sha256: str | None = None

    @model_validator(mode="after")
    def _generated_truth_is_pinned(self) -> Provenance:
        if self.method == "generated_from_rules" and not self.source_commit:
            raise ValueError(
                "source_commit is required for method='generated_from_rules': "
                "generated ground truth drifts when its source moves, and that is "
                "only detectable if the source is pinned"
            )
        if self.source_commit and not _GIT_SHA.match(self.source_commit):
            raise ValueError(f"source_commit must be a git sha, got {self.source_commit!r}")
        if self.scheme_file_sha256 and not _SHA256_HEX.match(self.scheme_file_sha256):
            raise ValueError("scheme_file_sha256 must be lowercase SHA-256 hex")
        return self


class Case(_CaseModel):
    """One dataset line."""

    schema_version: int = Field(
        default=CASE_SCHEMA_VERSION,
        json_schema_extra={"const": CASE_SCHEMA_VERSION},
    )
    id: str
    suite: str
    split: Split
    input: CaseInput
    expect: Expect
    faults: list[FaultDecl] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    must_pass: bool = False
    provenance: Provenance | None = None

    @field_validator("schema_version")
    @classmethod
    def _version_is_readable(cls, value: int) -> int:
        if value != CASE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {value} is not readable by this version "
                f"(expected {CASE_SCHEMA_VERSION})"
            )
        return value

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _CASE_ID.match(value):
            raise ValueError(
                f"id must match {_CASE_ID.pattern} (lowercase, 3 to 64 chars), got {value!r}"
            )
        return value

    @field_validator("suite")
    @classmethod
    def _suite_shape(cls, value: str) -> str:
        if not _SUITE.match(value):
            raise ValueError(
                f"suite must be dotted lowercase like 'nextrole.routing', got {value!r}"
            )
        return value

    @field_validator("tags")
    @classmethod
    def _tags_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError(f"tags must be unique, got {value!r}")
        return value

    @model_validator(mode="after")
    def _test_split_is_attributable(self) -> Case:
        if self.split == "test" and self.provenance is None:
            raise ValueError(
                "provenance is required on the test split: a published number has "
                "to say where its ground truth came from"
            )
        return self

    def fault_specs(self) -> list[FaultSpec]:
        """The declared faults, as the runtime specs the wrapper applies."""
        return [
            FaultSpec(tool=fault.tool, kind=fault.kind, after_calls=fault.after_calls)
            for fault in self.faults
        ]

    def canonical_json(self) -> str:
        """Stable serialization for hashing.

        Sorted keys and no insignificant whitespace, so reformatting a dataset
        file does not read as editing it.
        """
        return self.model_dump_json(exclude_none=False)


def _compile_or_raise(pattern: str, *, field: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(
            f"{field} pattern does not compile ({exc}); a bad pattern must fail "
            f"on the dataset, not halfway through a run"
        ) from exc


def _unique_ids(items: list[Any] | None, field: str, attr: str) -> None:
    if items is None:
        return
    seen = [getattr(item, attr) for item in items]
    if len(set(seen)) != len(seen):
        raise ValueError(f"{field} {attr}s must be unique within a case, got {seen!r}")


AnyCase: TypeAlias = Annotated[Case, Field(description="One dataset line, Case v1")]


__all__ = [
    "CASE_SCHEMA_VERSION",
    "ArgExpectation",
    "ArgMatch",
    "Case",
    "CaseInput",
    "Expect",
    "ExpectedCall",
    "Fact",
    "FactMatch",
    "FaultDecl",
    "ItemExpectation",
    "LabelMethod",
    "Message",
    "Provenance",
    "RouteExpectation",
    "RuleStep",
    "Split",
    "ToolCallExpectation",
    "ToolCallMode",
]
