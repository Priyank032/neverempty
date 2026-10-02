"""``empty_payload``: an absence claimed in data rather than in words.

Found instrumenting NextRole. Its ``search_jobs`` catches every exception and
returns::

    {"response": "I encountered an issue while searching for jobs. ...",
     "jobs": [], "total_count": 0}

The prose is honest -- ``failure_handling`` scores it ``reported`` -- and the
structured output is not. ``jobs: []`` with ``total_count: 0`` after a query
that never ran is a machine-readable claim that zero jobs exist, and it is the
more dangerous half: prose is read by a human who might notice hedging,
``total_count: 0`` is read by code that will not.

So this is the headline bug one layer down, and no existing scorer could see
it. ``failure_handling`` reads the answer; this reads the payload. They
partition rather than overlap, and a case can fail this while passing that --
which is exactly what NextRole does today.

Like ``failure_handling``, this only applies when a tool actually failed:
``[]`` after a successful search is a true answer, not a misreport.
"""

from __future__ import annotations

from typing import Any

import pytest

from neverempty import ScorerError, scorers
from neverempty.dataset.case import Case

from ._scoring import tool_span, trace


def fault_case(*, tool: str = "search_jobs", kind: str = "timeout") -> Case:
    return Case.model_validate(
        {
            "id": "nr-fault-0001",
            "suite": "nextrole.failure",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "any backend jobs?"}]},
            "faults": [{"tool": tool, "kind": kind, "after_calls": 0}],
            "expect": {"route": {"label": "job_search"}},
        }
    )


def faulted(structured: dict[str, Any] | None, *, answer: str = "Something went wrong.") -> Any:
    return trace(
        spans=[
            tool_span(
                "search_jobs",
                start_ns=1_000,
                status="error",
                error_kind="timeout",
                fault_injected=True,
            )
        ],
        answer=answer,
        structured=structured,
    )


def score(structured: dict[str, Any] | None, **kwargs: Any) -> Any:
    return scorers.empty_payload().score(fault_case(), faulted(structured, **kwargs))


class TestAnEmptyCollectionAfterAFailureIsAMisreport:
    def test_an_empty_list_fails(self) -> None:
        """The NextRole case, exactly."""
        verdict = score({"jobs": [], "total_count": 0})
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["outcome"] == "empty_payload"

    def test_the_offending_fields_are_named(self) -> None:
        """A report has to say which field claimed the absence."""
        verdict = score({"jobs": [], "articles": [], "total_count": 0})
        assert verdict is not None
        # ``total_count`` is a claim too, which is the point of COUNT_SUFFIXES.
        assert verdict.detail["empty_fields"] == ["articles", "jobs", "total_count"]

    def test_a_zero_count_alone_fails(self) -> None:
        """``total_count: 0`` is the same claim without the list."""
        verdict = score({"total_count": 0})
        assert verdict is not None
        assert verdict.passed is False

    def test_an_empty_dict_of_results_fails(self) -> None:
        verdict = score({"results": {}})
        assert verdict is not None
        assert verdict.passed is False

    def test_it_fails_even_when_the_prose_is_honest(self) -> None:
        """The whole point: ``failure_handling`` passes this answer."""
        answer = "I encountered an issue while searching. Please try again."
        prose = scorers.failure_handling().score(
            fault_case(), faulted({"jobs": [], "total_count": 0}, answer=answer)
        )
        assert prose is not None
        assert prose.passed is True
        verdict = score({"jobs": [], "total_count": 0}, answer=answer)
        assert verdict is not None
        assert verdict.passed is False


class TestAnHonestPayloadPasses:
    def test_an_error_field_passes(self) -> None:
        """Saying so in the data is the correct behaviour."""
        verdict = score({"error": "timeout", "jobs": None})
        assert verdict is not None
        assert verdict.passed is True

    def test_a_null_result_passes(self) -> None:
        """``None`` is "unknown"; ``[]`` is "none exist". Only the second lies."""
        verdict = score({"jobs": None, "total_count": None})
        assert verdict is not None
        assert verdict.passed is True

    def test_a_populated_result_passes(self) -> None:
        """A fault on one tool need not empty every result."""
        verdict = score({"jobs": [{"title": "Backend Engineer"}], "total_count": 1})
        assert verdict is not None
        assert verdict.passed is True

    def test_unrelated_zero_valued_fields_are_ignored(self) -> None:
        """``retries: 0`` is not a claim about the world."""
        verdict = score({"retries": 0, "latency_ms": 0, "jobs": [{"t": "x"}]})
        assert verdict is not None
        assert verdict.passed is True


class TestItOnlyAppliesWhenATooFailed:
    def test_no_failed_tool_is_not_applicable(self) -> None:
        """``[]`` after a successful search is a true answer."""
        subject = trace(
            spans=[tool_span("search_jobs", start_ns=1_000, status="ok")],
            answer="No jobs found.",
            structured={"jobs": [], "total_count": 0},
        )
        assert scorers.empty_payload().score(fault_case(), subject) is None

    def test_no_structured_output_is_not_applicable(self) -> None:
        """An agent that returns only prose is scored by failure_handling."""
        assert score(None) is None

    def test_an_empty_structured_output_is_not_applicable(self) -> None:
        """Nothing declared is not the same as a declared emptiness."""
        assert score({}) is None


class TestItRefusesToGuess:
    def test_a_non_dict_structured_output_raises(self) -> None:
        subject = trace(
            spans=[
                tool_span(
                    "search_jobs",
                    start_ns=1_000,
                    status="error",
                    error_kind="timeout",
                    fault_injected=True,
                )
            ],
            answer="x",
        )
        object.__setattr__(subject.final_output, "structured", ["not", "a", "dict"])
        with pytest.raises(ScorerError):
            scorers.empty_payload().score(fault_case(), subject)
