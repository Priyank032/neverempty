"""Defensive paths in the matching primitives.

Every branch here is reachable from a real dataset or a real trace, and every
one of them must fail as a *verdict* rather than as an exception. A scorer that
raises on a malformed expectation marks the case unscored, which the gate reads
as a broken run: a dataset typo would then look like a harness failure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from neverempty import scorers
from neverempty.dataset.case import Case
from neverempty.scorers.match import as_datetime, as_member_list, as_number, normalize
from neverempty.scorers.reading import recorded_args

from ._scoring import tool_span, trace


def case(**args: Any) -> Case:
    return Case.model_validate(
        {
            "id": "nr-args-0002",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {
                "tool_calls": {
                    "mode": "first",
                    "calls": [{"tool": "search_jobs", "args": args}],
                }
            },
        }
    )


def called(**args: Any) -> Any:
    return trace(spans=[tool_span("search_jobs", start_ns=1_000, args=json.dumps(args))])


class TestNumberParsing:
    def test_an_arbitrary_object_is_not_a_number(self) -> None:
        assert as_number({"value": 3}) is None

    def test_nan_is_not_a_number_for_comparison(self) -> None:
        """NaN compares unequal to itself, so admitting it would make a numeric
        match nondeterministic in a way no dataset could express."""
        assert as_number(float("nan")) is None

    def test_a_non_numeric_expected_value_fails_the_argument(self) -> None:
        """A dataset declaring ``match="numeric"`` with a text value is an
        authoring bug, but it must produce a verdict, not an exception."""
        verdict = scorers.arguments().score(
            case(years={"value": "many", "match": "numeric"}), called(years=3)
        )
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["arguments"]["years"]["note"] == "expected value is not numeric"


class TestDateParsing:
    def test_an_arbitrary_object_is_not_a_date(self) -> None:
        assert as_datetime(42) is None

    def test_an_already_aware_datetime_passes_through(self) -> None:
        aware = datetime(2026, 9, 1, tzinfo=timezone.utc)
        assert as_datetime(aware) == aware

    def test_a_non_date_expected_value_fails_the_argument(self) -> None:
        verdict = scorers.arguments().score(
            case(posted_after={"value": "someday", "match": "date"}),
            called(posted_after="2026-09-01"),
        )
        assert verdict is not None
        assert verdict.passed is False
        assert verdict.detail["arguments"]["posted_after"]["note"] == "expected value is not a date"


class TestMemberLists:
    def test_a_dict_is_not_a_member_list(self) -> None:
        assert as_member_list({"a": 1}) is None

    def test_a_tuple_is_a_member_list(self) -> None:
        assert as_member_list(("a", "b")) == ["a", "b"]

    def test_nested_lists_compare_order_free(self) -> None:
        verdict = scorers.arguments().score(
            case(groups={"value": [["a", "b"], ["c"]], "match": "set"}),
            called(groups=[["c"], ["b", "a"]]),
        )
        assert verdict is not None
        assert verdict.passed is True


class TestNormalization:
    def test_a_non_string_is_stringified(self) -> None:
        assert normalize(42) == "42"

    def test_none_normalises_to_its_repr_not_to_empty(self) -> None:
        """Silently mapping ``None`` to ``""`` would make a null argument match
        an empty-string expectation, which is the missing-versus-empty bug."""
        assert normalize(None) == "none"


class TestRecordedArgs:
    def test_a_non_string_args_attribute_is_unreadable(self) -> None:
        """An adapter that wrote a number there gave us nothing to parse."""
        span = tool_span("search_jobs", start_ns=1_000, attrs={"tool.args": 42})
        assert recorded_args(span) is None

    def test_a_json_array_is_not_an_argument_map(self) -> None:
        span = tool_span("search_jobs", start_ns=1_000, args="[1, 2, 3]")
        assert recorded_args(span) is None


class TestUnreadableAnswers:
    def test_an_answer_with_no_letters_is_not_treated_as_unreadable(self) -> None:
        """Punctuation and digits alone cannot indicate a script, so the answer
        is read as a covered-script answer that matched nothing: ``ignored``."""
        faulted = trace(
            spans=[tool_span("search_jobs", start_ns=1_000, status="error", error_kind="timeout")],
            answer="--- 42 ---",
        )
        fault = Case.model_validate(
            {
                "id": "nr-fault-0002",
                "suite": "nextrole.failure",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "jobs?"}]},
                "faults": [{"tool": "search_jobs", "kind": "timeout", "after_calls": 0}],
                "expect": {"route": {"label": "job_search"}},
            }
        )
        verdict = scorers.failure_handling().score(fault, faulted)
        assert verdict is not None
        assert verdict.detail["outcome"] == "ignored"
