"""The arguments scorer.

Acceptance row: "argument match modes including Unicode normalisation on Hindi
... not-applicable never scores as fail".

Seven modes, declared per argument in the dataset: ``exact``, ``normalized``,
``set``, ``numeric`` (with ``tol``), ``regex``, ``present``, ``date``. There is
deliberately no ``judge`` mode: an argument needing semantic matching means the
dataset is underspecified, and a judge would hide that rather than fix it.

``tool.args`` arrives as a redacted JSON string, because the span schema is a
flat scalar map. So these tests pass JSON, exactly as a real trace carries it.
"""

from __future__ import annotations

import json
from typing import Any

from neverempty import scorers
from neverempty.dataset.case import Case

from ._scoring import tool_span, trace

FULLWIDTH_TWO = "２"  # noqa: RUF001 - the confusable IS the subject of the test
"""A compatibility digit: NFKC folds it to ASCII "2", NFC does not."""


def case(tool: str = "search_jobs", *, mode: str = "first", **args: Any) -> Case:
    return Case.model_validate(
        {
            "id": "nr-args-0001",
            "suite": "nextrole.routing",
            "split": "dev",
            "input": {"messages": [{"role": "user", "content": "find me a job"}]},
            "expect": {"tool_calls": {"mode": mode, "calls": [{"tool": tool, "args": args}]}},
        }
    )


def called(tool: str = "search_jobs", **args: Any) -> Any:
    return trace(spans=[tool_span(tool, start_ns=1_000, args=json.dumps(args))])


def score_one(subject_case: Case, subject_trace: Any) -> Any:
    verdict = scorers.arguments().score(subject_case, subject_trace)
    assert verdict is not None
    return verdict


class TestExact:
    def test_an_identical_value_passes(self) -> None:
        verdict = score_one(case(city={"value": "Pune"}), called(city="Pune"))
        assert verdict.passed is True

    def test_a_different_value_fails(self) -> None:
        verdict = score_one(case(city={"value": "Pune"}), called(city="Mumbai"))
        assert verdict.passed is False

    def test_case_differences_fail_under_exact(self) -> None:
        """That is what distinguishes ``exact`` from ``normalized``."""
        verdict = score_one(case(city={"value": "Pune"}), called(city="pune"))
        assert verdict.passed is False

    def test_exact_is_the_default_mode(self) -> None:
        verdict = score_one(case(city={"value": "Pune"}), called(city="pune"))
        assert verdict.detail["arguments"]["city"]["match"] == "exact"

    def test_a_zero_argument_is_compared_not_treated_as_missing(self) -> None:
        """``0`` is a legitimate value: the same rule the tool wrapper enforces."""
        verdict = score_one(case(min_years={"value": 0}), called(min_years=0))
        assert verdict.passed is True

    def test_a_false_argument_is_compared_not_treated_as_missing(self) -> None:
        verdict = score_one(case(remote={"value": False}), called(remote=False))
        assert verdict.passed is True

    def test_an_empty_string_argument_is_compared(self) -> None:
        verdict = score_one(case(query={"value": ""}), called(query=""))
        assert verdict.passed is True

    def test_a_null_expected_value_matches_a_null_argument(self) -> None:
        verdict = score_one(case(company={"value": None}), called(company=None))
        assert verdict.passed is True


class TestNormalized:
    def test_case_is_ignored(self) -> None:
        verdict = score_one(
            case(city={"value": "Pune", "match": "normalized"}), called(city="pune")
        )
        assert verdict.passed is True

    def test_surrounding_and_repeated_whitespace_is_ignored(self) -> None:
        verdict = score_one(
            case(city={"value": "New Delhi", "match": "normalized"}),
            called(city="  new   delhi "),
        )
        assert verdict.passed is True

    def test_hindi_nfkc_equivalent_forms_match(self) -> None:
        """Devanagari nukta forms have two encodings. A dataset labelled by a
        human and an argument produced by a model routinely differ by exactly
        this, and calling that an agent error would be a measurement bug."""
        decomposed = "नौकरी"
        composed = "नौकरी"
        verdict = score_one(
            case(query={"value": composed, "match": "normalized"}), called(query=decomposed)
        )
        assert verdict.passed is True

    def test_a_decomposed_nukta_matches_its_precomposed_form(self) -> None:
        precomposed = "ड़"
        decomposed = "ड़"
        assert precomposed != decomposed
        verdict = score_one(
            case(query={"value": precomposed, "match": "normalized"}), called(query=decomposed)
        )
        assert verdict.passed is True

    def test_a_devanagari_digit_matches_its_ascii_compatibility_form(self) -> None:
        """NFKC folds compatibility characters, which is why the doc names NFKC
        rather than NFC."""
        verdict = score_one(
            case(years={"value": "2", "match": "normalized"}),
            called(years=FULLWIDTH_TWO),
        )
        assert verdict.passed is True

    def test_genuinely_different_hindi_words_still_fail(self) -> None:
        """Normalisation must not be so aggressive that it stops measuring."""
        verdict = score_one(
            case(query={"value": "नौकरी", "match": "normalized"}),
            called(query="वेतन"),
        )
        assert verdict.passed is False

    def test_a_non_string_value_is_compared_after_stringification(self) -> None:
        verdict = score_one(case(years={"value": 2, "match": "normalized"}), called(years="2"))
        assert verdict.passed is True


class TestSet:
    def test_order_does_not_matter(self) -> None:
        verdict = score_one(
            case(skills={"value": ["python", "sql"], "match": "set"}),
            called(skills=["sql", "python"]),
        )
        assert verdict.passed is True

    def test_a_missing_member_fails(self) -> None:
        verdict = score_one(
            case(skills={"value": ["python", "sql"], "match": "set"}), called(skills=["python"])
        )
        assert verdict.passed is False

    def test_an_extra_member_fails(self) -> None:
        verdict = score_one(
            case(skills={"value": ["python"], "match": "set"}),
            called(skills=["python", "java"]),
        )
        assert verdict.passed is False

    def test_duplicates_do_not_change_the_set(self) -> None:
        verdict = score_one(
            case(skills={"value": ["python", "sql"], "match": "set"}),
            called(skills=["python", "sql", "python"]),
        )
        assert verdict.passed is True

    def test_two_empty_lists_match(self) -> None:
        """An expectation of "no skills" is a real expectation, not a missing one."""
        verdict = score_one(case(skills={"value": [], "match": "set"}), called(skills=[]))
        assert verdict.passed is True

    def test_a_non_list_argument_fails_rather_than_crashing(self) -> None:
        verdict = score_one(
            case(skills={"value": ["python"], "match": "set"}), called(skills="python")
        )
        assert verdict.passed is False
        assert "not a list" in verdict.detail["arguments"]["skills"]["note"]

    def test_unhashable_members_are_compared_without_crashing(self) -> None:
        verdict = score_one(
            case(filters={"value": [{"a": 1}, {"b": 2}], "match": "set"}),
            called(filters=[{"b": 2}, {"a": 1}]),
        )
        assert verdict.passed is True


class TestNumeric:
    def test_an_exact_number_passes(self) -> None:
        verdict = score_one(case(years={"value": 3, "match": "numeric"}), called(years=3))
        assert verdict.passed is True

    def test_a_value_inside_the_tolerance_passes(self) -> None:
        verdict = score_one(
            case(salary={"value": 100.0, "match": "numeric", "tol": 5.0}), called(salary=103.0)
        )
        assert verdict.passed is True

    def test_a_value_outside_the_tolerance_fails(self) -> None:
        verdict = score_one(
            case(salary={"value": 100.0, "match": "numeric", "tol": 5.0}), called(salary=106.0)
        )
        assert verdict.passed is False

    def test_the_tolerance_is_inclusive(self) -> None:
        verdict = score_one(
            case(salary={"value": 100.0, "match": "numeric", "tol": 5.0}), called(salary=105.0)
        )
        assert verdict.passed is True

    def test_with_no_tolerance_the_comparison_is_equality(self) -> None:
        verdict = score_one(case(years={"value": 3, "match": "numeric"}), called(years=3.5))
        assert verdict.passed is False

    def test_an_int_and_a_float_of_the_same_value_match(self) -> None:
        verdict = score_one(case(years={"value": 3, "match": "numeric"}), called(years=3.0))
        assert verdict.passed is True

    def test_a_numeric_string_is_parsed(self) -> None:
        """Models return "3" for a number constantly; the dataset should not
        have to guess which."""
        verdict = score_one(case(years={"value": 3, "match": "numeric"}), called(years="3"))
        assert verdict.passed is True

    def test_an_unparseable_value_fails_rather_than_crashing(self) -> None:
        verdict = score_one(case(years={"value": 3, "match": "numeric"}), called(years="three"))
        assert verdict.passed is False
        assert "not numeric" in verdict.detail["arguments"]["years"]["note"]

    def test_a_boolean_is_not_a_number(self) -> None:
        """Python says ``True == 1``. An argument scorer must not."""
        verdict = score_one(case(years={"value": 1, "match": "numeric"}), called(years=True))
        assert verdict.passed is False


class TestRegex:
    def test_a_matching_pattern_passes(self) -> None:
        verdict = score_one(case(city={"value": "^Pun", "match": "regex"}), called(city="Pune"))
        assert verdict.passed is True

    def test_a_non_matching_pattern_fails(self) -> None:
        verdict = score_one(case(city={"value": "^Pun", "match": "regex"}), called(city="Mumbai"))
        assert verdict.passed is False

    def test_the_pattern_is_a_search_not_a_full_match(self) -> None:
        verdict = score_one(
            case(query={"value": "analyst", "match": "regex"}),
            called(query="senior data analyst remote"),
        )
        assert verdict.passed is True

    def test_a_non_string_argument_is_stringified_before_matching(self) -> None:
        verdict = score_one(case(years={"value": r"^\d+$", "match": "regex"}), called(years=3))
        assert verdict.passed is True


class TestPresent:
    def test_any_value_passes_when_only_presence_is_required(self) -> None:
        verdict = score_one(case(resume_id={"match": "present"}), called(resume_id="r-77"))
        assert verdict.passed is True

    def test_a_missing_argument_fails(self) -> None:
        verdict = score_one(case(resume_id={"match": "present"}), called(city="Pune"))
        assert verdict.passed is False

    def test_an_explicit_null_is_not_present(self) -> None:
        verdict = score_one(case(resume_id={"match": "present"}), called(resume_id=None))
        assert verdict.passed is False

    def test_an_empty_string_counts_as_present(self) -> None:
        """ "Present" is a question about the key, not about the value being
        interesting. Falsiness inference is exactly what this library forbids."""
        verdict = score_one(case(resume_id={"match": "present"}), called(resume_id=""))
        assert verdict.passed is True

    def test_an_empty_list_counts_as_present(self) -> None:
        verdict = score_one(case(skills={"match": "present"}), called(skills=[]))
        assert verdict.passed is True

    def test_zero_counts_as_present(self) -> None:
        verdict = score_one(case(years={"match": "present"}), called(years=0))
        assert verdict.passed is True


class TestDate:
    def test_the_same_instant_in_two_formats_matches(self) -> None:
        verdict = score_one(
            case(posted_after={"value": "2026-09-01T00:00:00Z", "match": "date"}),
            called(posted_after="2026-09-01T00:00:00+00:00"),
        )
        assert verdict.passed is True

    def test_a_different_instant_fails(self) -> None:
        verdict = score_one(
            case(posted_after={"value": "2026-09-01T00:00:00Z", "match": "date"}),
            called(posted_after="2026-09-02T00:00:00Z"),
        )
        assert verdict.passed is False

    def test_the_same_instant_in_two_timezones_matches(self) -> None:
        """Timezone-aware, so 05:30 IST and 00:00 UTC are the same moment."""
        verdict = score_one(
            case(posted_after={"value": "2026-09-01T00:00:00Z", "match": "date"}),
            called(posted_after="2026-09-01T05:30:00+05:30"),
        )
        assert verdict.passed is True

    def test_a_plain_date_matches_midnight_utc(self) -> None:
        verdict = score_one(
            case(posted_after={"value": "2026-09-01", "match": "date"}),
            called(posted_after="2026-09-01T00:00:00Z"),
        )
        assert verdict.passed is True

    def test_a_naive_datetime_is_read_as_utc(self) -> None:
        """Stated explicitly rather than left to the platform: a scorer whose
        answer depends on the CI runner timezone is not a measurement."""
        verdict = score_one(
            case(posted_after={"value": "2026-09-01T00:00:00Z", "match": "date"}),
            called(posted_after="2026-09-01T00:00:00"),
        )
        assert verdict.passed is True

    def test_an_unparseable_date_fails_rather_than_crashing(self) -> None:
        verdict = score_one(
            case(posted_after={"value": "2026-09-01", "match": "date"}),
            called(posted_after="last tuesday"),
        )
        assert verdict.passed is False
        assert "not a date" in verdict.detail["arguments"]["posted_after"]["note"]


class TestAllArgsExact:
    def test_all_arguments_correct_passes_the_call(self) -> None:
        verdict = score_one(
            case(city={"value": "Pune"}, remote={"value": True}),
            called(city="Pune", remote=True),
        )
        assert verdict.passed is True
        assert verdict.detail["all_args_exact"] is True

    def test_one_wrong_argument_fails_the_call(self) -> None:
        verdict = score_one(
            case(city={"value": "Pune"}, remote={"value": True}),
            called(city="Pune", remote=False),
        )
        assert verdict.passed is False
        assert verdict.detail["all_args_exact"] is False

    def test_the_per_argument_breakdown_is_reported(self) -> None:
        """Per-argument accuracy is its own metric, so the detail has to carry
        each argument rather than only the collapsed verdict."""
        verdict = score_one(
            case(city={"value": "Pune"}, remote={"value": True}),
            called(city="Mumbai", remote=True),
        )
        arguments = verdict.detail["arguments"]
        assert arguments["city"]["passed"] is False
        assert arguments["remote"]["passed"] is True

    def test_the_value_is_the_fraction_of_arguments_matched(self) -> None:
        verdict = score_one(
            case(city={"value": "Pune"}, remote={"value": True}),
            called(city="Mumbai", remote=True),
        )
        assert verdict.value == 0.5

    def test_an_unexpected_extra_argument_does_not_fail_the_call(self) -> None:
        """The dataset declares what matters. An argument it does not mention
        is unjudged, not wrong."""
        verdict = score_one(case(city={"value": "Pune"}), called(city="Pune", page=2))
        assert verdict.passed is True

    def test_the_actual_arguments_are_reported(self) -> None:
        verdict = score_one(case(city={"value": "Pune"}), called(city="Mumbai"))
        assert verdict.detail["arguments"]["city"]["actual"] == "Mumbai"


class TestMissingCall:
    def test_the_expected_tool_never_being_called_fails(self) -> None:
        """A tool that was never called cannot have had correct arguments. This
        is a genuine failure, not a missing measurement."""
        verdict = score_one(case("search_jobs", city={"value": "Pune"}), called("send_gmail"))
        assert verdict.passed is False
        assert verdict.detail["tool_called"] is False

    def test_a_call_with_no_recorded_args_is_not_applicable(self) -> None:
        """A span without ``tool.args`` was written by an adapter that does not
        record them. That is a harness gap, so it must not read as an agent
        error."""
        subject = trace(spans=[tool_span("search_jobs", start_ns=1_000)])
        verdict = scorers.arguments().score(case(city={"value": "Pune"}), subject)
        assert verdict is None

    def test_unparseable_recorded_args_are_not_applicable(self) -> None:
        subject = trace(spans=[tool_span("search_jobs", start_ns=1_000, args="{not json")])
        verdict = scorers.arguments().score(case(city={"value": "Pune"}), subject)
        assert verdict is None

    def test_the_first_call_is_the_one_scored_under_mode_first(self) -> None:
        subject = trace(
            spans=[
                tool_span("search_jobs", start_ns=1_000, args=json.dumps({"city": "Pune"})),
                tool_span("search_jobs", start_ns=2_000, args=json.dumps({"city": "Mumbai"})),
            ]
        )
        verdict = score_one(case(city={"value": "Pune"}), subject)
        assert verdict.passed is True


class TestRedaction:
    def test_a_redacted_argument_is_not_applicable_not_a_failure(self) -> None:
        """Redaction is a configuration choice; scoring it as an agent error
        would blame the agent for the harness. The argument is skipped and the
        reason recorded, so a report can say what it did not measure.

        A second, readable argument keeps the *call* measurable; a call whose
        every scored argument is redacted is not applicable outright, which the
        next test covers.
        """
        verdict = score_one(
            case(email={"value": "a@b.com"}, city={"value": "Pune"}),
            called(email="[REDACTED:ab12cd]", city="Pune"),
        )
        assert verdict.detail["arguments"]["email"]["passed"] is None
        assert verdict.detail["arguments"]["email"]["note"] == "redacted"

    def test_a_call_whose_every_scored_argument_is_redacted_is_not_applicable(self) -> None:
        subject = called(email="[REDACTED:ab12cd]")
        verdict = scorers.arguments().score(case(email={"value": "a@b.com"}), subject)
        assert verdict is None

    def test_redaction_does_not_hide_a_sibling_arguments_failure(self) -> None:
        verdict = score_one(
            case(email={"value": "a@b.com"}, city={"value": "Pune"}),
            called(email="[REDACTED:ab12cd]", city="Mumbai"),
        )
        assert verdict.passed is False
        assert verdict.detail["skipped"] == ["email"]

    def test_a_redacted_argument_is_excluded_from_the_fraction(self) -> None:
        """Otherwise a redacted suite would silently report a lower score."""
        verdict = score_one(
            case(email={"value": "a@b.com"}, city={"value": "Pune"}),
            called(email="[REDACTED:ab12cd]", city="Pune"),
        )
        assert verdict.value == 1.0
        assert verdict.passed is True

    def test_a_literal_lookalike_string_is_not_treated_as_redacted(self) -> None:
        """The placeholder shape is specific: six lowercase hex digits. A
        chat message that merely mentions redaction is just a value."""
        verdict = score_one(case(note={"value": "[REDACTED]"}), called(note="[REDACTED]"))
        assert verdict.passed is True


class TestNotApplicable:
    def test_a_case_with_no_tool_call_expectation_is_not_applicable(self) -> None:
        without = Case.model_validate(
            {
                "id": "nr-route-0001",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        assert scorers.arguments().score(without, called(city="Pune")) is None

    def test_an_expected_call_declaring_no_args_is_not_applicable(self) -> None:
        """The case asked which tool, not which arguments. A different scorer
        answers that question."""
        assert scorers.arguments().score(case("search_jobs"), called("search_jobs")) is None

    def test_the_scorer_declares_what_it_requires(self) -> None:
        assert scorers.arguments().requires == frozenset({"expect.tool_calls"})

    def test_the_scorer_is_named(self) -> None:
        assert scorers.arguments().name == "arguments"
