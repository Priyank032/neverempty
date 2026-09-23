"""Case v1, field by field, against the doc's constraints table.

The strictness rule is the opposite of the trace's: unknown top-level keys in
a case are a validation error, because a typo in eval data is a silent killer.
An expectation you meant to write and misspelled would otherwise be absent,
and an absent expectation scores "not applicable", never "fail".
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from toolproof import Case
from toolproof.dataset.case import (
    CASE_SCHEMA_VERSION,
    ArgExpectation,
    Fact,
    FaultDecl,
    ItemExpectation,
    Provenance,
    RouteExpectation,
    ToolCallExpectation,
)


def make_case(**overrides: Any) -> Case:
    base: dict[str, Any] = {
        "id": "nr-route-0142",
        "suite": "nextrole.routing",
        "split": "dev",
        "input": {"messages": [{"role": "user", "content": "show me jobs"}]},
        "expect": {"route": {"label": "job_search"}},
    }
    base.update(overrides)
    return Case.model_validate(base)


class TestSchemaVersion:
    def test_it_defaults_to_one(self) -> None:
        assert make_case().schema_version == CASE_SCHEMA_VERSION == 1

    def test_a_newer_version_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="schema_version"):
            make_case(schema_version=2)


class TestIdentity:
    @pytest.mark.parametrize("case_id", ["nr-route-0142", "yk-p07-pm-kisan", "a.b_c-1", "abc"])
    def test_conforming_ids_are_accepted(self, case_id: str) -> None:
        assert make_case(id=case_id).id == case_id

    @pytest.mark.parametrize(
        ("bad", "why"),
        [
            ("ab", "too short"),
            ("NR-ROUTE-1", "uppercase"),
            ("-leading-dash", "must start alphanumeric"),
            (".leading-dot", "must start alphanumeric"),
            ("has space", "space"),
            ("x" * 65, "too long"),
            ("", "empty"),
        ],
    )
    def test_malformed_ids_are_rejected(self, bad: str, why: str) -> None:
        with pytest.raises(ValidationError, match="id"):
            make_case(id=bad)

    @pytest.mark.parametrize(
        "suite", ["nextrole.routing", "yojanakhoj.consistency", "a.b.c", "x1.y_2"]
    )
    def test_conforming_suites_are_accepted(self, suite: str) -> None:
        assert make_case(suite=suite).suite == suite

    @pytest.mark.parametrize(
        "bad", ["nextrole", "Nextrole.routing", ".routing", "nextrole.", "a..b"]
    )
    def test_malformed_suites_are_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="suite"):
            make_case(suite=bad)

    @pytest.mark.parametrize("split", ["dev", "test"])
    def test_both_splits_are_accepted(self, split: str) -> None:
        provenance = {"labeller": "p", "method": "human", "labelled_at": "2026-09-23"}
        assert make_case(split=split, provenance=provenance).split == split

    def test_an_unknown_split_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="split"):
            make_case(split="train")


class TestStrictness:
    def test_an_unknown_top_level_key_is_rejected(self) -> None:
        """A typo in eval data is a silent killer, so it must not parse."""
        with pytest.raises(ValidationError, match="expects"):
            make_case(expects={"route": {"label": "x"}})

    def test_an_unknown_expect_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="rout"):
            make_case(expect={"rout": {"label": "job_search"}})

    def test_an_unknown_provenance_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_case(
                provenance={
                    "labeller": "p",
                    "method": "human",
                    "labelled_at": "2026-09-23",
                    "notes": "typo for note",
                }
            )


class TestInput:
    def test_messages_input_is_accepted(self) -> None:
        case = make_case(input={"messages": [{"role": "user", "content": "hi"}]})
        assert case.input.messages is not None
        assert case.input.messages[0].role == "user"

    def test_payload_input_is_accepted(self) -> None:
        case = make_case(input={"payload": {"persona_ref": "x", "lang": "hi"}})
        assert case.input.payload == {"persona_ref": "x", "lang": "hi"}

    def test_exactly_one_of_messages_or_payload_is_required(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            make_case(input={})

    def test_both_together_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            make_case(input={"messages": [{"role": "user", "content": "a"}], "payload": {}})

    def test_an_empty_messages_list_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="messages"):
            make_case(input={"messages": []})

    def test_an_empty_payload_object_is_a_valid_payload(self) -> None:
        """An empty dict is a payload the author chose, not an absent one."""
        assert make_case(input={"payload": {}}).input.payload == {}


class TestExpect:
    def test_at_least_one_expectation_is_required(self) -> None:
        """A case expecting nothing can never fail, so it measures nothing."""
        with pytest.raises(ValidationError, match="at least one"):
            make_case(expect={})

    def test_every_expect_key_is_individually_optional(self) -> None:
        for payload in (
            {"route": {"label": "x"}},
            {"forbidden_tools": ["send_email"]},
            {"facts": [{"id": "f1", "statement": "s", "match": "contains"}]},
            {"items": [{"item_id": "pm-kisan", "rule_result": None}]},
        ):
            assert make_case(expect=payload) is not None

    def test_an_absent_expectation_is_absent_not_empty(self) -> None:
        """The scorer must be able to tell 'not asked' from 'asked for none'."""
        case = make_case(expect={"route": {"label": "job_search"}})
        assert case.expect.tool_calls is None
        assert case.expect.forbidden_tools is None

    def test_an_explicitly_empty_forbidden_list_is_kept(self) -> None:
        case = make_case(expect={"route": {"label": "x"}, "forbidden_tools": []})
        assert case.expect.forbidden_tools == []


class TestRouteExpectation:
    def test_a_label_is_required(self) -> None:
        with pytest.raises(ValidationError):
            RouteExpectation()  # type: ignore[call-arg]

    def test_any_non_empty_label_is_allowed(self) -> None:
        """The library never knows a target's branch names; `blocked` or any
        later branch is a dataset edit, not a library change."""
        for label in ("job_search", "blocked", "clarify", "some_future_branch"):
            assert RouteExpectation(label=label).label == label

    def test_an_empty_label_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="label"):
            RouteExpectation(label="")

    def test_acceptable_must_not_contain_the_label(self) -> None:
        """Otherwise lenient and strict accuracy would silently coincide."""
        with pytest.raises(ValidationError, match="acceptable"):
            RouteExpectation(label="followup", acceptable=["followup", "email_draft"])

    def test_acceptable_alternatives_are_kept(self) -> None:
        route = RouteExpectation(label="followup", acceptable=["email_draft"])
        assert route.acceptable == ["email_draft"]

    def test_acceptable_defaults_to_empty(self) -> None:
        assert RouteExpectation(label="x").acceptable == []


class TestToolCallExpectation:
    @pytest.mark.parametrize("mode", ["first", "set", "sequence"])
    def test_every_documented_mode_is_accepted(self, mode: str) -> None:
        expectation = ToolCallExpectation.model_validate(
            {"mode": mode, "calls": [{"tool": "search_jobs"}]}
        )
        assert expectation.mode == mode

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="mode"):
            ToolCallExpectation.model_validate({"mode": "any", "calls": [{"tool": "t"}]})

    def test_at_least_one_call_is_required(self) -> None:
        with pytest.raises(ValidationError, match="calls"):
            ToolCallExpectation(mode="first", calls=[])

    def test_per_argument_match_modes_are_parsed(self) -> None:
        expectation = ToolCallExpectation.model_validate(
            {
                "mode": "first",
                "calls": [
                    {
                        "tool": "search_jobs",
                        "args": {
                            "city": {"value": "Pune", "match": "normalized"},
                            "radius": {"value": 25, "match": "numeric", "tol": 5},
                        },
                    }
                ],
            }
        )
        args = expectation.calls[0].args
        assert args["city"].match == "normalized"
        assert args["radius"].tol == 5


class TestArgExpectation:
    @pytest.mark.parametrize(
        "match", ["exact", "normalized", "set", "numeric", "regex", "present", "date"]
    )
    def test_every_documented_match_mode_is_accepted(self, match: str) -> None:
        assert ArgExpectation(value="v", match=match).match == match  # type: ignore[arg-type]

    def test_judge_is_not_a_valid_argument_match_mode(self) -> None:
        """The doc is explicit: if an argument needs semantic matching, the
        dataset is underspecified."""
        with pytest.raises(ValidationError, match="match"):
            ArgExpectation(value="v", match="judge")  # type: ignore[arg-type]

    def test_the_default_match_mode_is_exact(self) -> None:
        assert ArgExpectation(value="v").match == "exact"

    def test_tol_is_only_meaningful_for_numeric(self) -> None:
        with pytest.raises(ValidationError, match="tol"):
            ArgExpectation(value="v", match="exact", tol=1.0)

    def test_numeric_without_a_tol_is_allowed(self) -> None:
        assert ArgExpectation(value=1, match="numeric").tol is None

    def test_a_negative_tol_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tol"):
            ArgExpectation(value=1, match="numeric", tol=-1)

    def test_present_needs_no_value(self) -> None:
        assert ArgExpectation(match="present").value is None

    def test_a_falsy_expected_value_is_preserved(self) -> None:
        """0 and False are legitimate expected arguments."""
        for value in (0, False, ""):
            assert ArgExpectation(value=value).value == value

    def test_an_invalid_regex_is_rejected_at_load_time(self) -> None:
        """A bad pattern must fail on the dataset, not mid-run."""
        with pytest.raises(ValidationError, match="regex"):
            ArgExpectation(value="([unclosed", match="regex")


class TestFacts:
    @pytest.mark.parametrize("match", ["contains", "regex", "judge"])
    def test_every_documented_fact_match_mode_is_accepted(self, match: str) -> None:
        assert Fact(id="f1", statement="s", match=match).match == match  # type: ignore[arg-type]

    def test_an_unknown_fact_match_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="match"):
            Fact(id="f1", statement="s", match="semantic")  # type: ignore[arg-type]

    def test_a_statement_is_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError, match="statement"):
            Fact(id="f1", statement="", match="contains")

    def test_an_invalid_fact_regex_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="regex"):
            Fact(id="f1", statement="([unclosed", match="regex")

    def test_an_evidence_key_is_optional(self) -> None:
        assert Fact(id="f1", statement="s", match="judge").evidence_key is None

    def test_fact_ids_must_be_unique_within_a_case(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            make_case(
                expect={
                    "facts": [
                        {"id": "f1", "statement": "a", "match": "contains"},
                        {"id": "f1", "statement": "b", "match": "contains"},
                    ]
                }
            )


class TestItems:
    def test_rule_result_accepts_true_false_and_null(self) -> None:
        for result in (True, False, None):
            item = ItemExpectation(item_id="pm-kisan", rule_result=result)
            assert item.rule_result is result

    def test_null_rule_result_is_distinct_from_absent(self) -> None:
        """`null` means the rule could not evaluate, which is the case the doc
        expects to produce the most interesting finding."""
        item = ItemExpectation(item_id="pm-kisan", rule_result=None)
        assert item.rule_result is None
        assert "rule_result" in item.model_dump()

    def test_a_rule_trace_is_carried(self) -> None:
        item = ItemExpectation.model_validate(
            {
                "item_id": "pm-kisan",
                "rule_result": None,
                "rule_trace": [
                    {"criterion": "occupation", "result": True},
                    {"criterion": "is_income_taxpayer", "result": None, "missing_field": "x"},
                ],
            }
        )
        assert item.rule_trace[1].missing_field == "x"

    def test_item_ids_must_be_unique_within_a_case(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            make_case(
                expect={
                    "items": [
                        {"item_id": "pm-kisan", "rule_result": True},
                        {"item_id": "pm-kisan", "rule_result": False},
                    ]
                }
            )


class TestFaults:
    @pytest.mark.parametrize("kind", ["timeout", "upstream", "rate_limit", "empty", "truncated"])
    def test_every_documented_fault_kind_is_accepted(self, kind: str) -> None:
        assert FaultDecl(tool="search_jobs", kind=kind).kind == kind  # type: ignore[arg-type]

    def test_an_unknown_fault_kind_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            FaultDecl(tool="t", kind="explode")  # type: ignore[arg-type]

    def test_after_calls_defaults_to_zero_and_must_be_non_negative(self) -> None:
        assert FaultDecl(tool="t", kind="timeout").after_calls == 0
        with pytest.raises(ValidationError, match="after_calls"):
            FaultDecl(tool="t", kind="timeout", after_calls=-1)

    def test_a_case_carries_its_faults(self) -> None:
        case = make_case(faults=[{"tool": "search_jobs", "kind": "timeout", "after_calls": 0}])
        assert case.faults[0].tool == "search_jobs"

    def test_faults_convert_to_runtime_specs(self) -> None:
        """The dataset declaration and the wrapper's hook must agree."""
        from toolproof import FaultSpec

        case = make_case(faults=[{"tool": "search_jobs", "kind": "timeout", "after_calls": 2}])
        specs = case.fault_specs()
        assert specs == [FaultSpec(tool="search_jobs", kind="timeout", after_calls=2)]


class TestProvenance:
    @pytest.mark.parametrize("method", ["human", "llm_drafted_human_verified"])
    def test_hand_labelled_methods_need_no_source_commit(self, method: str) -> None:
        assert Provenance(method=method).method == method  # type: ignore[arg-type]

    def test_generated_from_rules_is_accepted_once_pinned(self) -> None:
        """Generated truth drifts when its source moves, so it must be pinned."""
        provenance = Provenance(method="generated_from_rules", source_commit="a" * 40)
        assert provenance.method == "generated_from_rules"

    def test_an_unknown_method_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="method"):
            Provenance(method="vibes")  # type: ignore[arg-type]

    def test_provenance_is_required_on_the_test_split(self) -> None:
        """A published number needs to say where its ground truth came from."""
        with pytest.raises(ValidationError, match="provenance"):
            make_case(split="test")

    def test_provenance_is_optional_on_the_dev_split(self) -> None:
        assert make_case(split="dev").provenance is None

    def test_generated_truth_must_pin_its_source_commit(self) -> None:
        """Ground-truth drift is only detectable if the source is pinned."""
        with pytest.raises(ValidationError, match="source_commit"):
            make_case(
                split="test",
                provenance={"method": "generated_from_rules"},
            )

    def test_generated_truth_with_a_source_commit_is_accepted(self) -> None:
        case = make_case(
            split="test",
            provenance={"method": "generated_from_rules", "source_commit": "a" * 40},
        )
        assert case.provenance is not None
        assert case.provenance.source_commit == "a" * 40

    def test_a_human_label_records_who_and_when(self) -> None:
        case = make_case(
            split="test",
            provenance={"labeller": "priyank", "method": "human", "labelled_at": "2026-09-23"},
        )
        assert case.provenance is not None
        assert case.provenance.labeller == "priyank"


class TestMustPassAndTags:
    def test_must_pass_defaults_to_false(self) -> None:
        assert make_case().must_pass is False

    def test_must_pass_is_carried(self) -> None:
        assert make_case(must_pass=True).must_pass is True

    def test_tags_default_to_empty(self) -> None:
        assert make_case().tags == []

    def test_tags_are_carried_and_free_form(self) -> None:
        case = make_case(tags=["hinglish", "ambiguous", "some-new-tag"])
        assert "hinglish" in case.tags

    def test_duplicate_tags_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tags"):
            make_case(tags=["hinglish", "hinglish"])


class TestRoundTrip:
    def test_the_docs_hinglish_example_parses(self) -> None:
        case = Case.model_validate(
            {
                "schema_version": 1,
                "id": "nr-route-0142",
                "suite": "nextrole.routing",
                "split": "test",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "mujhe Pune mein data analyst ki remote jobs dikhao, "
                                "2-4 saal experience"
                            ),
                        }
                    ]
                },
                "expect": {
                    "route": {"label": "job_search"},
                    "forbidden_tools": ["send_email"],
                },
                "tags": ["hinglish"],
                "must_pass": False,
                "provenance": {
                    "labeller": "priyank",
                    "method": "human",
                    "labelled_at": "2026-09-21",
                },
            }
        )
        assert case.expect.route is not None
        assert case.expect.route.label == "job_search"

    def test_the_docs_ambiguous_example_parses(self) -> None:
        case = Case.model_validate(
            {
                "schema_version": 1,
                "id": "nr-route-0087",
                "suite": "nextrole.routing",
                "split": "test",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "It's been 10 days since I applied to the Razorpay "
                                "SDE-2 role, can you write to the recruiter?"
                            ),
                        }
                    ]
                },
                "expect": {"route": {"label": "followup", "acceptable": ["email_draft"]}},
                "tags": ["ambiguous"],
                "must_pass": False,
                "provenance": {
                    "labeller": "priyank",
                    "method": "human",
                    "labelled_at": "2026-09-21",
                    "note": "followup because an application already exists",
                },
            }
        )
        assert case.expect.route is not None
        assert case.expect.route.acceptable == ["email_draft"]

    def test_the_docs_fault_example_parses(self) -> None:
        case = Case.model_validate(
            {
                "schema_version": 1,
                "id": "nr-fault-0009",
                "suite": "nextrole.failure",
                "split": "test",
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "any backend python jobs in Bangalore posted this week?",
                        }
                    ]
                },
                "faults": [{"tool": "search_jobs", "kind": "timeout", "after_calls": 0}],
                "expect": {
                    "route": {"label": "job_search"},
                    "forbidden_claims": [
                        {
                            "id": "absence",
                            "statement": "There are no matching jobs",
                            "match": "judge",
                        }
                    ],
                },
                "must_pass": True,
                "provenance": {"method": "human", "labeller": "p", "labelled_at": "2026-09-21"},
            }
        )
        assert case.must_pass is True
        assert case.faults[0].kind == "timeout"

    def test_the_docs_yojanakhoj_null_example_parses(self) -> None:
        case = Case.model_validate(
            {
                "schema_version": 1,
                "id": "yk-p11-pm-kisan",
                "suite": "yojanakhoj.consistency",
                "split": "test",
                "input": {
                    "payload": {
                        "persona_ref": "tests/personas.js#farmerMissingTaxStatus",
                        "lang": "en",
                        "profile": {"state": "MP", "occupation": "farmer", "land_acres": 2.0},
                    }
                },
                "expect": {
                    "items": [
                        {
                            "item_id": "pm-kisan",
                            "rule_result": None,
                            "rule_trace": [
                                {"criterion": "occupation", "result": True},
                                {
                                    "criterion": "is_income_taxpayer",
                                    "result": None,
                                    "missing_field": "is_income_taxpayer",
                                },
                            ],
                        }
                    ],
                    "forbidden_claims": [
                        {
                            "id": "definite-eligible",
                            "statement": "The person is definitely eligible",
                            "match": "judge",
                        }
                    ],
                },
                "must_pass": True,
                "provenance": {
                    "method": "generated_from_rules",
                    "source_commit": "b" * 40,
                    "scheme_file_sha256": "c" * 64,
                },
            }
        )
        assert case.expect.items is not None
        assert case.expect.items[0].rule_result is None

    def test_a_case_round_trips_through_json(self) -> None:
        case = make_case(
            split="test",
            provenance={"labeller": "p", "method": "human", "labelled_at": "2026-09-23"},
            faults=[{"tool": "t", "kind": "timeout"}],
            tags=["hinglish"],
        )
        once = case.model_dump_json()
        assert Case.model_validate_json(once).model_dump_json() == once
