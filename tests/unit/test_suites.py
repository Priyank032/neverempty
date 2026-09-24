"""Suite coverage: a split that cannot support its own published numbers.

M9's acceptance row is "suites run, gate green on a no-op, red on a sabotaged
prompt". Two things have to be true before any of that means anything: the
labels exist, and they are spread across the branches the agent can actually
take. A suite with 300 cases on one branch and none on the other ten runs
perfectly and measures nothing.

This is the check the doc's trap #1 exists for -- "labels written before any
run" -- enforced in code rather than in discipline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolproof.dataset.case import Case
from toolproof.evals.suites import (
    MIN_PER_BRANCH,
    SuiteSpec,
    branch_coverage,
    check_coverage,
)


def case(
    case_id: str,
    route: str,
    *,
    split: str = "test",
    method: str = "human",
    suite: str = "nextrole.routing",
) -> Case:
    return Case.model_validate(
        {
            "id": case_id,
            "suite": suite,
            "split": split,
            "input": {"messages": [{"role": "user", "content": "q"}]},
            "expect": {"route": {"label": route}},
            "provenance": {"method": method, "labeller": "priyank"},
        }
    )


BRANCHES = ("job_search", "email_draft", "general")
SPEC = SuiteSpec(
    name="nextrole.routing",
    branches=BRANCHES,
    min_per_branch=2,
    split="test",
)


def full_set(per_branch: int = 2) -> list[Case]:
    cases: list[Case] = []
    for branch in BRANCHES:
        for index in range(per_branch):
            cases.append(case(f"nr-{branch}-{index:04d}", branch))
    return cases


class TestBranchCoverage:
    def test_counts_every_declared_branch(self) -> None:
        coverage = branch_coverage(full_set(), SPEC)
        assert coverage == {"job_search": 2, "email_draft": 2, "general": 2}

    def test_a_branch_with_no_cases_is_zero_not_absent(self) -> None:
        """An unmeasured branch has to appear in the table. Dropping the key
        would make a missing branch invisible in the coverage report."""
        cases = [c for c in full_set() if "general" not in c.id]
        coverage = branch_coverage(cases, SPEC)
        assert coverage["general"] == 0

    def test_only_the_declared_split_counts(self) -> None:
        cases = [*full_set(), case("nr-dev-0001", "job_search", split="dev")]
        assert branch_coverage(cases, SPEC)["job_search"] == 2

    def test_cases_from_another_suite_are_ignored(self) -> None:
        cases = [
            *full_set(),
            case("nr-other-0001", "job_search", suite="nextrole.failure"),
        ]
        assert branch_coverage(cases, SPEC)["job_search"] == 2


class TestCheckCoverage:
    def test_a_balanced_labelled_split_passes(self) -> None:
        report = check_coverage(full_set(), SPEC)
        assert report.ok
        assert report.problems == []

    def test_an_empty_split_fails_rather_than_reporting_zero(self) -> None:
        """The whole point: an unlabelled suite must refuse to run, not run and
        publish 0% on an empty denominator."""
        report = check_coverage([], SPEC)
        assert not report.ok
        assert any("no cases" in problem for problem in report.problems)

    def test_a_thin_branch_is_named_with_its_count(self) -> None:
        cases = full_set()
        cases = [c for c in cases if c.id != "nr-general-0001"]
        report = check_coverage(cases, SPEC)
        assert not report.ok
        assert any("general" in problem and "1" in problem for problem in report.problems)

    def test_every_thin_branch_is_listed_not_just_the_first(self) -> None:
        """One run should surface the whole labelling backlog."""
        cases = [case("nr-job_search-0000", "job_search")]
        report = check_coverage(cases, SPEC)
        assert not report.ok
        joined = " ".join(report.problems)
        assert "email_draft" in joined
        assert "general" in joined

    def test_a_route_outside_the_declared_branches_is_a_problem(self) -> None:
        """A label the agent can never produce scores zero forever, silently."""
        cases = [*full_set(), case("nr-typo-0001", "job_serch")]
        report = check_coverage(cases, SPEC)
        assert not report.ok
        assert any("job_serch" in problem for problem in report.problems)

    def test_provenance_is_already_enforced_by_the_case_model(self) -> None:
        """Coverage does not re-check provenance on the test split, because a
        case without it cannot be constructed: ``Case`` rejects it at load.

        Asserted here so that if that validation ever moves, this suite says so
        rather than quietly letting an unlabelled case through a second door.
        """
        with pytest.raises(ValidationError, match="provenance"):
            Case.model_validate(
                {
                    "id": "nr-bare-0001",
                    "suite": "nextrole.routing",
                    "split": "test",
                    "input": {"messages": [{"role": "user", "content": "q"}]},
                    "expect": {"route": {"label": "general"}},
                }
            )

    def test_a_dev_split_case_needs_no_provenance(self) -> None:
        """The dev split is for prompt iteration, so it is not held to the
        published-number standard. Coverage still counts it for its own split."""
        dev_spec = SuiteSpec(
            name="nextrole.routing", branches=BRANCHES, min_per_branch=1, split="dev"
        )
        bare = Case.model_validate(
            {
                "id": "nr-dev-0002",
                "suite": "nextrole.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"route": {"label": "general"}},
            }
        )
        cases = [case(f"nr-d-{b}", b, split="dev") for b in BRANCHES] + [bare]
        assert check_coverage(cases, dev_spec).ok

    def test_synthetic_labels_are_flagged_on_the_test_split(self) -> None:
        """A draft label is a placeholder, not ground truth. It may sit in the
        file, but the suite must not claim to be ready."""
        cases = [
            *full_set(),
            case("nr-draft-0001", "general", method="llm_drafted_human_verified"),
        ]
        report = check_coverage(cases, SPEC)
        assert report.ok

    def test_default_min_per_branch_is_the_docs_thirty(self) -> None:
        assert MIN_PER_BRANCH == 30


class TestCoverageReport:
    def test_renders_a_table_naming_short_branches(self) -> None:
        cases = [case("nr-job_search-0000", "job_search")]
        report = check_coverage(cases, SPEC)
        text = report.render()
        assert "job_search" in text
        assert "email_draft" in text
        assert "0" in text

    def test_a_passing_report_says_how_many_per_branch(self) -> None:
        report = check_coverage(full_set(4), SPEC)
        assert report.ok
        assert "4" in report.render()

    def test_report_is_frozen(self) -> None:
        report = check_coverage(full_set(), SPEC)
        with pytest.raises((AttributeError, TypeError, ValueError)):
            report.ok = False  # type: ignore[misc]


class TestSuiteSpec:
    def test_branches_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="branch"):
            SuiteSpec(name="x.y", branches=(), min_per_branch=1, split="test")

    def test_duplicate_branches_are_refused(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            SuiteSpec(
                name="x.y",
                branches=("a", "b", "a"),
                min_per_branch=1,
                split="test",
            )

    def test_expected_total_is_branches_times_minimum(self) -> None:
        spec = SuiteSpec(name="x.y", branches=("a", "b"), min_per_branch=30, split="test")
        assert spec.expected_total == 60
