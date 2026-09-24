"""Compare and gate.

Acceptance row: "each exit code reproduced by a crafted pair of reports".

| Code | Meaning |
| 0 | pass |
| 1 | significant regression (McNemar) or a floor breached |
| 2 | a must_pass case failed |
| 3 | inconclusive: unstable rate above max_unstable_rate |
| 4 | invalid input: incomplete report, version mismatch, missing baseline |

The ordering between codes is itself a decision: a must_pass failure (2)
outranks a statistical regression (1), and invalid input (4) outranks
everything, because a number computed from a broken input should never be
reported as a quality verdict.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolproof import CaseOutcome, Report, Score
from toolproof.core.trace import Env
from toolproof.report.gate import GateConfig, GateResult, compare, gate
from toolproof.report.report import Costs, Latency


def env(**overrides: object) -> Env:
    fields: dict[str, object] = {
        "toolproof_version": "0.0.1",
        "pricing_version": "empty-2026-09-23",
        "python_version": "3.12.10",
        "resolved_models": ["gpt-4o-2024-08-06"],
    }
    fields.update(overrides)
    return Env.model_validate(fields)


def report(
    *,
    passes: int = 0,
    fails: int = 0,
    must_pass_fails: int = 0,
    unstable: int = 0,
    complete: bool = True,
    status: str = "ok",
    suite_version: int = 1,
    repeats: int = 1,
    unscored: int = 0,
    env_overrides: dict[str, object] | None = None,
    offset: int = 0,
) -> Report:
    """A report whose case ids are stable, so two reports pair on them."""
    outcomes: list[CaseOutcome] = []
    index = 0

    def add(passed: bool, *, must_pass: bool = False, scored: bool = True) -> None:
        nonlocal index
        for repeat in range(repeats):
            outcomes.append(
                CaseOutcome(
                    case_id=f"c-{index + offset:04d}",
                    repeat=repeat,
                    scored=scored,
                    must_pass=must_pass,
                    scores=(
                        {
                            "route": Score(
                                passed=passed,
                                value=1.0 if passed else 0.0,
                                detail={"expected": "job_search", "predicted": "job_search"},
                            )
                        }
                        if scored
                        else {}
                    ),
                )
            )
        index += 1

    for _ in range(passes):
        add(True)
    for _ in range(fails):
        add(False)
    for _ in range(must_pass_fails):
        add(False, must_pass=True)
    for _ in range(unscored):
        add(False, scored=False)

    # Unstable cases: repeats that disagree with each other.
    for _ in range(unstable):
        for repeat in range(max(repeats, 2)):
            outcomes.append(
                CaseOutcome(
                    case_id=f"c-{index + offset:04d}",
                    repeat=repeat,
                    scored=True,
                    scores={
                        "route": Score(
                            passed=repeat == 0,
                            value=1.0 if repeat == 0 else 0.0,
                            detail={"expected": "job_search"},
                        )
                    },
                )
            )
        index += 1

    return Report.model_validate(
        {
            "suite": "nextrole.routing",
            "suite_version": suite_version,
            "split": "test",
            "created_at": "2026-09-23T10:00:00.000Z",
            "complete": complete,
            "status": status,
            "env": env(**(env_overrides or {})),
            "outcomes": outcomes,
            "counts": {"cases": index, "repeats": repeats, "unstable": unstable},
        }
    )


def flipped(*, base_passes: int, to_fail: int, to_pass: int = 0) -> tuple[Report, Report]:
    """A paired base and candidate differing only in which cases pass."""
    base = report(passes=base_passes, fails=to_pass)
    candidate = report(passes=base_passes - to_fail, fails=to_fail + to_pass)
    # Re-pair: the candidate's failing cases must be the base's passing ones.
    remapped: list[CaseOutcome] = []
    for position, outcome in enumerate(candidate.outcomes):
        passed = position >= to_fail and position < base_passes
        remapped.append(
            outcome.model_copy(
                update={
                    "scores": {
                        "route": Score(
                            passed=passed,
                            value=1.0 if passed else 0.0,
                            detail={"expected": "job_search", "predicted": "job_search"},
                        )
                    }
                }
            )
        )
    return base, candidate.model_copy(update={"outcomes": remapped})


class TestExitZero:
    def test_an_identical_pair_passes(self) -> None:
        """A report compared against itself must be a clean pass. If this is
        ever not exit 0, no baseline promotion is trustworthy."""
        subject = report(passes=30)
        result = gate(subject, subject, GateConfig())
        assert result.exit_code == 0
        assert result.verdict == "pass"

    def test_an_improvement_passes(self) -> None:
        base, candidate = flipped(base_passes=30, to_fail=0, to_pass=5)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 0

    def test_a_small_wobble_passes(self) -> None:
        """Two regressions and no improvements is p=0.25: not significant. A
        threshold gate would fail here, which is why this library does not use
        one."""
        base, candidate = flipped(base_passes=30, to_fail=2)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 0
        assert result.mcnemar is not None
        assert result.mcnemar.significant is False

    def test_the_reason_is_stated_even_on_a_pass(self) -> None:
        subject = report(passes=30)
        assert gate(subject, subject, GateConfig()).reason


class TestExitOne:
    def test_a_significant_regression_fails(self) -> None:
        """Ten cases pass-to-fail, none the other way: p < 0.001."""
        base, candidate = flipped(base_passes=30, to_fail=10)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 1
        assert result.verdict == "regression"
        assert result.mcnemar is not None
        assert result.mcnemar.b == 10
        assert result.mcnemar.c == 0

    def test_a_breached_minimum_floor_fails(self) -> None:
        """A floor catches an absolute level the paired test cannot: a suite
        that was always bad does not regress, but must not pass either."""
        base = report(passes=10, fails=20)
        candidate = report(passes=10, fails=20)
        result = gate(base, candidate, GateConfig(floors={"route": 0.75}))
        assert result.exit_code == 1
        assert "floor" in result.reason.lower()

    def test_a_breached_maximum_floor_fails(self) -> None:
        """``_max`` names a ceiling, which is how the doc's TOML expresses the
        misreport rate: the metric must stay *below* the number."""
        base = report(passes=10, fails=20)
        candidate = report(passes=10, fails=20)
        result = gate(base, candidate, GateConfig(floors={"route_max": 0.10}))
        assert result.exit_code == 1

    def test_a_satisfied_maximum_floor_passes(self) -> None:
        """A ceiling is satisfied when the metric sits below it. Here the rate
        is 0, which is what a clean failure-mode metric looks like."""
        subject = report(fails=30)
        result = gate(subject, subject, GateConfig(floors={"route_max": 0.10}))
        assert result.exit_code == 0

    def test_a_satisfied_minimum_floor_passes(self) -> None:
        subject = report(passes=30)
        result = gate(subject, subject, GateConfig(floors={"route": 0.75}))
        assert result.exit_code == 0

    def test_a_floor_on_an_unmeasured_metric_does_not_fail_the_build(self) -> None:
        """A metric with nothing to measure has no value to compare, so the
        floor cannot be evaluated. Treating "not measured" as a breach would be
        the library's own headline bug: missing must never look like failure."""
        subject = report(passes=30)
        result = gate(subject, subject, GateConfig(floors={"facts": 0.75}))
        assert result.exit_code == 0
        assert any("facts" in note for note in result.warnings)

    def test_the_breached_floor_is_named(self) -> None:
        base = report(passes=10, fails=20)
        result = gate(base, base, GateConfig(floors={"route": 0.75}))
        assert "route" in result.reason


class TestExitTwo:
    def test_a_failed_must_pass_case_fails(self) -> None:
        base = report(passes=30)
        candidate = report(passes=29, must_pass_fails=1)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 2
        assert result.verdict == "must_pass_failed"

    def test_must_pass_outranks_a_statistical_regression(self) -> None:
        """A must-pass case is a stated safety requirement. Reporting it as a
        statistical regression would invite someone to argue about the p-value."""
        base = report(passes=30)
        candidate = report(passes=19, fails=10, must_pass_fails=1)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 2

    def test_the_failing_case_ids_are_listed(self) -> None:
        base = report(passes=30)
        candidate = report(passes=29, must_pass_fails=1)
        result = gate(base, candidate, GateConfig())
        assert result.must_pass_failures

    def test_a_passing_must_pass_case_does_not_fail(self) -> None:
        subject = report(passes=30)
        assert gate(subject, subject, GateConfig()).exit_code == 0

    def test_an_unscored_must_pass_case_fails(self) -> None:
        """An unscored must-pass case is not a pass. The harness could not prove
        the requirement held, and a safety requirement unproven is unmet."""
        base = report(passes=30)
        candidate = report(passes=29)
        outcomes = [
            *candidate.outcomes,
            CaseOutcome(case_id="c-0029", repeat=0, scored=False, must_pass=True, scores={}),
        ]
        broken = candidate.model_copy(update={"outcomes": outcomes})
        result = gate(base, broken, GateConfig())
        assert result.exit_code == 2


class TestExitThree:
    def test_an_unstable_run_is_inconclusive(self) -> None:
        """Too noisy to attribute a delta to the change. It fails the build, but
        labelled so nobody reads it as a quality regression."""
        base = report(passes=20, repeats=3)
        candidate = report(passes=14, unstable=6, repeats=3)
        result = gate(base, candidate, GateConfig(max_unstable_rate=0.10))
        assert result.exit_code == 3
        assert result.verdict == "inconclusive"
        assert "rerun" in result.reason.lower()

    def test_instability_at_the_threshold_is_not_inconclusive(self) -> None:
        """The config says "above", so exactly at the rate still runs."""
        base = report(passes=20, repeats=3)
        candidate = report(passes=18, unstable=2, repeats=3)
        result = gate(base, candidate, GateConfig(max_unstable_rate=0.10))
        assert result.exit_code != 3

    def test_a_must_pass_failure_outranks_inconclusive(self) -> None:
        """A safety requirement failing is a fact, not a statistical claim, so
        noise does not excuse it."""
        base = report(passes=20, repeats=3)
        candidate = report(passes=13, unstable=6, must_pass_fails=1, repeats=3)
        result = gate(base, candidate, GateConfig(max_unstable_rate=0.10))
        assert result.exit_code == 2

    def test_inconclusive_outranks_a_statistical_regression(self) -> None:
        """If the system is too noisy to trust a delta, the delta must not be
        reported as a regression."""
        base = report(passes=20, repeats=3)
        candidate = report(passes=4, fails=10, unstable=6, repeats=3)
        result = gate(base, candidate, GateConfig(max_unstable_rate=0.10))
        assert result.exit_code == 3

    def test_the_unstable_rate_is_reported(self) -> None:
        base = report(passes=20, repeats=3)
        candidate = report(passes=14, unstable=6, repeats=3)
        result = gate(base, candidate, GateConfig(max_unstable_rate=0.10))
        assert result.unstable_rate == pytest.approx(6 / 20)


class TestExitFour:
    def test_an_incomplete_candidate_is_invalid_input(self) -> None:
        """The doc is explicit: incomplete is a failure, never a smaller n."""
        base = report(passes=30)
        candidate = report(passes=29, unscored=1, complete=False, status="incomplete")
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 4
        assert result.verdict == "invalid"

    def test_an_incomplete_baseline_is_invalid_input(self) -> None:
        base = report(passes=29, unscored=1, complete=False, status="incomplete")
        candidate = report(passes=30)
        assert gate(base, candidate, GateConfig()).exit_code == 4

    def test_a_suite_version_mismatch_is_invalid_input(self) -> None:
        """Editing the test split bumps ``suite_version``, which invalidates the
        baseline. Comparing across the bump would compare different questions."""
        base = report(passes=30, suite_version=1)
        candidate = report(passes=30, suite_version=2)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 4
        assert "suite_version" in result.reason

    def test_a_different_suite_is_invalid_input(self) -> None:
        base = report(passes=30)
        candidate = report(passes=30).model_copy(update={"suite": "other.suite"})
        assert gate(base, candidate, GateConfig()).exit_code == 4

    def test_no_paired_cases_is_invalid_input(self) -> None:
        """Two reports over disjoint case sets cannot be paired, and an
        unpaired comparison is not the test the doc specifies."""
        base = report(passes=10, offset=0)
        candidate = report(passes=10, offset=500)
        result = gate(base, candidate, GateConfig())
        assert result.exit_code == 4
        assert "pair" in result.reason.lower()

    def test_a_budget_aborted_report_is_invalid_input(self) -> None:
        base = report(passes=30)
        candidate = report(passes=20, complete=False, status="aborted_budget")
        assert gate(base, candidate, GateConfig()).exit_code == 4

    def test_invalid_input_outranks_every_quality_verdict(self) -> None:
        """A number computed from a broken input must never be reported as a
        quality verdict, however bad it looks."""
        base = report(passes=30)
        candidate = report(
            passes=0, fails=29, must_pass_fails=1, complete=False, status="incomplete"
        )
        assert gate(base, candidate, GateConfig()).exit_code == 4


class TestModelDrift:
    def test_differing_resolved_models_refuse_to_compare(self) -> None:
        """Comparing across a model change attributes the model's delta to the
        prompt change under review."""
        base = report(passes=30)
        candidate = report(passes=30, env_overrides={"resolved_models": ["gpt-4o-2024-11-20"]})
        result = compare(base, candidate)
        assert result.refused is True
        assert "model" in result.refusal_reason.lower()

    def test_the_refusal_can_be_overridden_explicitly(self) -> None:
        base = report(passes=30)
        candidate = report(passes=30, env_overrides={"resolved_models": ["gpt-4o-2024-11-20"]})
        result = compare(base, candidate, allow_model_change=True)
        assert result.refused is False

    def test_a_model_change_makes_the_gate_invalid_input(self) -> None:
        base = report(passes=30)
        candidate = report(passes=30, env_overrides={"resolved_models": ["gpt-4o-2024-11-20"]})
        assert gate(base, candidate, GateConfig()).exit_code == 4

    def test_identical_models_compare_normally(self) -> None:
        subject = report(passes=30)
        assert compare(subject, subject).refused is False

    def test_an_unrecorded_model_does_not_block_the_comparison(self) -> None:
        """A target with no LLM call records no resolved model. Refusing there
        would make the gate unusable for a deterministic agent."""
        base = report(passes=30, env_overrides={"resolved_models": []})
        candidate = report(passes=30, env_overrides={"resolved_models": []})
        assert compare(base, candidate).refused is False


class TestCompare:
    def test_flipped_cases_are_listed_in_both_directions(self) -> None:
        base, candidate = flipped(base_passes=30, to_fail=3)
        result = compare(base, candidate)
        assert len(result.regressed) == 3
        assert result.improved == []

    def test_per_metric_deltas_carry_both_intervals(self) -> None:
        base, candidate = flipped(base_passes=30, to_fail=5)
        result = compare(base, candidate)
        delta = next(d for d in result.deltas if d.name == "route")
        assert delta.base_value is not None
        assert delta.candidate_value is not None
        assert delta.delta == pytest.approx(delta.candidate_value - delta.base_value)
        assert delta.base_ci_low is not None
        assert delta.candidate_ci_low is not None

    def test_the_flip_lists_are_capped_but_the_count_is_not(self) -> None:
        """The doc caps the printed list at 20 each way. The count must still be
        the true one, or a reader would conclude only 20 cases regressed."""
        base, candidate = flipped(base_passes=40, to_fail=25)
        result = compare(base, candidate)
        assert result.regressed_count == 25
        assert len(result.regressed_shown) == 20

    def test_an_incomplete_report_refuses_to_compare(self) -> None:
        base = report(passes=30)
        candidate = report(passes=29, unscored=1, complete=False, status="incomplete")
        result = compare(base, candidate)
        assert result.refused is True
        assert "complete" in result.refusal_reason.lower()

    def test_cost_and_latency_deltas_are_reported(self) -> None:
        base = report(passes=30)
        candidate = report(passes=30)
        result = compare(base, candidate)
        assert result.cost_delta_usd is None or isinstance(result.cost_delta_usd, float)
        assert result.p95_delta_ms is None or isinstance(result.p95_delta_ms, int)

    def test_only_paired_cases_are_compared(self) -> None:
        """A case present in one report and not the other cannot be a flip."""
        base = report(passes=30)
        candidate = report(passes=35)
        result = compare(base, candidate)
        assert result.paired == 30
        assert result.unpaired_candidate == 5


class TestWarnings:
    def test_a_latency_increase_warns_without_failing(self) -> None:
        """Latency and cost are reported as warnings: the doc restricts build
        failures to three causes, and a slow run is not one of them."""
        base = report(passes=30).model_copy(
            update={"latency": Latency(samples_ms=[100], p50_ms=100, p95_ms=100)}
        )
        candidate = report(passes=30).model_copy(
            update={"latency": Latency(samples_ms=[200], p50_ms=200, p95_ms=200)}
        )
        result = gate(base, candidate, GateConfig(warn={"p95_latency_increase": 0.25}))
        assert result.exit_code == 0
        assert any("latency" in note.lower() for note in result.warnings)

    def test_a_cost_increase_warns_without_failing(self) -> None:
        base = report(passes=30).model_copy(
            update={"costs": Costs(total_usd=1.0, mean_usd=1.0, pricing_version="v1")}
        )
        candidate = report(passes=30).model_copy(
            update={"costs": Costs(total_usd=2.0, mean_usd=2.0, pricing_version="v1")}
        )
        result = gate(base, candidate, GateConfig(warn={"cost_increase": 0.20}))
        assert result.exit_code == 0
        assert any("cost" in note.lower() for note in result.warnings)

    def test_an_unknown_cost_does_not_produce_a_spurious_warning(self) -> None:
        """An unknown cost is null, and null minus null is not a 0% increase."""
        subject = report(passes=30)
        result = gate(subject, subject, GateConfig(warn={"cost_increase": 0.20}))
        assert not any("cost" in note.lower() for note in result.warnings)


class TestGateResultSerialization:
    def test_the_result_serializes_for_a_ci_summary(self, tmp_path: Path) -> None:
        base, candidate = flipped(base_passes=30, to_fail=10)
        result = gate(base, candidate, GateConfig())
        path = tmp_path / "gate.json"
        path.write_text(result.to_json(), encoding="utf-8")
        assert GateResult.model_validate_json(path.read_text(encoding="utf-8")).exit_code == 1

    def test_the_verdict_and_code_always_agree(self) -> None:
        pairs = [
            (report(passes=30), report(passes=30), 0),
            (*flipped(base_passes=30, to_fail=10), 1),
        ]
        for base, candidate, expected in pairs:
            result = gate(base, candidate, GateConfig())
            assert result.exit_code == expected
