"""D11: the default ``max_unstable_rate`` fails every unchanged rerun of a
mildly nondeterministic agent.

Measured, with 3 repeats and the default limit of 0.10:

    per-call noise   expected unstable rate
         1.0%              3.1%   passes
         2.0%              5.8%   passes
         3.5%             10.2%   FAILS
        10.0%             26.9%   FAILS

So above roughly 3.5% per-call nondeterminism an agent can never pass, and the
reviewer's 10%-noise agent gated inconclusive on 5 of 5 unchanged reruns.

**This is not a bug.** The doc specifies ``max_unstable_rate = 0.10`` in three
places (lines 476, 487, 900) with the reason stated: a system that noisy is
"too noisy to attribute any delta to the change". Exit 3 says *inconclusive*,
not *regression*, precisely so it is not misread as the agent getting worse.

What was missing is that a user meeting it has no way to know why or what to
do, and doc row 995 names that exact outcome as the trap to avoid: "Red builds
with no code change, then someone (you) disables the gate." So the fix is the
message and the documentation, not the threshold.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from neverempty import Dataset, Report, Runner, Tracer, gate, scorers
from neverempty.report.gate import GateConfig
from neverempty.tracer.sinks import MemorySink


def _dataset(tmp_path: Path, *, cases: int = 60) -> Dataset:
    lines = [
        json.dumps(
            {
                "schema_version": 1,
                "id": f"demo-{index:04d}",
                "suite": "demo.routing",
                "split": "dev",
                "input": {"messages": [{"role": "user", "content": "q"}]},
                "expect": {"route": {"label": "job_search"}},
            }
        )
        for index in range(cases)
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Dataset.load(path, split="dev")


def _run(dataset: Dataset, *, noise: float, seed: int) -> Report:
    rng = random.Random(seed)

    async def agent(case: object, tracer: object) -> None:
        route = "general" if rng.random() < noise else "job_search"
        tracer.current_run.set_output(answer="a", route=route)  # type: ignore[attr-defined]

    runner = Runner(
        target=agent,
        scorers=[scorers.route()],
        tracer=Tracer(sink=MemorySink()),
        repeats=3,
        seed=1,
    )
    return asyncio.run(runner.run(dataset))


class TestTheBehaviourIsAsSpecified:
    def test_a_noisy_agent_gates_inconclusive(self, tmp_path: Path) -> None:
        dataset = _dataset(tmp_path)
        base = _run(dataset, noise=0.10, seed=1)
        result = gate(base, _run(dataset, noise=0.10, seed=2), config=GateConfig())
        assert result.exit_code == 3
        assert result.verdict == "inconclusive"

    def test_a_deterministic_agent_passes(self, tmp_path: Path) -> None:
        dataset = _dataset(tmp_path)
        base = _run(dataset, noise=0.0, seed=1)
        result = gate(base, _run(dataset, noise=0.0, seed=2), config=GateConfig())
        assert result.exit_code == 0

    def test_it_is_not_reported_as_a_regression(self, tmp_path: Path) -> None:
        """Exit 3, not 1: the agent is not worse, the run cannot tell."""
        dataset = _dataset(tmp_path)
        result = gate(
            _run(dataset, noise=0.10, seed=1),
            _run(dataset, noise=0.10, seed=2),
            config=GateConfig(),
        )
        assert result.verdict != "regression"


class TestTheMessageTellsTheUserWhatToDo:
    """Doc row 995's trap is a user who meets this, cannot act on it, and
    switches the gate off."""

    def test_the_reason_gives_the_repeats_and_the_limit(self, tmp_path: Path) -> None:
        dataset = _dataset(tmp_path)
        result = gate(
            _run(dataset, noise=0.10, seed=1),
            _run(dataset, noise=0.10, seed=2),
            config=GateConfig(),
        )
        assert "max_unstable_rate" in result.reason

    def test_the_reason_names_a_remedy(self, tmp_path: Path) -> None:
        dataset = _dataset(tmp_path)
        result = gate(
            _run(dataset, noise=0.10, seed=1),
            _run(dataset, noise=0.10, seed=2),
            config=GateConfig(),
        )
        assert "temperature" in result.reason or "seed" in result.reason


class TestTheConsequenceIsDocumented:
    def test_the_readme_states_the_noise_ceiling(self) -> None:
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
        assert "3.5%" in readme
        assert "max_unstable_rate" in readme


class TestTheSeedClaimIsHonest:
    """D13: the README said "all randomness comes from this" of the runner's
    seed. It does not reach the agent, and the reviewer measured six different
    route scores from one seed."""

    def test_the_readme_does_not_overclaim(self) -> None:
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
        assert "all randomness comes from this" not in readme

    def test_it_says_what_the_seed_actually_covers(self) -> None:
        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
        assert "cannot reach the agent" in readme

    def test_one_seed_with_a_noisy_agent_still_varies(self, tmp_path: Path) -> None:
        """The measurement behind the claim."""
        dataset = _dataset(tmp_path, cases=30)
        scores = set()
        for agent_seed in range(4):
            report = _run(dataset, noise=0.20, seed=agent_seed)
            route = next(m for m in report.metrics if m.name == "route")
            scores.add(route.value)
        assert len(scores) > 1, "a noisy agent produced identical scores; check the fixture"
