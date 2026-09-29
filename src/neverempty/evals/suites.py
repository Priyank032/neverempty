"""Suite coverage: refuse a split that cannot support its own numbers.

The doc's first and most likely trap is writing labels after seeing the agent's
answers. The second, quieter version of the same failure is running a suite whose
labels do not cover the branches the agent can take: it completes, it reports a
percentage, and the percentage is about whichever branches happened to be
labelled.

So coverage is checked before a run, not discovered afterwards from a confusion
matrix with empty rows. An unlabelled suite refuses to run rather than
publishing 0% over an empty denominator -- this library's own headline rule,
applied to its own dataset.

``min_per_branch`` defaults to the doc's 30, which is what its Wilson table
assumes: at 30 cases a per-branch interval spans roughly 25 points and is
labelled indicative; below that it is not worth printing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from neverempty.dataset.case import Case, Split

MIN_PER_BRANCH = 30
"""The doc's per-branch floor for a frozen test split.

Stated as a constant because it is the number the published per-branch intervals
depend on. Lowering it is a decision about what may be published, not a tuning
knob.
"""


@dataclass(frozen=True)
class SuiteSpec:
    """What one suite must contain before it may be run.

    ``branches`` is the set of routes the agent can actually produce, which is
    asserted against the compiled graph elsewhere. Declaring it here as well is
    what lets a label naming a route that does not exist be caught as a typo
    rather than scoring zero forever.
    """

    name: str
    branches: tuple[str, ...]
    min_per_branch: int = MIN_PER_BRANCH
    split: Split = "test"

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError(
                f"suite {self.name!r} declares no branches; a routing suite with no "
                f"branch list cannot tell a typo from a real label"
            )
        duplicates = sorted({b for b in self.branches if self.branches.count(b) > 1})
        if duplicates:
            raise ValueError(
                f"suite {self.name!r} lists duplicate branches {duplicates}; the "
                f"per-branch denominator would be wrong"
            )

    @property
    def expected_total(self) -> int:
        """How many cases a fully labelled split holds."""
        return len(self.branches) * self.min_per_branch


@dataclass(frozen=True)
class CoverageReport:
    """Whether a suite is ready, and what is missing if not."""

    spec: SuiteSpec
    coverage: dict[str, int]
    total: int
    problems: list[str] = field(default_factory=list)
    unknown_routes: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        """A table naming every branch, including the empty ones.

        Empty branches are the reason this prints every row rather than only the
        failures: the labelling backlog is the useful output here, and a branch
        missing from the table reads as a branch that is fine.
        """
        lines = [
            f"Suite {self.spec.name} ({self.spec.split} split): "
            f"{self.total} case(s), {len(self.spec.branches)} branch(es), "
            f"minimum {self.spec.min_per_branch} per branch.",
            "",
            "| branch | cases | short by |",
            "| --- | --- | --- |",
        ]
        for branch in self.spec.branches:
            count = self.coverage.get(branch, 0)
            short = max(0, self.spec.min_per_branch - count)
            lines.append(f"| {branch} | {count} | {short or '—'} |")
        if self.unknown_routes:
            lines += ["", "Routes labelled but not declared by the suite:"]
            lines += [
                f"- `{route}` ({count} case(s))"
                for route, count in sorted(self.unknown_routes.items())
            ]
        if self.problems:
            lines += ["", "Problems:"]
            lines += [f"- {problem}" for problem in self.problems]
        else:
            lines += ["", f"Ready: every branch has at least {self.spec.min_per_branch}."]
        return "\n".join(lines)


def _selected(cases: Sequence[Case], spec: SuiteSpec) -> list[Case]:
    return [case for case in cases if case.suite == spec.name and case.split == spec.split]


def branch_coverage(cases: Sequence[Case], spec: SuiteSpec) -> dict[str, int]:
    """Cases per declared branch, zeros included.

    Every declared branch appears in the result even with no cases, so a missing
    branch is a visible zero rather than an absent key.
    """
    coverage = dict.fromkeys(spec.branches, 0)
    for case in _selected(cases, spec):
        route = case.expect.route
        if route is not None and route.label in coverage:
            coverage[route.label] += 1
    return coverage


def check_coverage(cases: Sequence[Case], spec: SuiteSpec) -> CoverageReport:
    """Decide whether a suite may be run, listing every problem at once.

    Every problem is collected rather than raising on the first, because the
    useful output is the whole labelling backlog: fixing one branch at a time
    across eleven runs is how a suite stays unfinished.
    """
    selected = _selected(cases, spec)
    coverage = branch_coverage(cases, spec)
    problems: list[str] = []

    # Provenance is not re-checked here: ``Case`` already refuses a test-split
    # case without it, so a second check would be dead code pretending to guard.
    unknown: dict[str, int] = {}
    for case in selected:
        route = case.expect.route
        if route is not None and route.label not in coverage:
            unknown[route.label] = unknown.get(route.label, 0) + 1

    if not selected:
        problems.append(
            f"no cases for suite {spec.name!r} on the {spec.split!r} split. The suite "
            f"refuses to run rather than reporting a rate over an empty denominator: "
            f"expected about {spec.expected_total} labelled cases "
            f"({spec.min_per_branch} per branch across {len(spec.branches)} branches)."
        )

    for branch in spec.branches:
        count = coverage[branch]
        if count < spec.min_per_branch:
            problems.append(
                f"branch {branch!r} has {count} case(s), below the minimum of "
                f"{spec.min_per_branch}; a per-branch rate at this n is noise, "
                f"so the suite is not ready to publish one"
            )

    if unknown:
        listed = ", ".join(f"{route!r} ({count})" for route, count in sorted(unknown.items()))
        problems.append(
            f"these labelled routes are not branches of {spec.name!r}: {listed}. "
            f"A label the agent can never produce scores zero forever without "
            f"ever looking like an error"
        )

    return CoverageReport(
        spec=spec,
        coverage=coverage,
        total=len(selected),
        problems=problems,
        unknown_routes=unknown,
    )


__all__ = [
    "MIN_PER_BRANCH",
    "CoverageReport",
    "SuiteSpec",
    "branch_coverage",
    "check_coverage",
]
