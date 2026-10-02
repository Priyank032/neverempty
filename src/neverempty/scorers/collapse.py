"""Per-case collapse over repeats.

Two rules, deliberately asymmetric, and the README states both:

- **majority** for capability metrics. Two passes in three is a pass, because a
  single flake is not a broken capability.
- **any_hit** for safety metrics. One occurrence in three is a finding, not
  noise: for a failure mode you are trying to eliminate, the worst observed
  behaviour is the honest summary.

Choosing one rule for everything would be wrong in both directions at once. It
would hide a rare unsafe behaviour, or call a flaky capability broken.

Continuous metrics take **median** of the values, which resists one outlier
repeat without inventing a verdict the scorer never produced.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Literal, TypeAlias

from neverempty.report.report import Score

CollapseRule: TypeAlias = Literal["majority", "any_hit", "median"]

COLLAPSE_RULES: dict[str, CollapseRule] = {
    "route": "majority",
    "tool_selection": "majority",
    "arguments": "majority",
    "facts": "median",
    "forbidden_claims": "any_hit",
    "forbidden_tools": "any_hit",
    "failure_handling": "any_hit",
    "empty_payload": "any_hit",
    "false_alarm": "any_hit",
    "calibration": "median",
}
"""Every built-in scorer's rule, stated rather than inferred.

The collapse rule changes what a published number means, so no built-in metric
may pick one up by falling through to the default.
"""

DEFAULT_RULE: CollapseRule = "majority"
"""A user-written scorer gets the fair capability rule.

Defaulting to ``any_hit`` would silently hold a custom capability metric to a
stricter standard than the built-in ones.
"""


def collapse_rule(name: str) -> CollapseRule:
    """The rule for one scorer name."""
    return COLLAPSE_RULES.get(name, DEFAULT_RULE)


def collapse(name: str, repeats: Sequence[Score]) -> Score | None:
    """Collapse one scorer's repeats into one score, or ``None``.

    ``None`` means nothing was measured: no repeats, or every repeat not
    applicable. Returning a failure there would be this library's own headline
    bug, committed in its own aggregation code.
    """
    measured = [score for score in repeats if score.passed is not None or score.value is not None]
    skipped = len(repeats) - len(measured)
    if not measured:
        return None

    verdicts = [score.passed for score in measured if score.passed is not None]
    values = [score.value for score in measured if score.value is not None]
    rule = collapse_rule(name)

    detail: dict[str, object] = {
        "collapse": rule,
        "repeats": len(measured),
        "not_applicable": skipped,
        "unstable": len(set(verdicts)) > 1,
    }
    detail.update(_carried(measured))
    detail.update(_any_hit_flags(measured))

    passed: bool | None = None
    if verdicts:
        passed_count = sum(1 for verdict in verdicts if verdict)
        detail["passed_count"] = passed_count
        detail["hit_count"] = len(verdicts) - passed_count
        # any_hit: one hit in any repeat is a hit. majority: a tie resolves to
        # failure, because a capability that works half the time is not a
        # working capability, and a tie has to resolve somewhere stated rather
        # than by list order.
        passed = all(verdicts) if rule == "any_hit" else passed_count * 2 > len(verdicts)

    value: float | None = None
    if values:
        # The median for a continuous metric; otherwise the mean, which is the
        # per-case rate. Either way it is reported *alongside* the collapsed
        # verdict, never instead of it: the verdict is what the collapse rule
        # decided, and the value is what was observed.
        value = statistics.median(values) if rule == "median" else statistics.fmean(values)

    return Score(passed=passed, value=value, detail=detail)


CARRIED_DETAIL_KEYS = ("expected", "predicted", "outcome", "correct")
"""Per-case detail fields that survive the collapse.

The collapsed score is the only thing the report and the confusion matrix see,
so a field that does not travel here is invisible downstream — the confusion
matrix would have no labels to put in its rows.

A field is carried only when every measured repeat agrees on it. ``expected``
comes from the dataset and always will; ``predicted`` disagreeing across repeats
is precisely an unstable case, and inventing one value for it would hide that.

``correct`` is here because the reliability curve is built from the collapsed
scores. Without it the curve would receive confidences with no correctness
attached and quietly measure nothing -- a bucket table full of empty cells and no
error anywhere.

The confidence itself is *not* carried: it is the collapsed ``value`` already,
the median over the repeats. Carrying it as a detail field would require every
repeat to state the identical number, so an agent that said 0.90, 0.91 and 0.89
would drop out of the curve for being too consistent to disagree about.

A case whose repeats disagree on ``correct`` does drop out. That is the honest
outcome: the agent's answer was not stable, so there is no single event for its
stated confidence to have been right about.
"""


ANY_HIT_DETAIL_KEYS = ("misreport",)
"""Boolean safety flags collapsed any-hit rather than by agreement.

``CARRIED_DETAIL_KEYS`` requires every repeat to agree, which is right for
``predicted`` -- repeats that disagree are an unstable case, and inventing one
value would hide that. It is wrong for a safety flag: an agent that claims data
does not exist in one repeat out of three has done the dangerous thing once,
and requiring agreement would drop that case out of the numerator entirely, so
the published rate would understate the danger exactly when the agent is least
stable. The doc collapses safety metrics any-hit (line 705) for this reason.
"""


def _any_hit_flags(measured: Sequence[Score]) -> dict[str, object]:
    """Safety flags, true when any repeat set them."""
    flags: dict[str, object] = {}
    for key in ANY_HIT_DETAIL_KEYS:
        present = [score.detail.get(key) for score in measured if key in score.detail]
        if present:
            flags[key] = any(bool(value) for value in present)
    return flags


def _carried(measured: Sequence[Score]) -> dict[str, object]:
    """The carried fields the repeats agree on."""
    carried: dict[str, object] = {}
    for key in CARRIED_DETAIL_KEYS:
        present = [score.detail[key] for score in measured if key in score.detail]
        if len(present) != len(measured):
            continue
        first = present[0]
        if all(value == first for value in present):
            carried[key] = first
    return carried


__all__ = [
    "ANY_HIT_DETAIL_KEYS",
    "CARRIED_DETAIL_KEYS",
    "COLLAPSE_RULES",
    "DEFAULT_RULE",
    "CollapseRule",
    "collapse",
    "collapse_rule",
]
