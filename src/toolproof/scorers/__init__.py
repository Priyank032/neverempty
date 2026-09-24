"""Built-in scorers.

One hard rule, which is the same rule this library exists to enforce on agents,
applied to itself: a scorer that cannot decide returns ``None`` (not applicable)
or raises ``ScorerError``. It never returns ``passed=False`` because data was
missing. Missing must not look like failure.

Used as factories, so a scorer can take options later without changing the call
sites::

    from toolproof import scorers

    runner = Runner(target=..., scorers=[scorers.route(), scorers.arguments()])
"""

from toolproof.scorers.arguments import ArgumentsScorer, arguments
from toolproof.scorers.calibration import (
    CONFIDENCE_KEYS,
    CalibrationScorer,
    calibration,
    stated_confidence,
)
from toolproof.scorers.collapse import COLLAPSE_RULES, CollapseRule, collapse, collapse_rule
from toolproof.scorers.facts import (
    FactsScorer,
    ForbiddenClaimsScorer,
    facts,
    forbidden_claims,
)
from toolproof.scorers.failure import (
    ABSENCE_PATTERNS,
    FAILURE_PATTERNS,
    PATTERNS_VERSION,
    FailureHandlingScorer,
    FalseAlarmScorer,
    failure_handling,
    false_alarm,
)
from toolproof.scorers.route import RouteScorer, route
from toolproof.scorers.tools import (
    ForbiddenToolsScorer,
    ToolSelectionScorer,
    forbidden_tools,
    tool_selection,
)

__all__ = [
    "ABSENCE_PATTERNS",
    "COLLAPSE_RULES",
    "CONFIDENCE_KEYS",
    "FAILURE_PATTERNS",
    "PATTERNS_VERSION",
    "ArgumentsScorer",
    "CalibrationScorer",
    "CollapseRule",
    "FactsScorer",
    "FailureHandlingScorer",
    "FalseAlarmScorer",
    "ForbiddenClaimsScorer",
    "ForbiddenToolsScorer",
    "RouteScorer",
    "ToolSelectionScorer",
    "arguments",
    "calibration",
    "collapse",
    "collapse_rule",
    "facts",
    "failure_handling",
    "false_alarm",
    "forbidden_claims",
    "forbidden_tools",
    "route",
    "stated_confidence",
    "tool_selection",
]
