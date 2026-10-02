"""Built-in scorers.

One hard rule, which is the same rule this library exists to enforce on agents,
applied to itself: a scorer that cannot decide returns ``None`` (not applicable)
or raises ``ScorerError``. It never returns ``passed=False`` because data was
missing. Missing must not look like failure.

Used as factories, so a scorer can take options later without changing the call
sites::

    from neverempty import scorers

    runner = Runner(target=..., scorers=[scorers.route(), scorers.arguments()])
"""

from neverempty.scorers.arguments import ArgumentsScorer, arguments
from neverempty.scorers.calibration import (
    CONFIDENCE_KEYS,
    CalibrationScorer,
    calibration,
    stated_confidence,
)
from neverempty.scorers.collapse import COLLAPSE_RULES, CollapseRule, collapse, collapse_rule
from neverempty.scorers.facts import (
    FactsScorer,
    ForbiddenClaimsScorer,
    facts,
    forbidden_claims,
)
from neverempty.scorers.failure import (
    ABSENCE_PATTERNS,
    FAILURE_PATTERNS,
    PATTERNS_VERSION,
    FailureHandlingScorer,
    FalseAlarmScorer,
    failure_handling,
    false_alarm,
)
from neverempty.scorers.payload import empty_payload
from neverempty.scorers.route import RouteScorer, route
from neverempty.scorers.tools import (
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
    "empty_payload",
    "facts",
    "failure_handling",
    "false_alarm",
    "forbidden_claims",
    "forbidden_tools",
    "route",
    "stated_confidence",
    "tool_selection",
]
