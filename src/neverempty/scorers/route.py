"""The route scorer: strict and lenient branch accuracy."""

from __future__ import annotations

from neverempty.dataset.case import Case
from neverempty.report.report import Score
from neverempty.runner.runner import ScorerError
from neverempty.scorers.reading import route_prediction


class RouteScorer:
    """Did the agent take the branch the case says it should have?

    Strict accuracy compares against ``label`` alone. Lenient also accepts
    ``acceptable``, which exists for the genuinely ambiguous cases where two
    branches are both defensible and the label records the preferred one.

    Both numbers are reported. Publishing only the lenient one would flatter
    the agent; publishing only the strict one would penalise it for ambiguity
    the dataset itself acknowledges.
    """

    name = "route"
    requires = frozenset({"expect.route"})

    def score(self, case: Case, trace: object) -> Score | None:
        expectation = case.expect.route
        if expectation is None:
            return None

        prediction = route_prediction(trace)  # type: ignore[arg-type]
        if prediction is None:
            raise ScorerError(
                f"no route prediction available for case {case.id!r}: none of "
                f"graph.next (probe span), the last branch node span, or "
                f"final_output.route was set. The case is unscored, which the "
                f"gate treats as a failure of the run rather than of the agent."
            )

        predicted, source, span = prediction
        strict = predicted == expectation.label
        lenient = strict or predicted in expectation.acceptable

        detail: dict[str, object] = {
            "predicted": predicted,
            "expected": expectation.label,
            "acceptable": list(expectation.acceptable),
            "source": source,
            "lenient": lenient,
        }
        if span is not None:
            # A probe records what the graph would do next *and* what the state
            # field says. A disagreement is a finding about the agent, so it
            # travels in the detail rather than changing which source wins.
            state_route = span.attributes.get("graph.state_route")
            if state_route is not None:
                detail["state_route"] = state_route
            disagreement = span.attributes.get("graph.route_disagreement")
            if disagreement is not None:
                detail["route_disagreement"] = disagreement

        return Score(passed=strict, value=1.0 if strict else 0.0, detail=detail)


def route() -> RouteScorer:
    """The route scorer."""
    return RouteScorer()


__all__ = ["RouteScorer", "route"]
