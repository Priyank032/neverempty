"""D26: a scorer the package exports but a config cannot name.

``empty_payload`` was added to ``neverempty.scorers`` with tests, a collapse
rule and documentation, and was unusable from the CLI: ``KNOWN_SCORERS`` in
``config.py`` is a hand-maintained allowlist and nobody updated it. A config
naming it was rejected as a typo.

Found by using the tool on a third-party agent rather than by reading the code,
which is the only way this kind of gap surfaces -- every unit test passed, and
the scorer worked perfectly when constructed directly.

The two lists are now checked against each other. A scorer that the package
exports and a config cannot request is a feature that exists only for people
who read the source, and the error message it produces ("unknown scorer, a typo
here is an expectation nobody measures") actively misleads: the name was right
and the allowlist was wrong.
"""

from __future__ import annotations

import inspect

from neverempty import scorers as registry
from neverempty.config import KNOWN_SCORERS


def _exported_factories() -> set[str]:
    """Every public zero-argument factory the scorers package exports."""
    names: set[str] = set()
    for name in registry.__all__:
        candidate = getattr(registry, name, None)
        # Functions only: a TypeAlias like ``CollapseRule`` is callable enough
        # to fool ``callable()`` but is not a scorer factory.
        if not inspect.isfunction(candidate):
            continue
        signature = inspect.signature(candidate)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if not required:
            names.add(name)
    return names


class TestTheAllowlistMatchesThepackage:
    def test_every_exported_scorer_can_be_named_in_a_config(self) -> None:
        missing = sorted(_exported_factories() - KNOWN_SCORERS)
        assert not missing, (
            f"these scorers exist but a config cannot request them: {missing}. "
            f"Add them to KNOWN_SCORERS in config.py."
        )

    def test_every_allowed_name_has_a_factory(self) -> None:
        """The other direction: an allowlisted name with no factory passes
        validation and then fails at run time, after the preflight that exists
        to catch exactly that."""
        orphans = sorted(KNOWN_SCORERS - _exported_factories())
        assert not orphans, f"these names are allowed in a config but have no factory: {orphans}."

    def test_empty_payload_specifically(self) -> None:
        """The one that was missing, named so the regression is legible."""
        assert "empty_payload" in KNOWN_SCORERS


class TestTheBuilderConstructsThem:
    def test_every_allowed_scorer_builds(self) -> None:
        """Validation passing is not the same as the scorer existing."""
        from neverempty.cli import _build_scorers

        built = _build_scorers(sorted(KNOWN_SCORERS))
        assert len(built) == len(KNOWN_SCORERS)

    def test_each_one_has_a_score_method(self) -> None:
        from neverempty.cli import _build_scorers

        for scorer in _build_scorers(sorted(KNOWN_SCORERS)):
            assert callable(getattr(scorer, "score", None)), scorer
