"""D17: extras that install an SDK no code in the package uses.

``neverempty[anthropic]`` pulled in the Anthropic SDK and ``neverempty[otel]``
pulled in two OpenTelemetry packages, while the package contains no Anthropic
instrumentation and no OTel exporter -- ``instrument_openai`` and
``instrument_bedrock`` exist, ``instrument_anthropic`` does not, and the OTel
exporter is M14, which the design doc puts outside 0.1.0.

An extra is a claim that installing it gets you something. These got the user
a download and nothing else, which is the same shape as every other defect
here: an absence presented as a capability.

The ``jev`` extra already sets the precedent for the honest alternative --
declared with no dependency and a comment saying why -- so the two unbacked
extras are removed rather than left to imply support, and the ones with code
behind them are pinned here so the claim and the code stay in step.
"""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras: dict[str, list[str]] = data["project"]["optional-dependencies"]
    return extras


class TestEveryExtraHasCodeBehindIt:
    def test_there_is_no_anthropic_extra(self) -> None:
        assert "anthropic" not in _extras()

    def test_there_is_no_otel_extra(self) -> None:
        assert "otel" not in _extras()

    def test_the_readme_does_not_advertise_them(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        assert "neverempty[anthropic]" not in readme
        assert "neverempty[otel]" not in readme


class TestTheBackedExtrasStay:
    def test_langgraph_is_still_offered(self) -> None:
        assert _extras()["langgraph"]

    def test_openai_is_still_offered(self) -> None:
        """``instrument_openai`` exists, so the extra is a real claim."""
        from neverempty import Tracer

        assert hasattr(Tracer, "instrument_openai")
        assert _extras()["openai"]

    def test_bedrock_is_still_offered(self) -> None:
        from neverempty import Tracer

        assert hasattr(Tracer, "instrument_bedrock")
        assert _extras()["bedrock"]

    def test_jev_stays_empty_on_purpose(self) -> None:
        """Declared with no dependency: the extra names a capability whose
        backend the caller supplies. That is the honest shape."""
        assert _extras()["jev"] == []
