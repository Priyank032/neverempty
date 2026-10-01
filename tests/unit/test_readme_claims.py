"""The README must not claim more than the code does.

A third review read only the README, the GitHub API and the web, and found the
overclaims a stranger meets before any code runs. Those are the ones that cost
most: a reader who catches one stops trusting the rest, and the library's whole
pitch is that its numbers can be trusted.

Each test here pins a claim to what is actually true, so the prose cannot drift
back.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def _text() -> str:
    return README.read_text(encoding="utf-8")


class TestItDoesNotContradictItself:
    def test_it_does_not_call_itself_complete_and_planned(self) -> None:
        """ "The library is complete" sat above "Planned scope for 0.1.0"."""
        text = _text()
        assert not ("The library is complete" in text and "Planned scope" in text)

    def test_install_comes_before_the_first_command(self) -> None:
        """ "Start here" told a reader to run a command they could not install."""
        text = _text()
        assert text.index("## Installation") < text.index("## Start here")


class TestItSaysHowTheHeadlineMetricIsScored:
    """The one number the library exists to produce was never explained. A
    reader deciding whether to trust it could not tell a regex from a judge."""

    def test_the_readme_says_it_is_pattern_matched(self) -> None:
        text = _text()
        assert "misreport_as_empty" in text
        assert "pattern" in text.lower()

    def test_it_states_the_pattern_count(self) -> None:
        from neverempty.scorers.failure import ABSENCE_PATTERNS

        assert str(len(ABSENCE_PATTERNS)) in _text()

    def test_the_stated_count_is_the_real_one(self) -> None:
        """A number in prose that drifts from the code is the bug this library
        exists to catch, in its own README."""
        from neverempty.scorers.failure import ABSENCE_PATTERNS, FAILURE_PATTERNS

        text = _text()
        absence = re.search(r"\*\*(\d+) absence patterns\*\*", text)
        failure = re.search(r"\*\*(\d+) failure patterns\*\*", text)
        assert absence, "the README must state the absence pattern count"
        assert failure, "the README must state the failure pattern count"
        assert int(absence.group(1)) == len(ABSENCE_PATTERNS)
        assert int(failure.group(1)) == len(FAILURE_PATTERNS)

    def test_it_says_the_judge_fallback_is_not_wired_in(self) -> None:
        text = _text()
        assert "not yet" in text.lower() or "does not fall back" in text.lower()


class TestItDoesNotOverclaim:
    def test_it_does_not_claim_most_harnesses_miss_this(self) -> None:
        """ "Most" was never measured. A comparison is a claim; an adjective is
        not."""
        assert "most harnesses miss" not in _text()

    def test_it_does_not_claim_failure_can_never_look_empty(self) -> None:
        """True only for tools wrapped in @tool, and only in the tool message.
        The model can still write "no results" in its answer."""
        assert "can never look like an empty result" not in _text()

    def test_it_does_not_call_stub_output_real_numbers(self) -> None:
        assert "two reports with real numbers" not in _text()

    def test_it_does_not_call_other_languages_first_class(self) -> None:
        """There is a JSON trace contract, not an SDK."""
        assert "first-class" not in _text()

    def test_it_does_not_assert_gates_get_disabled(self) -> None:
        """An opinion stated as a fact about other people's behaviour."""
        assert "gets disabled within a month" not in _text()


class TestTheHonestClaimsSurvive:
    """The fixes must not delete what is true and specific."""

    def test_the_exit_codes_are_still_documented(self) -> None:
        text = _text()
        for code in ("McNemar", "must_pass", "inconclusive"):
            assert code in text

    def test_the_detection_floor_is_still_stated(self) -> None:
        assert "5 clean pass-to-fail flips" in _text()

    def test_the_tri_state_result_is_still_explained(self) -> None:
        text = _text()
        for name in ("Ok", "Empty", "Err"):
            assert name in text, name


class TestTheCopyableCoreIsRealAndNamedCorrectly:
    """The README now tells a reader which four files to copy instead of
    adopting the harness. Those paths have to exist, or the honesty is fake."""

    FILES = (
        "core/results.py",
        "core/classify.py",
        "core/faults.py",
        "scorers/failure.py",
    )

    def test_each_named_file_exists(self) -> None:
        for name in self.FILES:
            assert (ROOT / "src" / "neverempty" / name).is_file(), name

    def test_each_is_named_in_the_readme(self) -> None:
        text = _text()
        for name in self.FILES:
            assert f"`{name}`" in text, name

    def test_the_stated_line_count_is_roughly_right(self) -> None:
        """ "about 740 lines" has to stay about right, or it is just a number."""
        total = sum(
            len((ROOT / "src" / "neverempty" / name).read_text(encoding="utf-8").splitlines())
            for name in self.FILES
        )
        match = re.search(r"about (\d+) lines", _text())
        assert match, "the README must state the size"
        claimed = int(match.group(1))
        assert abs(total - claimed) <= 60, f"README says {claimed}, real total is {total}"

    def test_it_names_the_protocol_level_alternatives(self) -> None:
        """A reader should know the cheaper fix exists before adopting this."""
        text = _text()
        assert "isError" in text
        assert "ToolMessage.status" in text
