"""The pre-commit hook has to be able to run.

``.pre-commit-config.yaml`` existed from the first commit and was never
installed, so every commit in this repository's history bypassed it. One of
them reached ``main`` red -- a test appended and pushed without re-running the
formatter.

Installing it exposed a second problem: the hook pinned ``ruff`` ten minor
versions behind the project's own, and that version could not parse this
project's ``pyproject.toml``. It failed on every commit for a reason unrelated
to the code, which is the worst kind of check -- one that everyone learns to
pass ``--no-verify`` to.

So the version is asserted here. A check that cannot run is not a check, and a
check that always fails is worse than none.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _ruff_hook_rev() -> str:
    text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    match = re.search(
        r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?([\d.]+)",
        text,
    )
    assert match, "the ruff hook must declare a rev"
    return match.group(1)


def _installed_ruff() -> str:
    """The version this environment resolves, which is what CI runs.

    Comparing the pin to the ``>=0.6`` floor in pyproject.toml proves nothing:
    the broken v0.6.9 satisfied it. CI runs ``uv run ruff``, which resolves the
    newest matching release, so that is the version the hook has to match.
    """
    import subprocess
    import sys

    output = subprocess.run(
        [sys.executable, "-m", "ruff", "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = re.search(r"(\d+\.\d+\.\d+)", output)
    assert match, f"could not read a version from {output!r}"
    return match.group(1)


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


class TestTheHookCanParseTheProject:
    def test_the_pinned_ruff_matches_the_one_this_environment_runs(self) -> None:
        """Not "newer than the floor" -- the broken v0.6.9 satisfied that and
        still could not parse pyproject.toml. The hook has to run the same
        version as CI, or it disagrees with the check it exists to pre-empt."""
        assert _ruff_hook_rev() == _installed_ruff()

    def test_the_pin_is_a_real_release(self) -> None:
        """A rev like ``main`` would silently change under contributors."""
        assert re.fullmatch(r"\d+\.\d+\.\d+", _ruff_hook_rev())

    def test_the_format_hook_is_present(self) -> None:
        """``ruff format`` is the check most easily forgotten, because the code
        still works without it. It is the one that caught us."""
        text = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "ruff-format" in text


class TestContributorsAreToldToInstallIt:
    def test_contributing_says_to_install_the_hook(self) -> None:
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        assert "pre-commit install" in text

    def test_it_is_in_the_setup_block_not_an_aside(self) -> None:
        """Buried below the fold is the same as absent."""
        text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        setup = text[text.index("## Setup") : text.index("## The checks CI runs")]
        assert "pre-commit install" in setup
