"""The CLI has no subcommands yet; these cover the entry point contract."""

from __future__ import annotations

import pytest

from neverempty import __version__
from neverempty.cli import build_parser, main


def test_help_exits_zero_and_names_the_program(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    assert "neverempty" in capsys.readouterr().out


def test_version_flag_prints_the_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_argument_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--not-a-flag"])
    assert exc.value.code == 2


def test_parser_is_constructible_standalone() -> None:
    assert build_parser().prog == "neverempty"
