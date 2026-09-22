"""Command line entry point.

Subcommands land in M4 (``validate``) and M7 (``run``, ``compare``, ``gate``,
``baseline``, ``import``). For now the CLI only reports its version, so the
console script installed by the wheel is exercised from the first release.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from toolproof import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolproof",
        description="Evaluate and trace tool-calling LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"toolproof {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
