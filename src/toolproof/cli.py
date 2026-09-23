"""Command line entry point.

``run``, ``compare``, ``gate``, ``baseline`` and ``import`` land in M7.
``validate`` is here because a dataset has to be checkable before anything can
be run against it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toolproof import __version__
from toolproof.dataset.loader import Dataset, DatasetError, expand_paths

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolproof",
        description="Evaluate and trace tool-calling LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"toolproof {__version__}")

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    validate = subcommands.add_parser(
        "validate",
        help="check dataset files: schema, id uniqueness, split hash",
        description=(
            "Validate JSONL dataset files. Reports every bad line with its line "
            "number, rejects duplicate ids, and prints the test split hash."
        ),
    )
    validate.add_argument("paths", nargs="+", metavar="PATH", help="files or globs")
    validate.add_argument(
        "--expect-split-hash",
        metavar="SHA256",
        help=(
            "fail if the test split hash differs from this value; use in CI to "
            "catch an edited test split"
        ),
    )
    validate.add_argument(
        "--split",
        choices=("dev", "test"),
        help="validate only this split",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a machine-readable summary",
    )
    validate.set_defaults(handler=_validate)

    return parser


def _validate(args: argparse.Namespace) -> int:
    paths = expand_paths(args.paths)
    if not paths:
        _fail(
            args,
            f"no files matched: {' '.join(args.paths)}",
            # A glob matching nothing must never read as "all datasets valid".
            hint="check the path or the glob; an empty match is not a pass",
        )
        return EXIT_INVALID

    results: list[dict[str, Any]] = []
    ok = True

    for path in paths:
        entry: dict[str, Any] = {"file": str(path), "errors": []}
        try:
            dataset = Dataset.load(path, split=args.split)
        except (DatasetError, FileNotFoundError) as exc:
            ok = False
            problems = getattr(exc, "problems", None) or [str(exc)]
            entry["errors"] = problems
            results.append(entry)
            continue

        entry["counts"] = dataset.counts()
        entry["cases"] = len(dataset)
        entry["suites"] = sorted(dataset.suites)
        digest = dataset.split_hash()
        entry["test_split_hash"] = digest

        if args.expect_split_hash:
            try:
                dataset.verify_split_hash(args.expect_split_hash)
            except DatasetError as exc:
                ok = False
                entry["errors"] = [str(exc)]

        results.append(entry)

    if args.as_json:
        print(json.dumps({"ok": ok, "files": results}, indent=2))
        return EXIT_OK if ok else EXIT_INVALID

    _print_human(results)
    return EXIT_OK if ok else EXIT_INVALID


def _print_human(results: list[dict[str, Any]]) -> None:
    for entry in results:
        name = Path(entry["file"]).name
        if entry["errors"]:
            print(f"FAIL {name}")
            for problem in entry["errors"]:
                for line in str(problem).splitlines():
                    print(f"     {line}")
            continue

        counts = entry["counts"]
        summary = ", ".join(f"{count} {split}" for split, count in sorted(counts.items()))
        plural = "" if entry["cases"] == 1 else "s"
        print(f"OK   {name}: {entry['cases']} case{plural} ({summary})")
        digest = entry["test_split_hash"]
        if digest:
            print(f"     test split hash: {digest}")
        else:
            print("     no test split in this file")


def _fail(args: argparse.Namespace, message: str, *, hint: str = "") -> None:
    if args.as_json:
        print(json.dumps({"ok": False, "error": message, "files": []}, indent=2))
        return
    print(f"FAIL {message}")
    if hint:
        print(f"     {hint}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK

    result: int = handler(args)
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
