"""Command line entry point.

``gate`` is the one subcommand whose exit code is a contract: CI passes or
fails a build on it, and the codes come from the doc's table rather than from
this module's own error conventions. So ``gate`` returns them unchanged, and
every other subcommand uses ``EXIT_OK`` / ``EXIT_INVALID`` / ``EXIT_USAGE``.

``run`` and ``import`` land with the judge work; everything else is here.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from toolproof import __version__
from toolproof.dataset.loader import Dataset, DatasetError, expand_paths
from toolproof.report.gate import GateConfig
from toolproof.report.gate import compare as compare_reports
from toolproof.report.gate import gate as run_gate
from toolproof.report.render import render_gate, render_markdown
from toolproof.report.report import Report

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

    _add_compare(subcommands)
    _add_gate(subcommands)
    _add_render(subcommands)
    _add_baseline(subcommands)

    return parser


def _add_compare(subcommands: Any) -> None:
    compare = subcommands.add_parser(
        "compare",
        help="diff two reports: per-metric deltas, McNemar, flipped cases",
        description=(
            "Pair two reports on case_id and report what changed. Refuses to "
            "compare reports whose resolved models differ, or either of which "
            "is incomplete."
        ),
    )
    compare.add_argument("base", metavar="BASE", help="baseline report JSON")
    compare.add_argument("candidate", metavar="CANDIDATE", help="candidate report JSON")
    compare.add_argument(
        "--allow-model-change",
        action="store_true",
        help=(
            "compare even though the resolved model differs; the delta will "
            "include the model change as well as the change under review"
        ),
    )
    compare.add_argument("--primary", metavar="METRIC", help="metric the paired test uses")
    compare.add_argument("--json", action="store_true", dest="as_json")
    compare.set_defaults(handler=_compare)


def _add_gate(subcommands: Any) -> None:
    gate = subcommands.add_parser(
        "gate",
        help="fail a build on a regression (exit codes 0-4)",
        description=(
            "Decide whether a candidate report fails the build.\n"
            "\n"
            "Exit codes:\n"
            "  0  pass\n"
            "  1  significant regression (McNemar) or a floor breached\n"
            "  2  a must_pass case failed\n"
            "  3  inconclusive: unstable rate above the limit; rerun or reduce noise\n"
            "  4  invalid input: incomplete report, version mismatch, missing baseline\n"
            "\n"
            "Codes 1, 2 and 3 fail the build. Code 3 is labelled so it is not "
            "misread as a quality regression, and code 4 is an infrastructure "
            "failure rather than a verdict on the agent."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gate.add_argument("base", metavar="BASE", help="baseline report JSON")
    gate.add_argument("candidate", metavar="CANDIDATE", help="candidate report JSON")
    gate.add_argument("--config", metavar="TOML", help="read [gate] from this config file")
    gate.add_argument(
        "--floor",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "hard level, repeatable; NAME is a minimum and NAME_max is a "
            "maximum, matching the config file"
        ),
    )
    gate.add_argument("--paired-alpha", type=float, help="significance level (default 0.05)")
    gate.add_argument(
        "--max-unstable-rate",
        type=float,
        help="above this rate of unstable cases the result is inconclusive",
    )
    gate.add_argument("--primary", metavar="METRIC", help="metric the paired test uses")
    gate.add_argument("--allow-model-change", action="store_true")
    gate.add_argument("--summary", metavar="PATH", help="write the Markdown summary here")
    gate.add_argument("--json", action="store_true", dest="as_json")
    gate.set_defaults(handler=_gate)


def _add_render(subcommands: Any) -> None:
    render = subcommands.add_parser(
        "render",
        help="render a report as Markdown",
        description=(
            "Render a committed report as Markdown. The renderer reads only the "
            "report structure, so a table it produces can only contain numbers "
            "that are in the report."
        ),
    )
    render.add_argument("report", metavar="REPORT", help="report JSON")
    render.add_argument("--out", metavar="PATH", help="write here instead of stdout")
    render.set_defaults(handler=_render, as_json=False)


def _add_baseline(subcommands: Any) -> None:
    baseline = subcommands.add_parser(
        "baseline",
        help="manage the committed baseline",
        description="Promote a reviewed candidate report to the baseline.",
    )
    actions = baseline.add_subparsers(dest="baseline_command", metavar="<action>")
    promote = actions.add_parser(
        "promote",
        help="copy a candidate report to the baseline path",
        description=(
            "Promote a candidate to the baseline. Refuses an incomplete report, "
            "and refuses to overwrite an existing baseline without --force: a "
            "baseline is what every future run is judged against."
        ),
    )
    promote.add_argument("candidate", metavar="CANDIDATE", help="candidate report JSON")
    promote.add_argument("--to", required=True, metavar="PATH", help="baseline path")
    promote.add_argument("--force", action="store_true", help="overwrite an existing baseline")
    promote.set_defaults(handler=_baseline_promote, as_json=False)
    baseline.set_defaults(handler=_baseline_help, as_json=False)


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


def _load_report(path: str) -> Report | str:
    """The report, or a message saying why it could not be read."""
    target = Path(path)
    if not target.is_file():
        return f"report not found: {target}"
    try:
        return Report.load(target)
    except (ValueError, OSError) as exc:
        return f"{target.name} could not be read as a report: {exc}"


def _parse_floors(items: Sequence[str]) -> dict[str, float] | str:
    floors: dict[str, float] = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name:
            return f"--floor expects NAME=VALUE, got {item!r}"
        try:
            floors[name] = float(raw)
        except ValueError:
            return f"--floor value must be a number, got {raw!r} in {item!r}"
    return floors


def _gate_config(args: argparse.Namespace) -> GateConfig | str:
    config = GateConfig()
    if args.config:
        from toolproof.config import ConfigError, load_config

        try:
            loaded = load_config(args.config)
        except ConfigError as exc:
            return str(exc)
        config = loaded.gate.to_gate_config(seed=loaded.run.seed)

    floors = _parse_floors(args.floor)
    if isinstance(floors, str):
        return floors

    updates: dict[str, Any] = {}
    if floors:
        updates["floors"] = {**config.floors, **floors}
    if args.paired_alpha is not None:
        updates["paired_alpha"] = args.paired_alpha
    if args.max_unstable_rate is not None:
        updates["max_unstable_rate"] = args.max_unstable_rate
    if args.primary:
        updates["primary"] = args.primary
    return config.model_copy(update=updates) if updates else config


def _gate(args: argparse.Namespace) -> int:
    """Exit codes come from the doc's table and are returned unchanged."""
    base = _load_report(args.base)
    candidate = _load_report(args.candidate)
    for loaded in (base, candidate):
        if isinstance(loaded, str):
            # Invalid input is exit 4, never the CLI's own EXIT_INVALID: the
            # difference between "could not run" and "the agent regressed" is
            # the whole reason the code table exists.
            return _gate_invalid(args, loaded)

    config = _gate_config(args)
    if isinstance(config, str):
        print(f"FAIL {config}")
        return EXIT_USAGE

    assert isinstance(base, Report)
    assert isinstance(candidate, Report)
    result = run_gate(base, candidate, config, allow_model_change=args.allow_model_change)

    rendered = render_gate(result)
    print(result.to_json() if args.as_json else rendered, end="")

    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(rendered, encoding="utf-8", newline="\n")

    return result.exit_code


def _gate_invalid(args: argparse.Namespace, reason: str) -> int:
    from toolproof.report.gate import EXIT_CODES, GateResult

    result = GateResult(verdict="invalid", exit_code=EXIT_CODES["invalid"], reason=reason)
    print(result.to_json() if args.as_json else render_gate(result), end="")
    return result.exit_code


def _compare(args: argparse.Namespace) -> int:
    base = _load_report(args.base)
    candidate = _load_report(args.candidate)
    for loaded in (base, candidate):
        if isinstance(loaded, str):
            print(f"FAIL {loaded}")
            return EXIT_INVALID

    assert isinstance(base, Report)
    assert isinstance(candidate, Report)
    result = compare_reports(
        base,
        candidate,
        allow_model_change=args.allow_model_change,
        primary=args.primary,
    )

    if args.as_json:
        print(result.model_dump_json(indent=2))
        return EXIT_INVALID if result.refused else EXIT_OK

    if result.refused:
        print(f"FAIL refused to compare: {result.refusal_reason}")
        return EXIT_INVALID

    print(_render_compare(result))
    return EXIT_OK


def _render_compare(result: Any) -> str:
    lines = [
        f"Paired {result.paired} case(s); "
        f"{result.regressed_count} regressed, {result.improved_count} improved.",
        "",
        "| metric | base | candidate | delta |",
        "| --- | --- | --- | --- |",
    ]
    for delta in result.deltas:
        base_value = "-" if delta.base_value is None else f"{delta.base_value:.4f}"
        candidate_value = "-" if delta.candidate_value is None else f"{delta.candidate_value:.4f}"
        change = "-" if delta.delta is None else f"{delta.delta:+.4f}"
        lines.append(f"| {delta.name} | {base_value} | {candidate_value} | {change} |")

    if result.mcnemar is not None:
        test = result.mcnemar
        lines += [
            "",
            f"McNemar: b={test.b} (pass to fail), c={test.c} (fail to pass), "
            f"exact p={test.p_value:.4g}.",
        ]
    if result.regressed_shown:
        lines += ["", "Regressed: " + ", ".join(result.regressed_shown)]
    if result.improved_shown:
        lines += ["Improved: " + ", ".join(result.improved_shown)]
    return "\n".join(lines)


def _render(args: argparse.Namespace) -> int:
    loaded = _load_report(args.report)
    if isinstance(loaded, str):
        print(f"FAIL {loaded}")
        return EXIT_INVALID

    rendered = render_markdown(loaded)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")
    return EXIT_OK


def _baseline_promote(args: argparse.Namespace) -> int:
    loaded = _load_report(args.candidate)
    if isinstance(loaded, str):
        print(f"FAIL {loaded}")
        return EXIT_INVALID

    if not loaded.complete:
        print(
            f"FAIL refusing to promote an incomplete report (status="
            f"{loaded.status!r}); a baseline is what every future run is judged "
            f"against, so promoting a partial measurement would bake it into the gate"
        )
        return EXIT_INVALID

    target = Path(args.to)
    if target.exists() and not args.force:
        print(
            f"FAIL {target} already exists; pass --force to overwrite. Moving a "
            f"baseline silently moves the bar every future run is measured against."
        )
        return EXIT_INVALID

    loaded.save(target)
    print(f"OK   promoted {Path(args.candidate).name} to {target}")
    return EXIT_OK


def _baseline_help(args: argparse.Namespace) -> int:
    print("usage: toolproof baseline promote CANDIDATE --to PATH [--force]")
    return EXIT_USAGE


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
