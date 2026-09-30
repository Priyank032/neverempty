"""Command line entry point.

``gate`` is the one subcommand whose exit code is a contract: CI passes or
fails a build on it, and the codes come from the doc's table rather than from
this module's own error conventions. So ``gate`` returns them unchanged, and
every other subcommand uses ``EXIT_OK`` / ``EXIT_INVALID`` / ``EXIT_USAGE``.

``run`` executes the suites a config declares; it is the primary entry point.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neverempty import __version__
from neverempty.dataset.loader import Dataset, DatasetError, expand_paths
from neverempty.evals.importer import LLM_ERROR_RATE_CEILING
from neverempty.judge.judge import VERIFICATION_PROMPT_VERSION
from neverempty.report.gate import GateConfig
from neverempty.report.gate import compare as compare_reports
from neverempty.report.gate import gate as run_gate
from neverempty.report.render import render_gate, render_markdown
from neverempty.report.report import Report

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neverempty",
        description="Evaluate and trace tool-calling LLM agents.",
    )
    parser.add_argument("--version", action="version", version=f"neverempty {__version__}")

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

    _add_run(subcommands)
    _add_coverage(subcommands)
    _add_import(subcommands)
    _add_readme(subcommands)
    _add_compare(subcommands)
    _add_gate(subcommands)
    _add_render(subcommands)
    _add_baseline(subcommands)
    _add_judge(subcommands)

    return parser


def _add_run(subcommands: Any) -> None:
    runner = subcommands.add_parser(
        "run",
        help="run the suites a config declares and write a report",
        description=(
            "Resolve the configured target, build the named scorers, and run each "
            "suite. One report per suite. Exits non-zero when a run is incomplete, "
            "because an incomplete run is a failure of the run rather than a "
            "smaller sample."
        ),
    )
    runner.add_argument("config", metavar="CONFIG", help="path to neverempty.toml")
    runner.add_argument(
        "--out",
        metavar="PATH",
        help=(
            "where to write the report. With one suite this is the file; with "
            "several it is a directory. Defaults to a 'reports' directory beside "
            "the config."
        ),
    )
    runner.add_argument(
        "--suite",
        metavar="NAME",
        help="run only this suite, by name",
    )
    runner.add_argument(
        "--mode",
        choices=("live", "replay", "record"),
        help="override [run].mode, so one config serves a live run and a replay",
    )
    runner.add_argument(
        "--repeats",
        type=int,
        metavar="N",
        help="override [run].repeats",
    )
    runner.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    """Run each configured suite. One report per suite."""
    import asyncio

    from neverempty.config import ConfigError, load_config
    from neverempty.runner.runner import PreflightError, Runner, ScorerError

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID

    suites = list(config.suites)
    if args.suite:
        suites = [suite for suite in suites if suite.name == args.suite]
        if not suites:
            available = ", ".join(repr(s.name) for s in config.suites)
            print(
                f"no suite named {args.suite!r} in {args.config}; this config declares {available}",
                file=sys.stderr,
            )
            return EXIT_INVALID

    try:
        target = _resolve_entrypoint(config.target.entrypoint)
        stubs = _resolve_stubs(config.target.stubs)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID

    destination = _report_destination(args, len(suites))
    worst = EXIT_OK

    for suite in suites:
        try:
            dataset = Dataset.load(suite.path, split=suite.split)
        except DatasetError as exc:
            print(f"{suite.path}: {exc}", file=sys.stderr)
            for problem in exc.problems[:20]:
                print(f"  {problem}", file=sys.stderr)
            return EXIT_INVALID

        try:
            scorer_objects = _build_scorers(suite.scorers)
        except (AttributeError, TypeError) as exc:
            print(f"suite {suite.name!r}: {exc}", file=sys.stderr)
            return EXIT_INVALID

        run_config = config.run
        runner = Runner(
            target=target,
            scorers=scorer_objects,
            repeats=args.repeats or run_config.repeats,
            concurrency=run_config.concurrency,
            case_timeout_s=run_config.case_timeout_s,
            max_cost_usd=run_config.max_cost_usd,
            cache=run_config.cache,
            mode=args.mode or run_config.mode,
            seed=run_config.seed,
            stubs=stubs,
            suite_version=suite.suite_version,
            config_hash=config.config_hash(),
        )

        try:
            report = asyncio.run(runner.run(dataset))
        except PreflightError as exc:
            # A preflight refusal is the point, not a crash: an eval that could
            # touch the outside world must not start.
            print(f"preflight refused suite {suite.name!r}: {exc}", file=sys.stderr)
            return EXIT_INVALID
        except ScorerError as exc:
            print(f"suite {suite.name!r}: {exc}", file=sys.stderr)
            return EXIT_INVALID

        path = _report_path(destination, suite.name, len(suites))
        report.save(path)
        print(
            f"{suite.name}: status={report.status} "
            f"{report.counts.scored}/{report.counts.cases} scored -> {path}"
        )

        if not report.complete:
            # An incomplete run is a failure of the run, never a smaller sample,
            # so the exit code says so rather than leaving it to be noticed.
            print(
                f"  {report.counts.unscored} case(s) unscored; an incomplete run "
                f"is a failure of the run, not a smaller sample",
                file=sys.stderr,
            )
            worst = EXIT_INVALID

    return worst


def _resolve_entrypoint(spec: str) -> Any:
    """Import ``module:name`` and return it, requiring it to be callable."""
    resolved = _resolve_object(spec, "[target].entrypoint")
    if not callable(resolved):
        raise TypeError(f"[target].entrypoint {spec!r} is not callable")
    return resolved


def _resolve_object(spec: str, field: str) -> Any:
    """Import ``module:name`` and return whatever it is.

    Separate from ``_resolve_entrypoint`` because the two fields want different
    things: an entrypoint must be callable, while ``[target].stubs`` is normally
    a plain dict. Reusing the callable check here rejected a perfectly good
    mapping.
    """
    if ":" not in spec:
        raise ValueError(f"{field} must be 'module:name', got {spec!r}")
    module_name, _, attribute = spec.partition(":")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute, None)
    if resolved is None:
        raise AttributeError(f"{module_name!r} has no attribute {attribute!r}")
    return resolved


def _resolve_stubs(spec: str | None) -> dict[str, Any] | None:
    """Import the stub mapping, when the config names one.

    A callable is accepted as a factory, so a caller can build stubs lazily —
    but a dict is the normal case, and it is checked first because a dict is
    not callable and a Mapping subclass might be.
    """
    if not spec:
        return None
    mapping = _resolve_object(spec, "[target].stubs")
    if not isinstance(mapping, Mapping) and callable(mapping):
        mapping = mapping()
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"[target].stubs {spec!r} resolved to {type(mapping).__name__}, "
            f"not a mapping of tool name to stub"
        )
    return dict(mapping)


def _build_scorers(names: Sequence[str]) -> list[Any]:
    """Construct each named scorer from the scorers package.

    The config validator already rejects an unknown name, so a miss here means
    the registry and the validator have drifted apart, which is worth saying
    loudly rather than skipping.
    """
    from neverempty import scorers as registry

    built: list[Any] = []
    for name in names:
        factory = getattr(registry, name, None)
        if factory is None or not callable(factory):
            raise AttributeError(
                f"scorer {name!r} passed config validation but has no factory in "
                f"neverempty.scorers; the registry and the validator have drifted"
            )
        built.append(factory())
    return built


def _report_destination(args: argparse.Namespace, suite_count: int) -> Path:
    if args.out:
        return Path(args.out)
    return Path(args.config).resolve().parent / "reports"


def _report_path(destination: Path, suite: str, suite_count: int) -> Path:
    """Where one suite's report lands.

    With several suites the destination is a directory, so two suites cannot
    silently overwrite each other's report.
    """
    if suite_count == 1 and destination.suffix == ".json":
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    return destination / f"{suite}.json"


def _add_coverage(subcommands: Any) -> None:
    coverage = subcommands.add_parser(
        "coverage",
        help="report per-branch label coverage for each suite in a config",
        description=(
            "Read a config, load each suite's dataset, and report how many "
            "labelled cases each branch has. Exits non-zero when a suite is not "
            "ready to publish a number, so an unlabelled split fails CI rather "
            "than running and reporting a rate over an empty denominator."
        ),
    )
    coverage.add_argument("config", metavar="CONFIG", help="path to neverempty.toml")
    coverage.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="output format (markdown for a CI job summary)",
    )
    coverage.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "report coverage but exit 0 even when a suite is short; for seeing "
            "the labelling backlog without failing the build"
        ),
    )
    coverage.set_defaults(handler=_coverage)


def _coverage(args: argparse.Namespace) -> int:
    """Per-branch coverage for every suite the config declares."""
    from neverempty.config import ConfigError, load_config
    from neverempty.evals.suites import MIN_PER_BRANCH, SuiteSpec, check_coverage

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_INVALID

    branches = _declared_branches(config)
    if branches is None:
        print(
            "no branch list available: coverage needs the suite's branches to "
            "tell a mislabelled route from a real one. Set [target].entrypoint "
            "to an adapter exposing BRANCHES, or run 'neverempty validate'.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    blocks: list[str] = []
    ready = True
    for suite in config.suites:
        spec = SuiteSpec(
            name=suite.name,
            branches=branches,
            min_per_branch=MIN_PER_BRANCH,
            split=suite.split,
        )
        try:
            cases = Dataset.load(suite.path).cases
        except DatasetError as exc:
            # An empty or absent file is the starting state for labelling, and
            # reporting the backlog is exactly what this command is for. Any
            # other dataset error is a real problem and still stops the run.
            if not _is_empty_dataset(suite.path):
                print(f"{suite.path}: {exc}", file=sys.stderr)
                return EXIT_INVALID
            cases = []
        report = check_coverage(cases, spec)
        ready = ready and report.ok
        blocks.append(report.render())

    separator = "\n\n" if args.format == "markdown" else "\n\n" + "-" * 60 + "\n\n"
    print(separator.join(blocks))

    if ready or args.allow_incomplete:
        return EXIT_OK
    return EXIT_INVALID


def _is_empty_dataset(path: str) -> bool:
    """True when the file is absent or holds nothing but blank lines."""
    target = Path(path)
    if not target.exists():
        return True
    return not target.read_text(encoding="utf-8").strip()


def _declared_branches(config: Any) -> tuple[str, ...] | None:
    """The branch list from the configured adapter module.

    Read from the target rather than from the config, because the branches are a
    property of the agent: duplicating them in TOML would create a second place
    to forget when the router grows an intent.
    """
    entrypoint = config.target.entrypoint
    module_name = entrypoint.partition(":")[0]
    for candidate in (module_name, module_name.rpartition(".")[0]):
        if not candidate:
            continue
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue
        branches = getattr(module, "BRANCHES", None)
        if isinstance(branches, tuple) and branches:
            return branches
    # Fall back to the NextRole adapter, which is the one this repo ships.
    from neverempty.evals.nextrole import BRANCHES

    return BRANCHES


def _add_import(subcommands: Any) -> None:
    importer = subcommands.add_parser(
        "import",
        help="import traces exported by another language",
        description=(
            "Read Trace v1 JSONL produced by a non-Python exporter, validate "
            "every line against the same models the Python tracer uses, and "
            "check the export's own preconditions: that the match cache was "
            "bypassed, and that the LLM error rate is low enough to publish "
            "from. Refuses rather than importing data that cannot support a "
            "number."
        ),
    )
    importer.add_argument("traces", metavar="TRACES", help="traces.jsonl")
    importer.add_argument(
        "--cases",
        metavar="CASES",
        help="cases.jsonl, to pair each trace with its ground truth",
    )
    importer.add_argument(
        "--error-ceiling",
        type=float,
        default=LLM_ERROR_RATE_CEILING,
        metavar="RATE",
        help=(
            f"refuse above this share of traces carrying an LLM error "
            f"(default {LLM_ERROR_RATE_CEILING:.2f})"
        ),
    )
    importer.add_argument(
        "--allow-cached",
        action="store_true",
        help=(
            "accept an export that does not record a cache bypass. A cached "
            "run can serve one persona an explanation written for another in "
            "the same age and income bucket, so this is off by default"
        ),
    )
    importer.set_defaults(handler=_import)


def _import(args: argparse.Namespace) -> int:
    from neverempty.evals.importer import import_export

    report = import_export(
        args.traces,
        cases_path=args.cases,
        error_ceiling=args.error_ceiling,
        require_cache_bypass=not args.allow_cached,
    )
    print(report.render())
    return EXIT_OK if report.ok else EXIT_INVALID


def _add_readme(subcommands: Any) -> None:
    readme = subcommands.add_parser(
        "readme",
        help="render the README's Numbers section from committed reports",
        description=(
            "Build the published-numbers table from report files. Every row "
            "links back to the report it came from, carries its sample size and "
            "interval, and judge-derived numbers are cut when kappa is below the "
            "publishing threshold. A number with no report behind it cannot be "
            "produced, because there is nothing to produce it from."
        ),
    )
    readme.add_argument("reports", nargs="*", metavar="REPORT", help="committed report JSON files")
    readme.add_argument(
        "--check",
        metavar="README",
        help=(
            "compare against the Numbers section of this file and exit non-zero "
            "when they differ, so a stale published number fails CI"
        ),
    )
    readme.set_defaults(handler=_readme)


def _readme(args: argparse.Namespace) -> int:
    from neverempty.report.readme import load_reports, render_readme_numbers

    # Expanded here for the same reason ``validate`` does it: the README's own
    # documented command is a glob, and PowerShell and cmd do not expand it, so
    # the pattern reached ``open()`` verbatim and died on [Errno 22].
    if args.reports:
        expanded = expand_paths(args.reports)
        if not expanded:
            # stderr, not ``_fail``: this command's output is piped into a
            # README, so an error on stdout would be published as the Numbers
            # section. Rendering an empty section from an empty match would
            # publish "no numbers" as though it were a measured result.
            print(
                f"no files matched: {' '.join(args.reports)}; check the path "
                f"or the glob. An empty match is not a pass.",
                file=sys.stderr,
            )
            return EXIT_INVALID
        reports: list[str] | list[Path] = expanded
    else:
        reports = []

    try:
        loaded = load_reports(reports)
    except OSError as exc:
        print(f"cannot read a report: {exc}", file=sys.stderr)
        return EXIT_INVALID

    rendered = render_readme_numbers(loaded)

    if not args.check:
        print(rendered)
        return EXIT_OK

    target = Path(args.check)
    try:
        current = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {target}: {exc}", file=sys.stderr)
        return EXIT_INVALID

    existing = _numbers_section(current)
    if existing is None:
        print(
            f"{target} has no generated region to check. Wrap the numbers in "
            f"{BEGIN_MARKER} and {END_MARKER}.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    if existing.strip() == rendered.strip():
        print(f"{target}: Numbers section is up to date")
        return EXIT_OK

    print(
        f"{target}: the Numbers section does not match the committed reports. "
        f"Re-render it with 'neverempty readme <reports> > section.md'. A published "
        f"number that no longer matches its report is the failure this check "
        f"exists to catch.",
        file=sys.stderr,
    )
    return EXIT_INVALID


BEGIN_MARKER = "<!-- neverempty:numbers:begin -->"
END_MARKER = "<!-- neverempty:numbers:end -->"


def _numbers_section(text: str) -> str | None:
    """The generated region, between the two markers.

    Marker-delimited rather than heading-delimited so hand-written prose can sit
    in the same section without the check reading it as drift. Only what is
    between the markers is generated, and only that is compared.
    """
    start = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    return text[start + len(BEGIN_MARKER) : end].strip()


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


def _add_judge(subcommands: Any) -> None:
    judge = subcommands.add_parser(
        "judge",
        help="judge calibration against human labels",
        description="Measure the judge's agreement with human labels.",
    )
    actions = judge.add_subparsers(dest="judge_command", metavar="<action>")
    calibrate = actions.add_parser(
        "calibrate",
        help="report Cohen's kappa, the 3x3 matrix, and contradicted precision",
        description=(
            "Compare judge labels against human labels and report Cohen's kappa, "
            "the 3x3 confusion matrix, and precision and recall for "
            "'contradicted' specifically, which is the label that drives the "
            "headline number.\n"
            "\n"
            "Kappa rather than raw agreement: on a set that is 80% supported, a "
            "judge that always answers supported scores 80% agreement and has "
            "learned nothing. Kappa corrects for that."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    calibrate.add_argument("labels", metavar="LABELS", help="human-labelled JSONL")
    calibrate.add_argument(
        "--replay",
        required=True,
        metavar="PATH",
        help=(
            "JSONL of judge labels to compare against, keyed by id. Use a "
            "recorded judge run; live judging needs a provider binding."
        ),
    )
    calibrate.add_argument(
        "--fail-below-threshold",
        action="store_true",
        help=(
            "exit non-zero when kappa is below the publish threshold; for a "
            "release workflow, where an exploratory run wants exit 0"
        ),
    )
    calibrate.add_argument("--out", metavar="PATH", help="write the result JSON here")
    calibrate.add_argument("--json", action="store_true", dest="as_json")
    calibrate.set_defaults(handler=_judge_calibrate)
    judge.set_defaults(handler=_judge_help, as_json=False)


def _judge_help(args: argparse.Namespace) -> int:
    print("usage: neverempty judge calibrate LABELS --replay PATH [--out PATH] [--json]")
    return EXIT_USAGE


def _judge_calibrate(args: argparse.Namespace) -> int:
    from neverempty.judge.calibration import (
        KAPPA_PUBLISH_THRESHOLD,
        calibrate,
        load_calibration,
    )

    try:
        cases = load_calibration(args.labels)
        replayed = load_calibration(args.replay)
    except ValueError as exc:
        print(f"FAIL {exc}")
        return EXIT_INVALID

    by_id = {case.id: case.human_label for case in replayed}
    missing = [case.id for case in cases if case.id not in by_id]
    if missing:
        # Pairing is by id, so a replay missing a case would silently shrink the
        # set the kappa was computed on.
        print(
            f"FAIL replay is missing {len(missing)} case(s) present in the "
            f"labels: {', '.join(missing[:10])}"
        )
        return EXIT_INVALID

    result = calibrate(
        cases,
        judge_labels=[by_id[case.id] for case in cases],
        model_id=args.replay,
        prompt_version=VERIFICATION_PROMPT_VERSION,
    )

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")

    if args.as_json:
        print(result.model_dump_json(indent=2))
    else:
        print(_render_calibration(result, KAPPA_PUBLISH_THRESHOLD))

    if args.fail_below_threshold and not result.publishable:
        return EXIT_INVALID
    return EXIT_OK


def _render_calibration(result: Any, threshold: float) -> str:
    labels = ("supported", "contradicted", "not_in_evidence")
    kappa = "not measured" if result.kappa is None else f"{result.kappa:.3f}"
    agreement = "not measured" if result.agreement is None else f"{result.agreement:.1%}"

    lines = [
        f"Judge calibration: {result.scored} scored, {result.errors} judge_error, "
        f"{result.cases} total",
        "",
        f"Cohen's kappa : {kappa}",
        f"Raw agreement : {agreement}  (inflated by the base rate; kappa corrects it)",
        "",
        "Confusion matrix (rows human, columns judge):",
        "",
        "| human \\ judge | " + " | ".join(labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for human in labels:
        row = result.matrix.get(human, {})
        lines.append(f"| {human} | " + " | ".join(str(row.get(j, 0)) for j in labels) + " |")

    precision = result.contradicted_precision
    recall = result.contradicted_recall
    lines += [
        "",
        "contradicted (the label that drives the headline number):",
        f"  precision : {'not measured' if precision is None else f'{precision:.1%}'}",
        f"  recall    : {'not measured' if recall is None else f'{recall:.1%}'}",
    ]

    if result.by_language:
        lines += ["", "Agreement by language:"]
        for language, stats in sorted(result.by_language.items()):
            lines.append(f"  {language} : {stats['agreement']:.1%}  (n={int(stats['n'])})")

    if result.injections:
        lines += [
            "",
            f"Injection fixtures: {result.injections_held}/{result.injections} held "
            f"(the judge did not flip the label)",
        ]

    lines += [""]
    if result.publishable:
        lines.append(
            f"Kappa is at or above {threshold}, so judge-derived numbers may be published."
        )
    else:
        lines.append(
            f"Kappa is below {threshold}: judge-derived numbers are NOT publishable. "
            f"Report only the deterministic checks."
        )
    return "\n".join(lines)


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
        from neverempty.config import ConfigError, load_config

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
    from neverempty.report.gate import EXIT_CODES, GateResult

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
    print("usage: neverempty baseline promote CANDIDATE --to PATH [--force]")
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
