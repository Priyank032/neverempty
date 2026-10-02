"""The async runner: dataset and target in, Report out.

Every ambiguous situation has exactly one defined behaviour, because the
alternative is a report that silently means something different from run to
run. Those behaviours are the doc's runner-semantics table, and each one has a
test named after its row.

Three invariants hold throughout:

- **Unscored is never a smaller sample.** A case that timed out, whose target
  raised, or whose every scorer raised is ``unscored``, the run is
  ``incomplete``, and the gate treats that as a failure of the run.
- **Cancellation propagates.** A budget abort cancels pending work; an
  interrupt writes a partial report and re-raises. Neither is recorded as an
  agent failure.
- **Output is canonically ordered.** Results are sorted by
  ``(case_id, repeat)``, so two runs of one suite produce diffable JSON.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from neverempty.core.classify import is_uncatchable
from neverempty.core.faults import fault_scope
from neverempty.core.stubs import (
    registered_side_effect_tools,
    registered_tools,
    stub_scope,
)
from neverempty.core.trace import Env, Trace, TraceError, truncate_utf8
from neverempty.dataset.case import Case
from neverempty.dataset.loader import Dataset
from neverempty.metrics.aggregate import (
    build_confusion,
    build_metrics,
    count_crashed,
    count_unstable,
)
from neverempty.report.report import (
    CaseOutcome,
    Costs,
    Counts,
    JudgeInfo,
    Latency,
    Report,
    Score,
)
from neverempty.runner.cache import CacheMode, ResponseCache
from neverempty.tracer.sinks import MemorySink
from neverempty.tracer.tracer import Tracer

DEFAULT_CASE_TIMEOUT_S = 120.0


class PreflightError(RuntimeError):
    """A preflight check failed. Nothing has executed."""


class BudgetExceededError(RuntimeError):
    """Internal signal that the cost cap was reached."""


class _RunCancelledError(RuntimeError):
    """Internal signal that a target cancelled itself.

    Wrapped rather than re-raised as ``CancelledError``, because the runner
    cancels its own tasks to stop work and the two must stay distinguishable.
    Unwrapped back into a real ``CancelledError`` for the caller.
    """

    def __init__(self, outcome: Any, trace: Any) -> None:
        super().__init__("target cancelled")
        self.outcome = outcome
        self.trace = trace


class ScorerError(RuntimeError):
    """A scorer could not decide and said so explicitly."""


@runtime_checkable
class Scorer(Protocol):
    """What a scorer looks like.

    A scorer that cannot decide returns ``None`` or raises ``ScorerError``. It
    never returns ``passed=False`` because data was missing: that is the same
    rule this library exists to enforce on agents, applied to itself.
    """

    name: str
    requires: frozenset[str]

    def score(self, case: Case, trace: Trace) -> Score | None: ...


Target = Callable[[Case, Tracer], Any]
"""``async (Case, Tracer) -> None``; sets output on the run."""


def _scorer_name(scorer: Any, index: int) -> str:
    return str(getattr(scorer, "name", None) or getattr(scorer, "__name__", f"scorer_{index}"))


def _scorer_requires(scorer: Any) -> frozenset[str]:
    return frozenset(getattr(scorer, "requires", frozenset()))


def _expectation_present(case: Case, requirement: str) -> bool:
    """Whether a case declares the expectation a scorer needs.

    ``None`` means the case never asked, which makes the scorer not applicable.
    An empty list means it asked for none, which is a real expectation.
    """
    if not requirement.startswith("expect."):
        return True
    field = requirement.split(".", 1)[1]
    return getattr(case.expect, field, None) is not None


class Runner:
    """Turns a dataset and a target into a reproducible, budget-capped Report."""

    def __init__(
        self,
        *,
        target: Target,
        scorers: Sequence[Any],
        tracer: Tracer | None = None,
        repeats: int = 1,
        concurrency: int = 8,
        case_timeout_s: float = DEFAULT_CASE_TIMEOUT_S,
        max_cost_usd: float | None = None,
        cache: str | Path | None = None,
        mode: CacheMode = "live",
        seed: int | None = None,
        stubs: Mapping[str, Callable[..., Any]] | None = None,
        output_path: str | Path | None = None,
        force: bool = False,
        expect_split_hash: str | None = None,
        require_cache: bool = False,
        suite_version: int = 1,
        env_overrides: dict[str, Any] | None = None,
        config_hash: str | None = None,
        judge: Any = None,
        now: str | None = None,
        report_id: str | None = None,
        _force_cost: float | None = None,
    ) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")

        self.target = target
        self.scorers = list(scorers)
        self.repeats = repeats
        self.concurrency = concurrency
        self.case_timeout_s = case_timeout_s
        self.max_cost_usd = max_cost_usd
        self.mode = mode
        self.seed = seed
        self.stubs = dict(stubs or {})
        self.output_path = Path(output_path) if output_path else None
        self.force = force
        self.expect_split_hash = expect_split_hash
        self.require_cache = require_cache
        self.suite_version = suite_version
        self.env_overrides = dict(env_overrides or {})
        self.config_hash = config_hash

        # Determinism seam. A report carries a wall-clock timestamp and a random
        # id, which are exactly the two fields that stop two runs of identical
        # input from being byte-identical. Pinning them is what makes a golden
        # report reproducible by a user, not only by a monkeypatched test.
        self.judge = judge
        self.now = now
        self.report_id = report_id

        self.tracer = tracer or Tracer(sink=MemorySink())
        self.cache = ResponseCache(cache, mode=mode) if cache else None
        self.random = random.Random(seed)  # noqa: S311 - sampling, not crypto

        # Test seam: the empty pricing table reports every cost as unknown, so
        # budget behaviour needs a way to be exercised without inventing prices.
        self._force_cost = _force_cost

    async def run(self, dataset: Dataset) -> Report:
        """Execute every case, score it, and return the Report."""
        self._preflight(dataset)
        return await self._execute(dataset, dataset.cases, existing=[])

    async def resume(self, report: Report, dataset: Dataset) -> Report:
        """Re-run only the cases that are missing or unscored, then merge.

        A resumed report is ``complete`` only when every case has a scored
        outcome, so resuming does not launder an incomplete run into a clean one.
        """
        self._preflight(dataset)

        scored = {o.case_id for o in report.outcomes if o.scored}
        keep = [o for o in report.outcomes if o.scored]
        todo = [case for case in dataset.cases if case.id not in scored]

        return await self._execute(dataset, todo, existing=keep, resumed_from=report.report_id)

    def _preflight(self, dataset: Dataset) -> None:
        """Refuse to start if any check fails. Nothing has executed yet."""
        suites = dataset.suites
        if len(suites) > 1:
            raise PreflightError(
                f"dataset spans {len(suites)} suites ({sorted(suites)}); one report "
                f"describes one suite, and averaging two would hide both"
            )

        if self.expect_split_hash is not None:
            actual = dataset.split_hash()
            if actual != self.expect_split_hash:
                raise PreflightError(
                    f"test split hash mismatch: config records "
                    f"{self.expect_split_hash}, dataset hashes to {actual}. Editing "
                    f"the test split invalidates the baseline; bump suite_version."
                )

        if self.output_path is not None and self.output_path.exists() and not self.force:
            raise PreflightError(
                f"{self.output_path} already exists. Pass force=True to overwrite; "
                f"silently replacing a committed report would lose the number a "
                f"README links to."
            )

        # A fault is applied by the ``@tool`` wrapper, so one declared on a
        # function that is not wrapped does nothing at all: the case runs
        # normally, scores normally, and the suite publishes a
        # misreport-as-empty rate over opportunities that never existed. Doc
        # row 993 names that trap, and it is this library's own headline bug
        # pointed at the number the whole project exists to produce.
        declared = {fault.tool for case in dataset.cases for fault in case.faults}
        if declared:
            wrapped = registered_tools()
            missing = sorted(declared - wrapped)
            if missing:
                raise PreflightError(
                    f"{len(missing)} tool(s) have declared faults but are not "
                    f"wrapped with @tool: {', '.join(missing)}. A fault on an "
                    f"unwrapped function never fires, so the suite would report "
                    f"a misreport-as-empty rate measured over zero injections. "
                    f"Wrap them with @tool, or remove the faults from those "
                    f"cases. Imported wrapped tools: "
                    f"{', '.join(sorted(wrapped)) or '(none)'}."
                )

        unstubbed = sorted(registered_side_effect_tools() - set(self.stubs))
        if unstubbed:
            raise PreflightError(
                f"side-effecting tool(s) {unstubbed} have no stub bound. An eval run "
                f"must never be able to touch the outside world, so pass "
                f"stubs={{'name': fn}} for each."
            )

        # After the side-effect check: a run that could email a real recruiter
        # must fail on that, not on a judge misconfiguration.
        self._preflight_judge()

        # Only a replay run that actually needs recorded responses is blocked.
        # A target with no provider calls (a fixture agent, a rule engine) is a
        # legitimate replay run with nothing to read.
        if self.mode == "replay" and self.cache is None and self.require_cache:
            raise PreflightError(
                "mode='replay' with require_cache=True needs a cache directory: "
                "replay never calls a provider, so there would be nothing to read"
            )

    async def _execute(
        self,
        dataset: Dataset,
        cases: Sequence[Case],
        *,
        existing: Sequence[CaseOutcome],
        resumed_from: str | None = None,
    ) -> Report:
        outcomes: list[CaseOutcome] = list(existing)
        traces: list[Trace] = []

        # The tracer cannot know how the runner was configured, so it defaulted
        # to concurrency 1, no seed and no fault profile while the report said
        # otherwise -- two artifacts of one run disagreeing. The fault profile
        # is the one that matters most: a trace claiming null undoes the
        # protection ``_fault_profile`` exists for, letting a misreport-as-empty
        # number be read as coming from a run that injected nothing.
        #
        # Overrides the caller passed to the Tracer win: they were set
        # deliberately and on purpose.
        self.tracer.env_overrides = {
            "concurrency": self.concurrency,
            "seed": self.seed,
            "fault_profile": _fault_profile(dataset),
            "mode": "replay" if self.mode == "replay" else "live",
            **self.tracer.env_overrides,
        }

        spent = _Budget(self.max_cost_usd)
        semaphore = asyncio.Semaphore(self.concurrency)
        aborted = False

        jobs = [(case, repeat) for case in cases for repeat in range(self.repeats)]

        async def one(case: Case, repeat: int) -> None:
            async with semaphore:
                if spent.exhausted:
                    raise BudgetExceededError
                outcome, trace = await self._run_case(case, repeat)
                outcomes.append(outcome)
                if trace is not None:
                    traces.append(trace)
                # Null costs are passed too: an unknown cost is not a free one.
                spent.add(outcome.cost_usd)

        tasks = [asyncio.create_task(one(case, repeat)) for case, repeat in jobs]
        try:
            await self._gather(tasks)
        except BudgetExceededError:
            aborted = True
        except _RunCancelledError as exc:
            # A target cancelled itself: write what completed to the partial
            # path, never to the final one, then surface a real CancelledError.
            # A partial report at the real path would be read as a whole run.
            report = self._build(
                dataset, outcomes, traces, spent, aborted=False, resumed_from=resumed_from
            )
            self._write_partial(report)
            raise asyncio.CancelledError from exc
        except BaseException as exc:
            if is_uncatchable(exc):
                report = self._build(
                    dataset, outcomes, traces, spent, aborted=False, resumed_from=resumed_from
                )
                self._write_partial(report)
            raise

        report = self._build(
            dataset, outcomes, traces, spent, aborted=aborted, resumed_from=resumed_from
        )
        if self.output_path is not None:
            report.save(self.output_path)
        return report

    async def _gather(self, tasks: list[asyncio.Task[None]]) -> None:
        """Await every task; on the first failure cancel the rest and re-raise.

        Cancelling siblings before re-raising matters: a task left pending
        surfaces later as "Task exception was never retrieved", and on an
        interrupt it would run after the partial report was already written.
        """
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            failure: BaseException | None = None
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None and failure is None:
                    failure = exc
            if failure is not None:
                for remaining in pending:
                    remaining.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                raise failure

    async def _run_case(self, case: Case, repeat: int) -> tuple[CaseOutcome, Trace | None]:
        """Execute one case at one repeat and score it."""
        outcome = CaseOutcome(
            case_id=case.id,
            repeat=repeat,
            must_pass=case.must_pass,
            tags=list(case.tags),
        )

        trace: Trace | None = None
        timed_out = False
        self_cancelled = False
        target_error: BaseException | None = None

        run_context = self.tracer.run(
            case_id=case.id, repeat=repeat, suite=case.suite, tags={"split": case.split}
        )
        try:
            async with run_context as run:
                with fault_scope(case.fault_specs()), stub_scope(self.stubs):
                    # A deadline is enforced with a task plus wait(), not
                    # wait_for(): wait_for cancels the inner task and reports
                    # CancelledError, which is indistinguishable from a real
                    # cancellation the caller must see. A timeout is a case
                    # outcome; a cancellation is the run ending.
                    task = asyncio.ensure_future(self._call_target(case))
                    done, _ = await asyncio.wait({task}, timeout=self.case_timeout_s)
                    if not done:
                        # The deadline passed. Cancel the work and record a
                        # timeout: that is a case outcome, not a run ending.
                        timed_out = True
                        task.cancel()
                        with contextlib.suppress(BaseException):
                            await task
                    elif task.cancelled():
                        # The target cancelled itself. A cancellation ends the
                        # run and is never an agent failure, so it must reach
                        # the caller -- but raising here would be caught by the
                        # gather loop as an ordinary task cancellation, which is
                        # what the runner does to *stop* work. Flagged instead.
                        self_cancelled = True
                    else:
                        exc = task.exception()
                        if exc is not None:
                            if is_uncatchable(exc):
                                raise exc
                            target_error = exc
                if timed_out:
                    run.set_status("timeout")
                elif target_error is not None:
                    run.set_status("target_error")
                    run.set_error(type(target_error).__name__, str(target_error))
        finally:
            trace = run_context.trace

        if self_cancelled:
            raise _RunCancelledError(outcome, trace)

        if trace is not None:
            outcome.trace_id = trace.trace_id
            outcome.trace_status = trace.status
            outcome.duration_ms = trace.duration_ms
            outcome.cost_usd = trace.cost.usd
            outcome.cost_unknown_reason = trace.cost.unknown_reason
            if self._force_cost is not None:
                outcome.cost_usd = self._force_cost
                outcome.cost_unknown_reason = None
            if trace.error is not None:
                outcome.error = trace.error

        if timed_out:
            # Scorers are not run: there is no complete trace to score.
            return outcome, trace
        if target_error is not None:
            outcome.error = outcome.error or TraceError(
                kind=type(target_error).__name__,
                message=truncate_utf8(str(target_error), 2048),
            )
            return outcome, trace

        if trace is not None:
            await self._score(case, trace, outcome)
        return outcome, trace

    async def _call_target(self, case: Case) -> None:
        result = self.target(case, self.tracer)
        if inspect.isawaitable(result):
            await result

    async def _score(self, case: Case, trace: Trace, outcome: CaseOutcome) -> None:
        """Run every applicable scorer, isolating failures to that scorer."""
        for index, scorer in enumerate(self.scorers):
            name = _scorer_name(scorer, index)
            requires = _scorer_requires(scorer)
            if any(not _expectation_present(case, item) for item in requires):
                outcome.not_applicable.append(name)
                continue

            try:
                verdict = await self._invoke_scorer(scorer, case, trace)
            except BaseException as exc:
                if is_uncatchable(exc):
                    raise _RunCancelledError(outcome, trace) from exc
                outcome.scorer_errors[name] = f"{type(exc).__name__}: {exc}"
                continue

            if verdict is None:
                # Not applicable, never a failure.
                outcome.not_applicable.append(name)
            else:
                outcome.scores[name] = verdict

        outcome.scored = bool(outcome.scores)

    async def _invoke_scorer(self, scorer: Any, case: Case, trace: Trace) -> Score | None:
        # ``score_async`` is how a judge-backed scorer reaches its judge. A
        # scorer offering both is called through the async one, because the sync
        # path deliberately skips judge-mode expectations rather than guessing.
        method = getattr(scorer, "score_async", None) or getattr(scorer, "score", scorer)
        result = method(case, trace)
        if inspect.isawaitable(result):
            awaited: Score | None = await result
            return awaited
        typed: Score | None = result
        return typed

    def _build(
        self,
        dataset: Dataset,
        outcomes: list[CaseOutcome],
        traces: list[Trace],
        spent: _Budget,
        *,
        aborted: bool,
        resumed_from: str | None,
    ) -> Report:
        suite = next(iter(dataset.suites), "unknown.suite")
        split = next(iter(dataset.splits), "dev")

        case_ids = {outcome.case_id for outcome in outcomes}
        scored_ids = {o.case_id for o in outcomes if o.scored}
        unscored = len(case_ids - scored_ids)
        expected = len(dataset.cases)
        complete = not aborted and unscored == 0 and len(case_ids) == expected

        status = "ok"
        if aborted:
            status = "aborted_budget"
        elif not complete:
            status = "incomplete"
        elif self.judge is not None and getattr(self.judge, "degraded", False):
            # The judge failed often enough to taint its own numbers. The run is
            # still complete: every case was scored, and the deterministic
            # metrics in this report are real measurements. Only the
            # judge-derived ones are excluded from the gate.
            status = "degraded"

        # Only measured cases. A case whose trace never materialized has no
        # duration, and counting it as 0 would publish a percentile lower than
        # any latency the run actually observed.
        durations = sorted(
            outcome.duration_ms for outcome in outcomes if outcome.duration_ms is not None
        )
        known_costs = [o.cost_usd for o in outcomes if o.cost_usd is not None]

        seed = self.seed if self.seed is not None else 0
        scorer_names = [_scorer_name(scorer, index) for index, scorer in enumerate(self.scorers)]
        expected_labels = {
            case.id: case.expect.route.label
            for case in dataset.cases
            if case.expect.route is not None
        }

        fields: dict[str, Any] = {}
        if self.report_id is not None:
            fields["report_id"] = self.report_id

        return Report(
            **fields,
            suite=suite,
            suite_version=self.suite_version,
            split=split,
            created_at=self.now
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            complete=complete,
            status=status,  # type: ignore[arg-type]
            env=self._env(_fault_profile(dataset)),
            config_hash=self.config_hash,
            counts=Counts(
                cases=len(case_ids),
                repeats=self.repeats,
                scored=len(scored_ids),
                unscored=unscored,
                unstable=count_unstable(outcomes),
                crashed=count_crashed(outcomes),
            ),
            metrics=build_metrics(outcomes, seed=seed, scorer_names=scorer_names),
            outcomes=outcomes,
            confusion=build_confusion(outcomes, expected_labels=expected_labels),
            judge=self._judge_info(),
            costs=Costs(
                total_usd=sum(known_costs) if known_costs else None,
                mean_usd=(sum(known_costs) / len(known_costs)) if known_costs else None,
                unknown_count=sum(1 for o in outcomes if o.cost_usd is None),
                pricing_version=self.tracer.pricing.version,
                budget_usd=self.max_cost_usd,
            ),
            latency=Latency(
                samples_ms=durations,
                p50_ms=_nearest_rank(durations, 0.50),
                p95_ms=_nearest_rank(durations, 0.95),
                concurrency=self.concurrency,
            ),
            resumed_from=resumed_from,
            split_hash=dataset.split_hash(),
            traces=traces,
        )

    def _judge_info(self) -> JudgeInfo:
        """The report's judge block, from the judge itself when there is one."""
        if self.judge is None:
            return JudgeInfo()
        info = getattr(self.judge, "info", None)
        if callable(info):
            built: JudgeInfo = info()
            return built
        return JudgeInfo()

    def _preflight_judge(self) -> None:
        """Refuse a judge whose family matches the model the run will use.

        ``ClaimJudge`` already refuses a matching *configured* family. This checks
        the family the run will actually resolve to, because a config can name one
        model and the target can call another.

        Ordered after the side-effect check on purpose: a run that could email a
        real recruiter must fail on that, not on a judge misconfiguration.
        """
        if self.judge is None:
            return

        judge_family = getattr(self.judge, "family", None)
        if not isinstance(judge_family, str):
            return

        from neverempty.judge.judge import infer_family

        for model_id in self.env_overrides.get("resolved_models", []):
            inferred = infer_family(str(model_id))
            if inferred is not None and inferred == judge_family:
                raise PreflightError(
                    f"judge family {judge_family!r} matches the target's resolved "
                    f"model {model_id!r}. A model judging its own family's output "
                    f"is the bias with the strongest evidence behind it, so the "
                    f"run refuses to start."
                )

    def _env(self, fault_profile: str | None = None) -> Env:
        from neverempty import __version__

        fields: dict[str, Any] = {
            "fault_profile": fault_profile,
            "neverempty_version": __version__,
            "pricing_version": self.tracer.pricing.version,
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "concurrency": self.concurrency,
            "seed": self.seed,
            "mode": "replay" if self.mode == "replay" else "live",
        }
        fields.update(self.env_overrides)
        return Env(**fields)

    def _write_partial(self, report: Report) -> None:
        """Write to ``<path>.partial.json``, never to the final path."""
        if self.output_path is None:
            return
        partial = self.output_path.with_suffix("")
        partial = partial.with_name(f"{partial.name}.partial.json")
        report.save(partial)


class _Budget:
    """Tracks spend against a cap.

    An unknown cost is never counted as zero. With the empty pricing table every
    cost is null, so a budget that treated null as free would never fire — which
    is exactly the missing-versus-zero bug, in the budget.
    """

    __slots__ = ("cap", "spent", "unknown")

    def __init__(self, cap: float | None) -> None:
        self.cap = cap
        self.spent = 0.0
        self.unknown = 0

    def add(self, amount: float | None) -> None:
        """Record one case's cost, known or not.

        The caller previously skipped nulls entirely, so with an unpriced model
        the budget counted nothing and never fired -- the class's own docstring
        named that bug and the call site reintroduced it.
        """
        if amount is None:
            self.unknown += 1
        else:
            self.spent += amount

    @property
    def unenforceable(self) -> bool:
        """A cap was set and at least one case's cost could not be computed.

        Spending is then unbounded below the cap: a run can burn 50M tokens on
        an unpriced model and stay under any budget, because none of it counts.
        Refusing is the honest answer -- running without a cap is a choice, but
        believing you have one that is not there is the bug this library exists
        to prevent, pointed at the wallet.
        """
        return self.cap is not None and self.unknown > 0

    @property
    def exhausted(self) -> bool:
        return self.cap is not None and (self.spent >= self.cap or self.unknown > 0)


def _fault_profile(dataset: Dataset) -> str | None:
    """A stable description of the faults this run injected.

    Recorded in ``env`` so a misreport-as-empty number can never be read as
    coming from a run that injected nothing.
    """
    kinds = sorted(
        {f"{fault.tool}:{fault.kind}" for case in dataset.cases for fault in case.faults}
    )
    return ",".join(kinds) if kinds else None


def _nearest_rank(sorted_values: list[int], quantile: float) -> int | None:
    """Nearest-rank percentile. ``None`` for an empty sample, never 0."""
    if not sorted_values:
        return None
    index = max(int(quantile * len(sorted_values) + 0.5) - 1, 0)
    return sorted_values[min(index, len(sorted_values) - 1)]


__all__ = [
    "DEFAULT_CASE_TIMEOUT_S",
    "BudgetExceededError",
    "PreflightError",
    "Runner",
    "Scorer",
    "ScorerError",
    "Target",
]
