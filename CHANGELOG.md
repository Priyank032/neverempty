# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in any minor
release. The public API is exactly what `toolproof/__init__.py` exports via
`__all__`; nothing else is public. The trace and case JSON Schemas are
versioned independently of the package.

## [Unreleased]

### Added

- `Ok`, `Empty` and `Err`: the tool result union, discriminated on `status`
  and frozen. `truncated` is a flag on `Ok`, not a fourth variant.
- `to_model()`, the model-facing renderer. An `Err` renders as an explicit
  instruction not to claim absence, and never carries the error message or the
  exception class into the prompt.
- `@tool`: wraps sync and async functions so they return a `ToolResult` and
  never raise. Preserves name, docstring, signature and type hints, so
  LangChain and OpenAI tool-schema generation still work.
- Strict mode, on by default: returning `None`, `[]` or `{}` without declaring
  `empty_when` or `never_empty` raises `AmbiguousEmptyError`. There is no
  falsiness inference, so `0`, `False` and `""` stay `Ok`.
- Default error classification table, plus a `classify_error` override hook.
  HTTP status is read structurally, so core needs no provider SDK.
- `asyncio.CancelledError`, `KeyboardInterrupt` and `SystemExit` propagate
  untouched, including out of a predicate.
- Fault injection hook: `fault_scope` and `FaultSpec`, keyed on
  `(tool_name, call_index)` through a context variable. An error-shaped fault
  never calls the real dependency.
- Retries, off by default; only `retryable` errors retry, with exponential
  backoff and jitter.
- `Trace`, `Span`, `Usage`, `Cost`, `Env` and `FinalOutput`: the Trace v1
  models. Unknown keys are preserved on read, so another language can add
  fields; a `schema_version` above this reader's is rejected outright.
- `Tracer`: contextvars-based capture, so a `@tool` call or an instrumented LLM
  call inside `tracer.run()` attaches to the right trace and parent span under
  asyncio concurrency, with no plumbing. Works as `with` and `async with`.
- `SpanHandle.record_usage()` writes OpenTelemetry GenAI attribute names and
  keeps the resolved model separate from the requested one, so model drift is
  detectable. Token counts come from provider usage fields only.
- `MemorySink`, `JsonlSink`, `MultiSink`, `NullSink`. A sink failure never
  raises into the agent; it is counted on `tracer.dropped_traces` and logged at
  ERROR, because silent data loss is the bug class this library exists to
  prevent.
- `redact.keys()` and friends, applied before anything reaches a sink. A
  matched value becomes `[REDACTED:<hash>]`, so equal values stay comparable
  without either being written.
- `Pricing`: versioned table with an `as_of` date and source URL per model. The
  default table is **empty**, so every cost is `null` with a reason until you
  supply prices. No price ships that cannot be cited.
- `schemas/trace.v1.json`, generated from the models and committed. Output is
  normalised so it is byte-identical across pydantic 2.9 and 2.13; CI diffs it.
- `scripts/gen_schemas.py` to regenerate the committed schemas.
- `Case`: the Case v1 dataset model. Unknown keys are **rejected**, the
  opposite of the trace rule, because a misspelled expectation is an absent
  one and an absent expectation reports not-applicable rather than failing.
- Every `expect` key is optional but at least one is required: a case that
  expects nothing can never fail, so it measures nothing. Absent (`None`) stays
  distinguishable from "asked for none" (`[]`).
- Route labels are free strings. The library never knows a target's branch
  names, so adding one is a dataset edit, not a library change.
- `ArgExpectation` supports `exact`, `normalized`, `set`, `numeric`, `regex`,
  `present` and `date`. There is deliberately no `judge` mode for arguments: if
  an argument needs semantic matching, the dataset is underspecified.
- Regex patterns compile at load time, so a bad pattern fails on the dataset
  rather than halfway through a run.
- `provenance` is required on the test split, and `generated_from_rules` must
  pin a `source_commit`: generated ground truth drifts when its source moves.
- `Dataset.load()` reports **every** bad line with its file and line number in
  one pass, rejects duplicate ids within a suite, and refuses to produce an
  empty dataset.
- `Dataset.split_hash()`: content hash of the test split, order-independent.
  Reordering or reformatting the file leaves it unchanged; editing, adding or
  deleting a test case changes it. Dev cases are excluded, since tuning on dev
  is allowed and must not invalidate a baseline.
- `toolproof validate`: schema, id uniqueness and split-hash checking, with
  `--expect-split-hash` for CI, `--json` for machines, and a non-zero exit when
  a glob matches nothing, so an empty match never reads as a pass.
- `schemas/case.v1.json`, generated and committed alongside the trace schema.
- `route_probe()` (extra: `langgraph`): compiles a graph with `interrupt_before`
  on every branch node, runs to the interrupt, and reports the pending branch.
  A routing eval therefore reads the route without executing a branch, which is
  a safety requirement rather than an optimisation.
- The route comes from the edge the graph took, not the state field. When the
  two disagree the edge wins and the disagreement is recorded on the span as
  `graph.route_disagreement`, because it is a finding about the agent.
- A renamed or missing branch fails when the probe is built, listing the
  available nodes. An interrupt that never fires lets the branch run for real,
  so this cannot be allowed to fail quietly. `assert_branches_exist()` exposes
  the same check for a target's import-time guard.
- Each probe run gets its own checkpointer thread, so concurrent cases cannot
  resume each other's graph.
- `tracer.langchain_handler()`: node, tool and LLM spans for any LangChain
  runnable. Parenting comes from LangChain's own run tree, since a callback's
  start and end can arrive in different contexts.
- `tracer.instrument_openai()` and `tracer.instrument_bedrock()`: usage,
  resolved model, finish reason and latency by duck typing, so neither SDK is
  imported and core stays at one dependency. Both are idempotent.
- No prompt or completion text is ever recorded by any adapter.
- Every callback is guarded: a LangChain payload shape this version does not
  recognise costs a span, never the agent's request. Cancellation still
  propagates.
- `examples/fixture_agent`: a LangGraph orchestrator with seven branches and a
  fake LLM, used by the integration tests. No network, no provider SDK.
- `Runner`: async execution with repeats, a concurrency semaphore, per-case
  timeouts, a cost budget, preflight checks and resume. Every row of the
  runner-semantics table has a test named after it.
- `Report`, `CaseOutcome`, `Score` and `Metric`: the report structure. A metric
  with `applicable=0` cannot carry a value, so "not measured" can never render
  as `0`.
- Unscored is never a smaller sample: a timeout, a raising target, or a case
  where every scorer raised leaves the run `incomplete`.
- A scorer that raises costs only itself; other scorers still apply. A scorer
  returning `None` is not-applicable, never a failure.
- An unknown cost is never counted as zero against the budget, so a run with no
  pricing table does not get a silently infinite budget.
- A budget abort cancels pending work, keeps completed cases, and writes a valid
  `aborted_budget` report. An interrupt writes `<path>.partial.json` and never
  the final path, then re-raises.
- Results are sorted by `(case_id, repeat)`, so two runs of one suite produce
  diffable JSON regardless of concurrency.
- `resume()` re-runs only missing or unscored cases, merges without duplicating,
  and records `resumed_from`. A resumed report is `complete` only when every
  case has a scored outcome.
- `stub_scope()` and the side-effect registry: `@tool(side_effect=True)`
  registers at decoration time, and the runner refuses to start unless every
  such tool has a stub bound. A stubbed call is traced identically to a real
  one, with `tool.stubbed` recorded so a report cannot mistake one for the other.
- `ResponseCache`: record/replay keyed on SHA-256 over the documented fields.
  Editing a prompt changes the key by design, so a replay test cannot mask a
  prompt edit. A replay miss is a hard error naming the key; replay never falls
  back to a provider.

## [0.0.1] - 2026-09-22

Name reservation and repository skeleton. No public API yet.

### Added

- Repository skeleton: `pyproject.toml` (hatchling), ruff, `mypy --strict`,
  pytest with coverage.
- `ci.yml`: ruff, ruff format, `mypy --strict`, pytest on Python 3.10 to 3.13,
  coverage gate, wheel build, and a clean-venv install check.
- `release.yml`: PyPI trusted publishing over OIDC with no stored API token,
  plus a GitHub release rendered from this file.
- Empty typed package that imports cleanly, ships `py.typed`, and depends only
  on pydantic v2.
- `toolproof --version` console script.

[Unreleased]: https://github.com/priyank-agrawal/toolproof/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/priyank-agrawal/toolproof/releases/tag/v0.0.1
