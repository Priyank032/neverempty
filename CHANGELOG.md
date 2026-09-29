# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in any minor
release. The public API is exactly what `toolproof/__init__.py` exports via
`__all__`; nothing else is public. The trace and case JSON Schemas are
versioned independently of the package.

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-09-29

The library is complete against the design doc. What it does **not** yet carry
is measured results for the two dogfood agents: those need hand-written ground
truth (330 routing labels, ~33 fault labels, ~60 judge labels), and a label
written by a model would make every published number a measure of one model
agreeing with another. `toolproof coverage` refuses to run a suite until they
exist, and the README's Numbers section is generated from committed reports, so
it stays empty rather than carrying a placeholder.

### Added
- `toolproof.evals`: the eval harness. `SuiteSpec` and `check_coverage` refuse a
  suite that cannot support its own numbers — an empty split, a branch below the
  per-branch floor, or a label naming a route the agent cannot produce.
  `toolproof coverage` exits non-zero on any of those, so an unlabelled suite
  fails CI rather than publishing a rate over an empty denominator. This
  library's own headline rule, applied to its own dataset.
- `NextRoleAdapter`: routing cases go through `route_probe` (zero branch
  executions, so a 330-case routing run cannot send an email), anything else
  executes the graph with stubs bound. `BRANCHES` is the eleven intents the live
  router actually returns, not the design doc's seven: the doc predates the
  router growing four intents and includes a `clarify` branch that does not
  exist, and a branch list disagreeing with the router would score a whole intent
  zero forever without ever looking like an error. That resizes the frozen split
  to 330, because eleven branches at the doc's own 30-per-branch floor is 330 and
  210 would leave 19 per branch — noise by its own Wilson table.
- `RecordingStub`: returns an explicit `Ok`, never `None`. A stub returning
  `None` would raise under strict mode, and a falsy one would be the ambiguity
  this library exists to remove.
- YojanaKhoj consistency scoring with the doc's five categories, each carrying
  its own denominator. Pooling them would let a structurally protected category
  hide an exposed one: the hard rule filter removes rule-false schemes before the
  LLM sees them, so verdict contradictions there are near zero by construction,
  and the null cases are where the risk lives.
- `evaluateRule` returns `null` for a missing field, and that null survives into
  `expect.items` untouched. "Cannot evaluate" and "ineligible" are different
  claims. A definite yes or no on a null rule is counted as an overclaim, and a
  high `llm_confidence` there is its own separate finding.
- Both of matchEngine's LLM-degraded fallback paths are excluded from scoring on
  both sides of the bridge. They emit a fixed bilingual string and set
  `llm_confidence` to either a rescaled ranking score or a hardcoded constant;
  scoring that text would measure the fallback string, and scoring that number as
  a confidence would measure the score function.
- `toolproof import`: validates Trace v1 JSONL from another language and refuses
  an export that cannot prove the match cache was bypassed, or whose LLM error
  rate is above 10%. A cached run can serve one persona an explanation written
  for another in the same age and income bucket, which contaminates exactly the
  rate being measured; the refusal says that is a failure of the export rather
  than of the agent.
- `integrations/node/`: the export script and its test, written for the
  YojanaKhoj repo rather than committed into it. Exercised against that repo's
  real `ruleEvaluator` and its 103 real scheme files.
- `JevJudge`: a typed-decision backend behind `JudgeModel`, rendering the same
  JSON the text judge's parser reads so the parser, the retries and the cache are
  shared rather than duplicated. Not run and not endorsed: access is waitlisted
  and the vendor's calibration claim is a claim. It is deliberately absent from
  `toolproof.__all__`, because an unrun, unvalidated backend is not part of the
  promised API. Every confidence it sees is recorded, so the reliability curve
  can settle the calibration question with data.
- `toolproof readme`: renders the published-numbers table from committed reports
  and `--check` fails CI on drift. Judge-derived numbers are cut below kappa 0.6,
  and an unmeasured kappa is not a passing one. An incomplete run publishes
  nothing and is named under "Not published".

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
- `scorers`: the built-in scorer set, used as factories —  `route()`,
  `tool_selection()`, `forbidden_tools()`, `arguments()`, `facts()`,
  `forbidden_claims()`, `failure_handling()` and `false_alarm()`. A scorer that
  cannot decide returns `None` or raises `ScorerError`; none of them ever
  returns `passed=False` because data was missing.
- Route prediction follows one fixed source order — `graph.next` from the probe
  span, then the last branch node span before `finalize`, then
  `final_output.route` — because the three can disagree, and an unstated
  preference would make two runs of one suite mean different things. With no
  source at all the case is unscored, not failed.
- Seven argument match modes: `exact`, `normalized` (casefold plus Unicode
  NFKC, so a decomposed Devanagari nukta matches its precomposed form),
  `set` (order-free, works on unhashable members), `numeric` (with `tol`;
  a `bool` is not a number), `regex`, `present` (a question about the key, so
  `""`, `[]` and `0` are all present) and `date` (timezone-aware, naive read as
  UTC). There is no `judge` mode by design.
- A redacted argument scores as not-applicable with `note: "redacted"` and
  leaves the denominator, so enabling redaction cannot silently lower a score.
  Same for arguments an adapter never recorded.
- `failure_handling` reports one of `misreport`, `reported` or `ignored`, from
  per-language pattern lists covering English, Hindi and Hinglish. An answer the
  lists cannot read is not-applicable pending the judge, never counted in either
  direction, and the matched pattern travels in the detail so the lists' own
  precision stays measurable. `PATTERNS_VERSION` is recorded because changing
  the patterns changes the headline number.
- `false_alarm`, the mirror metric: claiming failure when a tool truthfully
  returned nothing. Computable only because `Empty` and `Err` are distinct
  statuses.
- `collapse()` and `COLLAPSE_RULES`: majority for capability metrics, any-hit
  for safety metrics, median for fact recall. Every built-in scorer declares its
  rule explicitly; an unmeasurable repeat leaves the denominator rather than
  counting as a failure.

- `metrics.stats`: pure-Python Wilson intervals, exact one-sided McNemar, seeded
  bootstrap (10k resamples) and nearest-rank percentiles. No SciPy, so the gate
  runs offline with pydantic as the only dependency. Every function returns
  `None` rather than a number when there is nothing to measure: an empty sample
  has no percentile and no interval, and `0` would read as a real result.
- The McNemar test is one-sided deliberately. The gate exists to catch the
  candidate getting worse, so an improvement can never fail a build.
- `metrics.aggregate`: repeats collapse to one observation per case *before*
  aggregation. Counting three repeats as three cases would inflate every n and
  shrink every interval, making a 70-case suite look like a 210-case one.
- `Report.metrics`, `confusion` and `counts.unstable` are now filled by the
  runner. The confusion matrix has the documented extra `unscored` column, so
  its rows always sum to the case count.
- `compare`: pairs two reports on `case_id`, reports per-metric deltas with both
  intervals, the McNemar counts and p-value, and the cases that flipped each
  way. Refuses to compare when either report is incomplete or the resolved
  models differ, unless `--allow-model-change`. An unknown cost on either side
  makes the delta unknown, never a 0% change.
- `gate`: the five documented exit codes, ordered so that invalid input (4)
  outranks a must-pass failure (2), which outranks inconclusive (3), which
  outranks a statistical regression (1). A quality verdict computed from a
  broken input is worse than no verdict, and a run too noisy to attribute a
  delta must not report one.
- A floor on a metric that was not measured is a warning, never a breach. The
  library applies its own rule to itself: missing must not look like failure.
- `render_markdown` and `render_gate`: the renderer reads only the report
  structure, so a table it produces can only contain numbers that are in a
  committed report. It prints "not measured" for `applicable=0`, suppresses
  percentages below n=10 while still showing the counts, and labels anything
  below n=50 indicative.
- `toolproof.toml` loader: strict, so an unknown key is an error. A misspelled
  `max_cost_usd` that loaded silently would remove the budget cap from a live
  run. The design doc's own config file loads unchanged, and `config_hash()` is
  over the parsed values so reformatting does not read as a change.
- CLI: `compare`, `gate`, `render` and `baseline promote`. `gate` returns the
  doc's exit codes unchanged rather than remapping them to the CLI's own.
  `baseline promote` refuses an incomplete report and refuses to overwrite an
  existing baseline without `--force`.
- `Runner` takes `now` and `report_id`, the two fields that otherwise stop two
  runs of identical input from being byte-identical. A golden report is
  reproducible with the public API rather than only under a monkeypatch.
- Golden tests: a committed report JSON and its rendered Markdown, byte-compared
  in CI. Any change to the report shape, the statistics or the key ordering now
  shows up as a diff a reviewer has to approve.

### Fixed

- Trace v1 caught three bugs in the Node exporter before it ran once: missing
  required `usage` and `cost`, missing required `env` fields, and a trace-level
  status of `"error"`, which is not a member of the enum (an LLM failure inside
  the target is a `target_error`). That is the cross-language contract earning
  its keep — the same schema validates on both sides.
- The `calibration` scorer was added to the runtime in M12 but never registered
  in the config's known-scorer list, so a config naming it was rejected. The
  config validator caught it, which is what it is for.

- `CalibrationResult.model_id` collided with pydantic's protected `model_`
  namespace without declaring `protected_namespaces=()`. pydantic warns on such
  a field, this project turns warnings into errors, and the warning fires on
  some 2.x versions and not others — so on pydantic 2.9, well inside the
  declared `>=2.7,<3` range, importing `toolproof` at all raised. The local
  suite stayed green because the pinned dev environment happened to use a
  version where it does not fire. A test now walks every model in the package
  and fails on any undeclared `model_*` field, so the next one cannot reach a
  single CI leg unnoticed.

- `collapse()` dropped the scorer's own detail, so the confusion matrix had no
  labels for its rows. Declared per-case fields (`expected`, `predicted`,
  `outcome`, `correct`) now survive the collapse, and only when every repeat
  agrees: a disagreeing prediction is precisely an unstable case, and inventing
  one value for it would hide that.

- `ClaimJudge`: two versioned calls at temperature 0. Claim extraction splits a
  free-text answer into at most 12 atomic claims; verification labels each one
  `supported`, `contradicted` or `not_in_evidence`, one call per claim, so the
  verifier never sees the other claims, the conversation, or the agent identity.
- Malformed output is retried twice with the same input and then labelled
  `judge_error`. A label outside the three is a parse failure, not a fourth
  opinion: a judge that guesses when it could not parse its own output is worse
  than no judge. `judge_error` is never cached, so a transient outage does not
  become permanent.
- The self-preference check refuses to construct a judge whose model family
  matches the agent's, and the runner re-checks it against the model the run
  actually resolves to. It fails closed: an unrecognised model id with no
  declared family is refused, and a declared family that disagrees with the one
  inferred from the id is refused rather than one being picked.
- Claim and evidence are delimited data, and the closing delimiters are escaped
  so content cannot break out of its own block. The prompt instruction is the
  second line of defence, not the only one.
- Four prompt-injection fixtures ship with the library (`load_injection_fixtures`):
  a direct override, a fake system turn, a delimiter break-out, and one in Hindi.
  Each fixture's human label is what a judge that ignored the instruction would
  say, so a flip is visible as `injections_held` falling below `injections`.
- `self_consistency()` samples one claim repeatedly and reports whether the
  labels agree, bypassing the cache: sampling a cache three times would report
  perfect consistency and measure nothing.
- Verdicts are cached on SHA-256 of claim, evidence, prompt version and model.
  The claim *id* is deliberately excluded, so two cases making the same claim
  about the same evidence share one verdict. A prompt-version or model change
  invalidates the cache, because a cached verdict from an older prompt answers a
  different question.
- `JudgeModel` protocol plus `ScriptedJudge`, a deterministic offline fake. Core
  ships no provider SDK, so every judge behaviour is testable without credentials
  and a tracing-only user gains no dependency.
- `cohens_kappa` and `calibrate`: agreement against human labels, the 3x3 matrix,
  and precision and recall for `contradicted` specifically. Kappa rather than raw
  agreement because raw agreement is inflated by the base rate — on a set that is
  80% `supported`, a judge that always answers `supported` scores 80% agreement
  and kappa 0. Below 0.6 the result is marked not publishable.
- `judge_error` outcomes are excluded from kappa and counted separately. Folding
  them in as wrong labels would understate agreement and hide an outage.
- Agreement is sliceable by language, so a failure in Hindi cannot hide behind
  English.
- `toolproof judge calibrate`: prints kappa, the matrix, contradicted precision
  and recall, per-language agreement and the injection tally. Exits 0 for a low
  kappa (a bad judge is a valid measurement) unless `--fail-below-threshold`.
- A judge error rate above 2% makes the run `degraded`. A degraded run is still
  `complete` and still gates on its deterministic metrics; only the judge-derived
  ones are excluded, because a flaky judge is not a reason to stop checking
  routing. An *unscored* case still makes the run `incomplete`, which is the
  stronger claim and wins.
- `scorers.facts(judge=...)` and `scorers.forbidden_claims(judge=...)` decide
  `judge`-mode expectations. A `judge_error` leaves the expectation unmeasured,
  never missing: an outage is not evidence that the agent omitted a fact. For
  forbidden claims that rule is safety-critical — a claim the judge could not
  decide is unmeasured, never clean.

- `scorers.calibration()` and the reliability curve: predictions bucketed by
  stated confidence, accuracy measured per bucket, and expected calibration
  error over the whole sample. Accuracy alone cannot see this — two agents with
  identical accuracy differ completely in whether their confidence can be acted
  on, and a clarify-branch or judge-escalation threshold set from a vendor's
  calibration claim rather than a measured curve is set from nothing.
- Ten buckets of width 0.1, half-open `[low, high)` with the top bucket closed,
  so a confidence of exactly 1.0 has a home and no prediction lands in two
  buckets. ECE is sample-weighted: an unweighted mean of bucket gaps would let a
  bucket holding two predictions move the headline number as much as one holding
  two hundred. The bucket count is printed with every ECE, because ECE is a
  function of the bucketing and two curves bucketed differently are not
  comparable.
- The scorer never decides correctness. It names an existing scorer as its
  ground-truth source, so the curve and that scorer's metric can never disagree
  about whether a given case was right. `passed` is always `None`: a confident
  wrong answer is already a route failure, and failing it twice would gate twice
  on one event.
- A missing confidence is not a zero one. No `structured` output, no confidence
  key, a boolean, a string or an explicit null all read as not applicable, so an
  agent that stated nothing never appears in the bottom bucket. A value outside
  `[0, 1]` raises rather than being rescaled — a curve computed from an
  undeclared scale is not reproducible.
- Repeats collapse before bucketing, so 70 cases at 3 repeats contribute 70
  predictions rather than 210. The collapsed confidence is the median, so
  repeats stating 0.90, 0.91 and 0.89 survive; a case whose repeats disagree
  about whether the agent was *right* drops out, because an unstable answer
  gives its stated confidence no single event to have been right about.
- `render_reliability`: the curve as a Markdown section, with the same
  suppression rule as every other metric — a bucket below n=10 prints its hit
  count, not a percentage, and empty buckets are omitted rather than printed as
  zero. The overconfidence direction is named in words, because a signed number
  alone invites a sign error in the reading.

### Deferred

- `toolproof.langchain.wrap(base_tool)` is deferred past 0.1.0. The `@tool`
  decorator and the LangChain callback handler already cover both authoring a
  tool and tracing one the framework invokes.

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

[Unreleased]: https://github.com/priyank-agrawal/toolproof/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/priyank-agrawal/toolproof/releases/tag/v0.1.0
[0.0.1]: https://github.com/priyank-agrawal/toolproof/releases/tag/v0.0.1
