# Architecture

Why the pieces are shaped the way they are. Every section here answers "why is
this not simpler", because in each case the simpler version has a specific failure
mode.

## The dependency rule

Core depends on **pydantic v2 only**.

The statistics are pure Python: Wilson is a closed form, the binomial tail is a
short sum, the bootstrap is a loop over a seeded `random.Random`. No NumPy, no
SciPy.

That is not minimalism for its own sake. A CI gate that needs a scientific stack
is a gate that someone eventually disables when the install breaks, and a disabled
gate measures nothing. The same reasoning puts every provider SDK behind an extra
and keeps the judge usable with an offline fake.

```
toolproof/
  core/          results, @tool, faults, trace models      <- pydantic only
  tracer/        contextvars capture, sinks, redaction, pricing
  dataset/       Case v1, loader, split hashing
  runner/        async execution, repeats, budget, cache, preflight
  scorers/       route, tools, arguments, facts, failure, calibration
  metrics/       Wilson, McNemar, bootstrap, reliability curve
  report/        Report model, markdown renderer, compare, gate, readme
  judge/         claim extraction, verification, caching, calibration
  evals/         suite specs, coverage gate, target adapters, importer
  integrations/  langchain, langgraph, provider usage capture  <- extras
```

## Why three result variants and not two

`Ok | Empty | Err`, discriminated on `status`.

A two-variant design (`Ok | Err`) forces "no matching records" into one of them.
Put it in `Ok` with an empty list and you have recreated the original bug. Put it
in `Err` and every genuinely empty query looks like a failure, which destroys the
false-alarm measurement.

The three-way distinction is load-bearing all the way through: the scorers
distinguish misreport from false alarm, and the judge distinguishes
`contradicted` from `not_in_evidence` for the same reason.

`truncated` is a **flag on `Ok`**, not a fourth variant. A truncated result is
still a successful result; making it a variant would force every caller to handle
a case that is not semantically different.

## Why falsiness is never inferred

```python
@tool()
async def count_matches() -> int:
    return 0  # Ok, not Empty
```

`0`, `False` and `""` are legitimate values. A falsiness check would turn "the
count is zero" into "there is no count", which is the same category error as the
original bug, one level down.

So `empty_when` is the only route from a plain value to `Empty`, and an
undeclared `None`/`[]`/`{}` raises `AmbiguousEmptyError` rather than being
guessed at.

## Why the trace is a JSON contract, not a Python object

`schemas/trace.v1.json` is generated from the pydantic models and committed.

An agent written in Node can emit traces that validate against it and be scored
by the same code as a Python agent. That is not hypothetical — the YojanaKhoj
bridge does exactly this, and the schema caught three real bugs in the exporter
before it ran once (missing required `usage`/`cost`, missing `env` fields, and a
trace status outside the enum).

Two properties make this work:

- **Unknown keys are preserved on read.** Another language can add a field
  without this reader dropping it.
- **A `schema_version` above this reader's is rejected outright.** Silently
  ignoring a newer version would mean scoring data you do not understand.

Attribute names follow OpenTelemetry GenAI conventions (`gen_ai.request.model`,
`gen_ai.usage.input_tokens`), so the exporter in 0.2 is a mapping rather than a
migration.

## Why the tracer records no prompt text

Usage, models, finish reasons, latency. Never a prompt, never a completion.

Redaction cannot help with something that should never have been captured, and a
tracer that logs prompts becomes a PII store that someone eventually commits to
git. Redaction runs on what *is* captured — tool arguments, structured output —
and it runs **before the sink**, never after.

## Why the runner refuses to start

Preflight, in order:

1. Dataset validates; test-split hash matches the recorded `suite_version`.
2. **Every tool marked `side_effect=True` has a stub bound.**
3. Judge's model family differs from the agent's.
4. Baseline, if gating, is readable and version-compatible.

The side-effect check comes before the judge check, and that ordering was a bug I
fixed: a run that could email a real recruiter must fail on *that*, not on a judge
misconfiguration. The check also does not reason about reachability — a routing
suite that can never enter a branch still needs the stub, because "probably
unreachable" is not a safety guarantee.

## Why repeats collapse before aggregation

Three repeats of 330 cases is a **330-case sample**, not 990.

Treating it as 990 would shrink every confidence interval by roughly √3 without
the sample having earned it. So collapse happens first, per case, with the rule
stated in the report:

| rule | used for | why |
| --- | --- | --- |
| majority | capability metrics | one flake is not a broken capability |
| any_hit | safety metrics | the worst observed behaviour is the honest summary |
| median | continuous metrics | resists one outlier repeat without inventing a verdict |

A case whose repeats disagree is counted as **unstable** and reported. High
instability makes the gate inconclusive rather than passing, because no
single-run delta is trustworthy at that point.

## Why the gate is a paired test

A naive threshold (`accuracy > 0.75`) is one line of YAML and fires on noise.
After a few red builds with no code change, someone disables it.

So the gate is:

- **Paired** on `case_id`. Comparing per-case outcomes rather than two
  proportions is what makes it sensitive enough to be useful at n=210.
- **Exact McNemar**, via a binomial tail rather than a chi-square approximation,
  because the discordant counts are small.
- **One-sided.** An improvement must never fail a build.
- **Floors as a backstop only**, and a floor on an unmeasured metric is a warning
  rather than a breach — otherwise deleting a scorer looks like passing.

## Why the judge is one scorer among many

Deterministic checks first: `contains`, `regex`, argument match modes,
pattern-based absence detection. The judge is used only where string logic cannot
decide, and it is treated as an instrument that must itself be calibrated.

Four properties:

- **`JudgeModel` is a Protocol** — one method, two strings in, one string out.
  Core imports no provider SDK, and `ScriptedJudge` lets a downstream author test
  a judge-backed scorer with no credentials.
- **The judge's model family must differ from the agent's.** Fails closed: an
  unrecognised model id with no declared family is refused, and a declared family
  disagreeing with the inferred one is refused rather than one being picked.
- **Malformed output is retried twice, then labelled `judge_error`.** A label
  outside the three is a parse failure, not a fourth opinion. `judge_error` is
  never cached, so a transient outage does not become permanent.
- **Claim and evidence are delimited data with escaped closing delimiters**, so
  the prompt instruction is the second line of defence rather than the only one.
  Four injection fixtures ship and are asserted not to flip a label.

A `judge_error` leaves an expectation **unmeasured, never missing** — an outage is
not evidence that the agent omitted a fact. For forbidden claims that rule is
safety-critical: a claim the judge could not decide is unmeasured, never clean.

## Why caching is content-addressed

SHA-256 over (claim, evidence, prompt version, model). The claim **id is
excluded**, so renaming a claim does not invalidate the cache, and two identical
claims share one call.

The prompt version is in the key because changing the prompt changes the
instrument, which invalidates the calibration too.

## Why coverage is a gate and not a warning

`check_coverage` refuses a suite that cannot support its own numbers: an empty
split, a branch below the per-branch floor, or a label naming a route the agent
cannot produce.

The last one is the subtle case. A typo like `job_serch` in a label scores zero
forever without ever looking like an error — the case simply always fails, and it
looks like the agent's fault.

This is the library's own headline rule applied to its own dataset: an unlabelled
suite is a refusal, not a 0%.

## What is deliberately not here

- **A hosted dashboard or observability backend.** Different product, and a
  dependency that would make the gate require a network.
- **An agent framework.** The target is `async (Case, Tracer) -> None`. The
  LangGraph adapter builds such a callable; it is not required.
- **`toolproof.langchain.wrap(base_tool)`**, deferred past 0.1.0. The `@tool`
  decorator and the callback handler already cover authoring a tool and tracing
  one the framework invokes, so `wrap` would be a third way to do the same thing
  and a public symbol to keep compatible.
- **An OpenTelemetry exporter**, scheduled for 0.2. The trace already follows OTel
  naming, so it is a mapping rather than a redesign.
