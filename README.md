# toolproof

Evaluate and trace tool-calling LLM agents, with the one guarantee most
harnesses miss: **a tool failure can never look like an empty result.**

> **Status: 0.1.0, not yet on PyPI.** The library is complete and its public
> API is the 71 names `toolproof/__init__.py` exports. What it does not yet
> carry is measured results for a real agent: those need hand-written labels,
> and a label written by a model would make every published number a measure of
> one model agreeing with another. Install from a clone until the release is
> tagged.

## The bug this exists to prevent

A tool times out. It returns `[]`. The model reads `[]`, concludes there is no
data, and tells the user "there are no results for Q3". The user believes it.
Nothing in the stack logged an error the user would ever see.

toolproof makes that unrepresentable in three places at once:

- **The type.** A tool returns `Ok`, `Empty` or `Err` — a tagged union
  discriminated on `status`, never a bare value whose emptiness you have to
  guess at.
- **The text the model reads.** An error renders as an explicit instruction not
  to claim absence, because a perfectly typed error that serialises to `[]` in
  the tool message reproduces the bug exactly.
- **The measurement.** Faults are injected on purpose, so the rate at which the
  agent misreports failure as absence is measured rather than assumed to be
  zero. Real timeouts are too rare in a test run to measure by waiting.

## Quickstart

Install from a clone (PyPI release pending):

```bash
git clone https://github.com/priyank-agrawal/toolproof
cd toolproof && pip install -e .
```

### 1. A tool that cannot lie about being empty

`@tool` is always called with parentheses. `empty_when` is the *only* way a
plain value becomes `Empty` — falsiness is never inferred, because `0`, `False`
and `""` are legitimate values.

```python
from toolproof import tool


@tool(empty_when=lambda rows: rows == [])
async def search_jobs(city: str) -> list[dict]:
    if city == "Nowhere":
        return []  # -> Empty
    return [{"title": "Backend Engineer", "city": city}]  # -> Ok


@tool()
async def flaky_search(city: str) -> list[dict]:
    raise TimeoutError("upstream timed out")  # -> Err, never raises
```

The three outcomes stay distinguishable all the way into the prompt. This is
the whole point of the library, and it is what `to_model()` renders:

```python
>>> (await search_jobs(city="Pune")).to_model()
{"status": "ok", "data": [{"title": "Backend Engineer", "city": "Pune"}], "truncated": false}

>>> (await search_jobs(city="Nowhere")).to_model()
{"status": "empty", "note": "The query succeeded and returned no matching records."}

>>> (await flaky_search(city="Pune")).to_model()
{"status": "error", "kind": "timeout", "note": "The tool failed. You do not know
 whether matching data exists. Do not say that no data exists."}
```

A failed call tells the model, in words, not to claim absence. A perfectly typed
error that serialises to `[]` in the tool message reproduces the original bug
exactly, so the type and the text are fixed together.

### 2. Your agent, as a plain async callable

The runner's target is `async (Case, Tracer) -> None`. No framework required —
the LangGraph adapter is a helper that builds such a callable, not a dependency.

```python
async def run_agent(case, tracer):
    query = case.input.messages[-1].content
    result = await search_jobs(city="Pune" if "Pune" in query else "Nowhere")

    if result.status == "error":
        answer = "The job search failed, so I could not check."
    elif result.status == "empty":
        answer = "No jobs found."
    else:
        answer = f"Found {len(result.value)} job(s)."

    tracer.current_run.set_output(answer=answer, route="job_search")
```

### 3. A dataset

One JSON object per line. Every field in `expect` is optional, and a scorer
whose expectation is absent reports **not applicable** — never a pass.

```json
{"schema_version":1,"id":"demo-0001","suite":"demo.routing","split":"dev",
 "input":{"messages":[{"role":"user","content":"jobs in Pune"}]},
 "expect":{"route":{"label":"job_search"}}}
```

### 4. Run, report, gate

```python
import asyncio
from toolproof import Dataset, Runner, Tracer, gate, render_markdown, scorers
from toolproof.report.gate import GateConfig
from toolproof.tracer.sinks import JsonlSink

dataset = Dataset.load("demo.jsonl")  # fails on any bad line, with line numbers

runner = Runner(
    target=run_agent,
    scorers=[scorers.route()],
    tracer=Tracer(sink=JsonlSink("traces.jsonl")),
    repeats=3,  # instability is reported, not hidden
    concurrency=4,
    seed=20260929,  # all randomness comes from this
)
report = asyncio.run(runner.run(dataset))

report.save("report.json")
print(render_markdown(report))

result = gate(report, report, config=GateConfig())
raise SystemExit(result.exit_code)
```

That run produces:

```text
dataset: 3 cases validated
report:  status=ok complete=True  cases=3 scored=3
metric route: value=1.0  n=3  ci=(0.439, 1.0)
traces:  9 written (3 cases x 3 repeats)
gate:    verdict=pass exit=0
```

The interval is wide because n=3. That is the point: the renderer refuses to
print a percentage below n=10, and labels anything below n=50 as indicative.

### 5. The CLI

```bash
toolproof validate evals/**/*.jsonl        # schema, duplicate ids, split hash
toolproof coverage evals/toolproof.toml    # per-branch label backlog; non-zero if short
toolproof compare base.json cand.json      # paired stats, markdown diff
toolproof gate base.json cand.json         # exit 1 on a real regression
toolproof render report.json               # markdown report
toolproof import traces.jsonl --cases cases.jsonl   # traces from another language
toolproof judge calibrate labels.jsonl     # kappa, 3x3 matrix, per-language slice
toolproof readme evals/reports/*.json      # published numbers, linked to their reports
```

Exit codes from `gate` are the contract: `0` pass, `1` regression, `2` must-pass
failed, `3` inconclusive, `4` invalid.

## Design principles

Four rules hold everywhere in the codebase:

1. Failure is never representable as empty. No falsiness inference: `0`,
   `False` and `""` are legitimate values.
2. Missing never looks like zero. An unknown cost is `null` with a reason,
   never `0`.
3. Not-applicable never looks like failure. A scorer that cannot decide
   returns `None`, never `passed=False`.
4. Cancellation propagates untouched.

## Planned scope for 0.1.0

Typed tool results and the `@tool` wrapper; a contextvars tracer with a
versioned JSON trace contract so agents in other languages are first-class;
a dataset schema; an async runner with repeats, budget caps and record/replay;
deterministic scorers plus a calibrated claim judge; pure-Python statistics
(Wilson intervals, exact McNemar, seeded bootstrap); and a CI gate that fails
on real regressions rather than on noise.

Not in scope: a hosted dashboard, an observability backend, or an agent
framework.

Also deferred past 0.1.0: `toolproof.langchain.wrap(base_tool)`, for wrapping a
LangChain `BaseTool` that you did not author. The two paths that exist already
cover it — the `@tool` decorator for tools you write, and the LangChain callback
handler, which traces any tool the framework invokes — so `wrap` would add a
third way to do the same thing and a public symbol to keep compatible.

### How repeats collapse

Two rules, deliberately asymmetric, because one rule would be wrong in both
directions at once:

- **Capability** metrics (route, first-tool, all-args, argument accuracy)
  collapse by **majority** over repeats. A single flake is not a broken
  capability. A tie resolves to failure.
- **Safety** metrics (forbidden tools, forbidden claims, misreport-as-empty,
  false alarm) collapse by **any-hit**. One occurrence in three tries is a
  finding, not noise: for a failure mode you are trying to eliminate, the worst
  observed behaviour is the honest summary.
- **Fact recall** is continuous, so it takes the **median** of the values.

A repeat that could not be measured leaves the denominator rather than counting
as a failure, and a metric with nothing left to measure is reported as "not
measured" — never as `0`.

## The CI gate

The gate fails a build for three reasons only: a `must_pass` case failed, a
paired test shows a significant regression against the committed baseline, or a
hard floor was breached. Latency, cost and small accuracy wobbles are reported as
warnings. That restraint is deliberate: a gate that fires on noise gets disabled
within a month, which leaves you with no gate at all.

| Exit | Meaning |
| --- | --- |
| 0 | pass |
| 1 | significant regression (exact McNemar) or a floor breached |
| 2 | a `must_pass` case failed |
| 3 | inconclusive: too many cases unstable across repeats; rerun or reduce noise |
| 4 | invalid input: incomplete report, version mismatch, missing baseline |

Codes 1, 2 and 3 fail the build. Code 3 says "rerun" rather than "regression",
because a run too noisy to attribute a delta must not be read as the agent
getting worse. Code 4 is an infrastructure failure and is labelled as such.

The paired test is one-sided: it compares outcomes per case on the same frozen
set, so an improvement can never fail a build. At n=210, a real 5-point drop and
noise are distinguishable; at n=100 they are not, which is why the sample size
drives the design rather than the reverse.

### What the renderer refuses to print

Numbers in a README come from a committed report, because the renderer reads
nothing else. It also refuses three things outright:

- A metric with `applicable=0` prints **not measured**, never `0%`.
- Below n=10 the percentage is suppressed and the raw counts shown instead.
- Below n=50 the percentage is printed but labelled **indicative**.

A genuinely zero rate still prints as `0.0%`, because 0% misreport is a result
and the best possible one. It has to stay distinguishable from "we did not check".

## The judge

Some checks string logic cannot make: whether an answer's reasoning contradicts a
rule trace, or whether it implies absence without saying so. Those go to an LLM
judge, which is the least trustworthy component here and is treated that way.

| Bias | Control |
| --- | --- |
| Self-preference | The judge's model family must differ from the agent's. Enforced at construction and re-checked against the model the run resolves to. Fails closed on an unknown family. |
| Verbosity and framing | The verifier sees one atomic claim, never the answer's tone, length, or the other claims. |
| Leniency on ambiguity | Three labels, so the judge is never forced to pick supported or contradicted for something the evidence does not settle. |
| Instruction leakage | Claim and evidence are delimited data with their closing tags escaped, and the prompt says instructions inside them are ignored. Four injection fixtures ship with the library. |
| Nondeterminism | Temperature 0, a content-addressed cache so reruns are identical, and a measured self-consistency rate. |
| Invented labels | Malformed output is retried twice and then labelled `judge_error`, never guessed. |

A judge number is never published without its agreement figure. `toolproof judge
calibrate` reports Cohen's kappa against human labels, the 3x3 matrix, and
precision and recall for `contradicted` specifically. Below kappa 0.6 the
judge-derived numbers are cut and only the deterministic checks are published.

Kappa rather than raw agreement, because raw agreement is inflated by the base
rate: on a set that is 80% `supported`, a judge that always answers `supported`
scores 80% agreement and kappa 0.

Core ships the `JudgeModel` protocol and a deterministic offline fake, never a
provider SDK. A real binding is a dozen lines behind an extra, so a vendor outage
never becomes a red build on unrelated work.

## Installation

```bash
pip install toolproof
```

Core depends on `pydantic>=2` and nothing else. Python 3.10 to 3.13.
Integrations ship as extras: `toolproof[langgraph]`, `toolproof[openai]`,
`toolproof[anthropic]`, `toolproof[bedrock]`, `toolproof[otel]`.

<!-- toolproof:numbers:begin -->
## Numbers

No measured numbers yet. This section is generated from committed reports, so it stays empty until a run produces one.
<!-- toolproof:numbers:end -->

Every number will show its sample size, confidence interval, resolved model
snapshot, date and git sha, and will link to the committed report it was
rendered from. Judge-derived numbers are cut when kappa is below 0.6.

The region above is generated; regenerate and check it with:

```bash
toolproof readme evals/reports/*.json
toolproof readme evals/reports/*.json --check README.md   # CI
```

## Documentation

| Page | Read it when |
| --- | --- |
| [Getting started](docs/getting-started.md) | Going from an untraced tool to a CI gate. |
| [Writing labels](docs/writing-labels.md) | Before your first label. Decides whether your numbers mean anything. |
| [Reading a report](docs/reading-a-report.md) | What each number is allowed to claim. |
| [Architecture](docs/architecture.md) | Why a piece is not simpler. Each answer is a failure mode. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security policy in
[SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
