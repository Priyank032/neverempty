# Getting started

This walks from an untraced tool to a CI gate that fails on a real regression.
Every command and output below was run against the version in this repository.

## Install

```bash
git clone https://github.com/Priyank032/neverempty
cd neverempty
pip install -e .
```

Core depends on **pydantic v2 only**. Everything else is an extra:

```bash
pip install -e ".[langgraph]"   # route_probe, the LangChain callback handler
pip install -e ".[openai]"      # usage capture from the OpenAI SDK
pip install -e ".[bedrock]"     # usage capture from Bedrock converse
pip install -e ".[all]"         # all of the above
```

The statistics are pure Python — no NumPy, no SciPy. A CI gate that needs a
scientific stack is a gate someone eventually disables.

## Step 1: make a tool honest

The bug this library exists to prevent is a tool that fails, returns `[]`, and
gets read as "no data exists". So the first thing to change is the tool.

```python
from neverempty import tool


@tool(empty_when=lambda rows: rows == [])
async def search_jobs(city: str) -> list[dict]:
    return await db.query(city)
```

Two things to notice:

- `@tool` takes parentheses, always. It is a factory, not a bare decorator.
- `empty_when` is the **only** way a plain return value becomes `Empty`.
  Falsiness is never inferred, because `0`, `False` and `""` are legitimate
  values that a falsiness check would silently turn into "nothing found".

If you return `None`, `[]` or `{}` without declaring `empty_when` or
`never_empty`, you get `AmbiguousEmptyError`. That is deliberate: the library
refuses to guess what your empty container meant.

```python
@tool()  # no empty_when declared
async def count_rows() -> int:
    return 0  # fine: 0 is a value, not an absence


@tool()
async def list_rows() -> list:
    return []  # AmbiguousEmptyError: declare what [] means
```

In production you can set `strict=False` to keep serving. The empty container
then passes through as `Ok` — so use it only when you have decided that an empty
value really is an ordinary result for that tool, and prefer declaring
`never_empty=True`, which says the same thing explicitly and keeps strict mode on
everywhere else.

## Step 2: read what the model reads

A perfectly typed `Err` that serialises to `[]` in the tool message reproduces
the original bug exactly. So the renderer is part of the contract:

```python
>>> (await search_jobs(city="Pune")).to_model()
{"status": "ok", "data": [...], "truncated": false}

>>> (await search_jobs(city="Nowhere")).to_model()
{"status": "empty", "note": "The query succeeded and returned no matching records."}

>>> (await search_jobs(city="Pune")).to_model()      # after a timeout
{"status": "error", "kind": "timeout", "note": "The tool failed. You do not know
 whether matching data exists. Do not say that no data exists."}
```

That last sentence is the one the whole library is built around. Whether the
model actually obeys it is measured in step 5, not assumed.

## Step 3: trace a run

The tracer uses `contextvars`, so a `@tool` call nested inside a graph node
inside a run gets the right parents with no plumbing:

```python
from neverempty import Tracer
from neverempty.tracer.sinks import JsonlSink

tracer = Tracer(sink=JsonlSink("traces.jsonl"))

async with tracer.run(case_id="demo-0001", tags={"suite": "demo"}) as run:
    rows = await search_jobs(city="Pune")
    run.set_output(answer=f"Found {len(rows.value)} jobs.", route="job_search")
```

Three rules hold here and are tested:

- **No prompt or completion text is ever recorded.** Usage, models, finish
  reasons and latency only. A tracer is not a prompt logger, and redaction
  cannot help with something that should never have been captured.
- **Redaction runs before the sink**, never after.
- **Unknown cost is `null` with a reason**, never `0`. A report says "cost
  unknown for 12/210 traces" rather than implying those calls were free.

## Step 4: write a dataset

One JSON object per line. Labels go in before you run the agent — see
[Writing labels](writing-labels.md) for why that ordering is not negotiable.

```json
{"schema_version":1,"id":"demo-0001","suite":"demo.routing","split":"dev",
 "input":{"messages":[{"role":"user","content":"jobs in Pune"}]},
 "expect":{"route":{"label":"job_search"}}}
```

```bash
neverempty validate demo.jsonl
```

Validation names every bad line with its line number, rejects duplicate ids, and
prints the test-split hash. Editing a hashed test split requires bumping
`suite_version`, which invalidates the baseline — so a quiet edit cannot happen.

## Step 5: run and score

```python
import asyncio
from neverempty import Dataset, Runner, Tracer, scorers
from neverempty.tracer.sinks import JsonlSink


async def run_agent(case, tracer):
    """The target contract: async (Case, Tracer) -> None."""
    query = case.input.messages[-1].content
    rows = await search_jobs(city=extract_city(query))
    tracer.current_run.set_output(answer=summarise(rows), route="job_search")


dataset = Dataset.load("demo.jsonl")

runner = Runner(
    target=run_agent,
    scorers=[scorers.route(), scorers.forbidden_tools()],
    tracer=Tracer(sink=JsonlSink("traces.jsonl")),
    repeats=3,
    concurrency=8,
    max_cost_usd=5.00,
    seed=20260929,
    stubs={"send_gmail": lambda **kw: "stubbed"},
)
report = asyncio.run(runner.run(dataset))
```

Or, with a config file, the same run is one command:

```bash
neverempty run evals/neverempty.toml --out evals/reports/candidate.json
```

That reads `[target].entrypoint`, builds the scorers each suite names, and writes
one report per suite. It exits non-zero when a run is incomplete, because an
incomplete run is a failure of the run rather than a smaller sample.

`stubs` is not optional for a tool marked `side_effect=True`. The runner refuses
to start without one, and it does not reason about whether that tool is
reachable — a run that *could* email a real recruiter fails at preflight.

## Step 6: measure failure handling on purpose

Real timeouts almost never happen in a test run, so the misreport rate would be
0 out of 0 opportunities. Inject them:

```json
{"schema_version":1,"id":"demo-fault-0001","suite":"demo.failure","split":"dev",
 "input":{"messages":[{"role":"user","content":"jobs in Pune"}]},
 "faults":[{"tool":"search_jobs","kind":"timeout","after_calls":0}],
 "expect":{"forbidden_claims":[{"id":"absence","statement":"There are no jobs","match":"judge"}]},
 "must_pass":true}
```

Kinds: `timeout`, `upstream`, `rate_limit`, `empty`, `truncated`. The fault is
applied by the `@tool` wrapper through a context variable, so your agent needs no
changes and the real dependency is never called.

**Every fault suite needs genuine-empty mirror cases** — a real query with no
results and no fault injected. Without them, an agent that answers "something
went wrong" to everything scores a perfect misreport rate.

## Step 7: gate in CI

```bash
neverempty gate baseline.json candidate.json
```

| Exit | Meaning |
| --- | --- |
| 0 | pass |
| 1 | regression (paired McNemar, one-sided) |
| 2 | a `must_pass` case failed |
| 3 | inconclusive (too unstable, or too few paired cases) |
| 4 | invalid (incomplete run, mismatched `suite_version`, unreadable baseline) |

The test is **paired and one-sided**. Paired because comparing outcomes per case
rather than two proportions is what makes it sensitive at n=210. One-sided
because an improvement must never fail a build.

A floor on an unmeasured metric is a warning, never a breach — otherwise
deleting a scorer would look like passing.

## Next

- [Writing labels](writing-labels.md) — the part that decides whether your
  numbers mean anything.
- [Reading a report](reading-a-report.md) — what each number is allowed to claim.
- [Architecture](architecture.md) — why the pieces are shaped this way.

### Capping payload size

There is **no default size cap**. `truncated` stays `false` until a
`truncated_when` predicate says otherwise, so a runaway tool can return a
megabyte and it reaches the model whole. Declare the cap you want:

```python
import json

CAP_BYTES = 8_000


@tool(
    never_empty=True,
    truncated_when=lambda rows: len(json.dumps(rows, default=str)) > CAP_BYTES,
)
async def search_jobs(city: str) -> list[dict]: ...
```

The predicate flags the result; it does not shrink it, because dropping rows
silently is the sibling bug — the model reads a short list as the whole list.
Slice inside the tool and let `truncated_when` label what you did:

```python
@tool(never_empty=True, truncated_when=lambda rows: len(rows) >= 100)
async def search_jobs(city: str) -> list[dict]:
    return (await db.query(city))[:100]
```

`to_model()` then adds an explicit note telling the model the result is
incomplete and not to state a total.

### Calling a blocking tool from async code

Use `asyncio.to_thread`, not `loop.run_in_executor`:

```python
await asyncio.to_thread(blocking_tool)  # span recorded
await loop.run_in_executor(None, blocking_tool)  # span lost
```

`to_thread` copies the caller's context into the worker thread;
`run_in_executor` does not, so the tool cannot see the open run and its span
is never recorded. The tool still runs and still returns the right result,
which is what makes it dangerous: tool-selection scoring treats the tool as
never called, and a missing measurement looks exactly like a correct one.

The wrapper warns once per tool when it finds no run while one is open, so the
loss is visible rather than silent. It cannot recover the span: a thread with
no context has no parent to attach to, and inventing one would file the tool
under the wrong node.

If you must use an executor, pass the context explicitly:

```python
import contextvars, functools

context = contextvars.copy_context()
await loop.run_in_executor(None, functools.partial(context.run, blocking_tool))
```
