# Writing labels

This is the part of the work that decides whether your published number means
anything, and it is the part most likely to be skipped. The code cannot do it for
you, and it deliberately refuses to pretend otherwise: `neverempty coverage` exits
non-zero until the labels exist.

## The one rule that matters most

**Write the label before you run the agent on that case.**

Not before publishing. Not before the final run. Before the agent has ever
answered that query.

The reason is not procedural fussiness. The moment you have seen the agent's
answer, your label is partly a reaction to it. Every miss starts to look like a
label problem, you "fix" a few, and the published accuracy stops being a measure
of the agent and becomes a measure of how well the prompt fits these particular
queries. An interviewer who asks "did you tune on the test set?" ends the
conversation, and they are right to.

Three practices follow from it:

1. **Prompt iteration happens on the `dev` split only.** The `test` split is
   content-hashed; editing it requires bumping `suite_version`, which invalidates
   the baseline. A quiet edit is not possible.
2. **Run the test split once per prompt version**, not once per idea.
3. **At least 30% of queries should come from someone else.** Your own phrasing
   correlates with the prompt you wrote. Ask a few people what they would
   actually type.

## How many

The floor is 30 per branch, because that is what the intervals assume. Computed
with this library's own `wilson_interval`, for an agent measured at 86%:

| n | What it is | 95% interval | half-width |
| --- | --- | --- | --- |
| 30 | one branch | 70.3% – 94.7% | ±12.2 pts — indicative only |
| 100 | a small suite | 77.9% – 91.5% | ±6.8 pts |
| 210 | the doc's 7-branch split | 80.9% – 90.2% | ±4.7 pts |
| 330 | an 11-branch split | 81.9% – 89.4% | ±3.7 pts |

At n=100 a real 5-point regression and pure noise look identical. That is why the
renderer labels anything below n=50 as indicative and refuses to print a
percentage at all below n=10.

`neverempty coverage` tells you exactly what is missing:

```text
Suite nextrole.routing (test split): 0 case(s), 11 branch(es), minimum 30 per branch.

| branch | cases | short by |
| --- | --- | --- |
| job_search | 0 | 30 |
| email_draft | 0 | 30 |
...
```

## Ambiguous cases

Some queries genuinely have two defensible answers. Do not resolve that by
picking one and pretending.

```json
{"id":"nr-route-0087","suite":"nextrole.routing","split":"test",
 "input":{"messages":[{"role":"user","content":"It's been 10 days since I applied to the Razorpay SDE-2 role, can you write to the recruiter?"}]},
 "expect":{"route":{"label":"followup","acceptable":["email_draft"]}},
 "tags":["ambiguous"],
 "provenance":{"method":"human","labeller":"priyank","labelled_at":"2026-09-25",
               "note":"followup because an application already exists"}}
```

Strict accuracy uses `label`; lenient also accepts `acceptable`. **Both are
published.** Publishing only the lenient number flatters the agent; publishing
only the strict one penalises it for ambiguity the dataset itself admits.

The `note` field is worth using. In six months you will not remember why you
chose `followup`, and a label you cannot justify is one you will be tempted to
change.

## `must_pass`

Reserve it for failures that must never ship. A `must_pass` failure fails the
gate regardless of statistics — exit code 2, ahead of an ordinary regression.

Good uses: a fault case where claiming absence would mislead a user; a case where
a forbidden tool would send a real email.

Bad use: ordinary routing accuracy. If every case is `must_pass`, the gate is a
tripwire and someone will disable it.

## Fault cases need mirrors

A fault suite measures the **misreport-as-empty rate**: of the cases where a tool
failed, how many answers claimed that no data exists.

```json
{"id":"nr-fault-0001","faults":[{"tool":"search_jobs","kind":"timeout","after_calls":0}],
 "expect":{"forbidden_claims":[{"id":"absence","statement":"There are no matching jobs","match":"judge"}]},
 "must_pass":true}
```

That number alone is gameable. An agent that answers "something went wrong" to
every query scores a perfect 0% misreport rate and is useless. So every fault
suite needs **genuine-empty mirror cases** — a real query with no results and no
fault injected — which feed the false-alarm rate:

```json
{"id":"nr-empty-0001",
 "input":{"messages":[{"role":"user","content":"quantum computing roles in Nagpur under 2 LPA"}]},
 "expect":{"forbidden_claims":[{"id":"false-alarm","statement":"The job search failed","match":"judge"}]}}
```

The two numbers are published together. Either one alone can be gamed by
answering everything the same way; together they cannot.

## Provenance

Required on the `test` split — the `Case` model rejects a test case without it.

| `method` | Meaning |
| --- | --- |
| `human` | You wrote and labelled it. What a published number needs. |
| `llm_drafted_human_verified` | A model drafted it, you checked it. Acceptable, and recorded as such. |
| `generated_from_rules` | Ground truth from a rule engine, pinned to a source commit and file hash. |

`generated_from_rules` is how the YojanaKhoj consistency suite works: a rule
engine decides eligibility deterministically and that becomes ground truth. Note
what it does and does not license — **rules are ground truth for consistency, not
for real-world eligibility.** The scheme's rules may themselves be wrong; the
suite measures whether the model's prose agrees with them.

## Judge calibration labels

If any expectation uses `match="judge"`, the judge itself is an unmeasured
instrument until you calibrate it. That takes about 60 labelled claim/evidence
pairs, stratified across the three labels and both languages.

```bash
neverempty judge calibrate evals/calibration/judge.v1.jsonl
```

It prints Cohen's kappa, the 3×3 matrix, precision and recall for
`contradicted`, and a per-language slice.

**Kappa rather than raw agreement**, and the reason is worth internalising: on a
set that is 80% `supported`, a judge that answers `supported` to everything
scores 80% agreement and kappa 0.000. Raw agreement is inflated by the base rate;
kappa is not.

Below kappa 0.6, the judge-derived numbers are cut and only the deterministic
checks are published. `neverempty readme` enforces that — it will not print them.

The per-language slice earns its place immediately. A judge that is 90% overall
can be 100% in English and 67% in Hindi, and the aggregate hides it completely.

## Case format rules

Four rules the loader enforces that will reject your first dataset if you have
not met them. Each is checked at load time with the line number, so nothing
reaches a run half-valid.

| Field | Rule |
| --- | --- |
| `id` | `^[a-z0-9][a-z0-9._-]{2,63}$` — lowercase, 3 to 64 characters. `"c1"` is too short. |
| `suite` | Dotted lowercase, like `nextrole.routing`. A bare `"toy"` is refused. |
| `input` | Exactly one of `messages` or `payload`, never both and never neither. |
| `expect` | At least one expectation. A case that expects nothing can never fail. |

The `id` rule exists because case ids are the join key between a baseline and a
candidate report; an id that varies by case changes what the gate pairs. The
`suite` rule keeps suites addressable as a namespace once a project has more
than one. `expect` is refused empty for the reason that runs through the whole
library: a case with no expectation is not a passing case, it is an unmeasured
one, and counting it as a pass would inflate every number above it.

A minimal valid case:

```json
{"schema_version":1,"id":"nr-route-0001","suite":"nextrole.routing","split":"dev",
 "input":{"messages":[{"role":"user","content":"any backend python jobs?"}]},
 "expect":{"route":{"label":"job_search"}}}
```

The `test` split additionally requires `provenance` on every case, so a
published number can always be traced to where its ground truth came from.

## The `expect` reference

Every field is optional, and a scorer whose expectation is absent reports
**not applicable**, never a pass. Each example below is loaded by
`tests/unit/test_docs_examples.py`, so none of it can drift from the models.

### `route`

```json
{"route": {"label": "job_search", "acceptable": ["general"]}}
```

`acceptable` is the lenient set and must not contain `label`; the loader
refuses it if it does, because strict and lenient accuracy would then be the
same number.

### `tool_calls`

```json
{"tool_calls": {"mode": "first", "calls": [{"tool": "db_lookup"}]}}
```

`mode` is required and picks how strictly the call list is read:

| `mode` | Asserts |
| --- | --- |
| `first` | the first tool called is `calls[0]` |
| `set` | exactly these tools were called, order ignored |
| `sequence` | these tools, in this order |

`calls` needs at least one entry. Each entry is `{"tool": "<name>", "args":
{...}}`, and `args` is optional.

### `tool_calls[].args`

Each argument is checked under its own `match` mode:

```json
{"tool_calls": {"mode": "first", "calls": [
  {"tool": "search_jobs", "args": {"city": {"match": "exact", "value": "Pune"}}}]}}
```

| `match` | Compares |
| --- | --- |
| `exact` | equality, after JSON round-trip |
| `normalized` | case-folded and whitespace-stripped |
| `set` | membership-equal, order and duplicates ignored |
| `numeric` | numeric equality within `tol` |
| `regex` | `value` is a pattern, compiled at load time |
| `present` | the key was passed at all; `value` is ignored |
| `date` | parsed as a date, so `2026-10-01` equals `2026-10-01T00:00:00Z` |

`tol` is only meaningful with `numeric` and is refused elsewhere, so a
tolerance can never be silently ignored. A `regex` that does not compile fails
at load, not mid-run.

### `facts`

```json
{"facts": [
  {"id": "f1", "statement": "10 lakh", "match": "contains"},
  {"id": "f2", "statement": "^Rs ?[0-9]+$", "match": "regex"},
  {"id": "f3", "statement": "the scheme covers farmers", "match": "judge",
   "evidence_key": "scheme_text"}]}
```

`match` is `contains`, `regex` or `judge`. Only `judge` calls a model, and
`evidence_key` names the evidence it is allowed to read — a judged fact with
no evidence key is judged against nothing.

Note the two different claim types: a dataset `Fact` has
`(id, statement, match, evidence_key)`, while the judge's own `Claim` has
`(id, text)`. The judge receives `Claim`s built from your `Fact`s.

### `items`

Per-item ground truth for generated suites:

```json
{"items": [{"item_id": "PMKSY", "rule_result": null,
  "rule_trace": [{"criterion": "land_holding", "result": null,
                  "missing_field": "land_area"}]}]}
```

`rule_result` is genuinely three-way: `null` means the rule could not
evaluate, which is a different claim from `false` ("ineligible") and is the
case most likely to catch an agent overclaiming.

## A worked starting point

`evals/nextrole/routing.jsonl` and `evals/nextrole/failure.jsonl` ship with a few
worked examples covering the unambiguous case, the ambiguous case, an injected
fault, a Hinglish fault, and a genuine-empty mirror. Copy their shape.

## Implementing a JudgeModel

The judge ships a protocol and a deterministic offline fake, never a provider
SDK, so a vendor outage never turns into a red build on unrelated work. A real
binding is one method:

```python
class JudgeModel(Protocol):
    async def complete(self, *, system: str, user: str, temperature: float) -> str: ...
```

The judge owns the prompts, the parsing, the retries and the cache. A backend
only turns two strings into one string.

### What the string must contain

The judge's own prompt asks for this, and the parser requires it:

```json
{"label": "supported", "rationale": "<= 200 characters"}
```

A bare `"supported"` is **not** accepted — it parses as nothing and the claim
comes back `judge_error`. The parser is lenient about the wrapper and strict
about the content: a fenced block or a "Here is the JSON:" preamble is
recovered, an unrecognised label is not.

A working binding, with the Anthropic SDK as the example:

```python
class AnthropicJudge:
    def __init__(self, client, model: str = "claude-sonnet-5") -> None:
        self._client = client
        self._model = model

    async def complete(self, *, system: str, user: str, temperature: float) -> str:
        response = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=256,
        )
        return response.content[0].text
```

Wire it up with the agent's family declared, which is how the family check is
enforced:

```python
from neverempty import scorers
from neverempty.judge.judge import ClaimJudge

judge = ClaimJudge(
    model=AnthropicJudge(client),
    model_id="claude-sonnet-5",
    agent_family="openai",  # the family the *agent* runs on
    cache_dir=".neverempty-judge-cache",
)
scorer = scorers.facts(judge=judge)
```

`agent_family` is required. The judge's family must differ from the agent's and
the run refuses to start if it does not: a model grading its own output
measures agreement with itself, not correctness. The family is inferred from
`model_id` where the prefix is recognisable, and `judge_family=` declares it
when it is not — an unknown family is refused rather than assumed.

`cache_dir` is content-addressed on the claim, evidence, prompt version and
model, so reruns are cheap and stable.

### A judge on OpenRouter, or any OpenAI-compatible gateway

`JudgeModel` is one method, so a gateway binding is a few lines. OpenRouter,
Together, Groq, Fireworks and a local vLLM all speak the OpenAI chat API, so
the same class serves all of them -- only `base_url` and the model id change.

```python
from openai import AsyncOpenAI


class GatewayJudge:
    """Any OpenAI-compatible endpoint. Only base_url and model differ."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def complete(self, *, system: str, user: str, temperature: float) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=256,
        )
        return response.choices[0].message.content or ""
```

Wire it up with the agent's family declared:

```python
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
judge = ClaimJudge(
    model=GatewayJudge(client, "anthropic/claude-sonnet-4"),
    model_id="anthropic/claude-sonnet-4",  # the gateway's own id
    agent_family="openai",  # what the *agent* runs on
    cache_dir=".neverempty-judge-cache",
)
```

Pass the gateway's id verbatim. `infer_family` reads the vendor segment, so
`anthropic/claude-sonnet-4` resolves to `anthropic` and is checked against the
agent's `openai` as it should be.

**A gateway makes the family check matter more, not less.** Routing both the
agent and the judge through one endpoint makes it easy to run both on the same
underlying model while the two ids look different. If the vendor segment is one
this library does not know, `infer_family` returns `None` and the run refuses
to start rather than assume; declare `judge_family=` explicitly when you are
certain.

Two gateway-specific things worth knowing:

- **Silent routing.** OpenRouter can fall back to a different provider for the
  same model id. The judge records the id you gave, not the one that served the
  request, so a report cannot show that drift. Pin the provider in your
  OpenRouter settings if the number has to be reproducible.
- **`temperature=0` is advisory.** The judge asks for it, and not every
  provider behind a gateway honours it. `judge calibrate` measures the
  agreement you actually get, which is the number to trust over the setting.

### Three labels, four outcomes

A judgment is `supported`, `contradicted` or `not_in_evidence`. A `Verdict`
carries a fourth value, `judge_error`, which is not a judgment — it is the
harness saying the judge could not be read after its retries, and it carries
the reason so a degraded run can say what went wrong. It never counts as a
pass or a fail.

### Two different claim types

They are easy to confuse and do different jobs:

| Type | Shape | Where it comes from |
| --- | --- | --- |
| `Fact` | `(id, statement, match, evidence_key)` | your dataset's `expect.facts` |
| `Claim` | `(id, text)` | built from a `Fact` and handed to the judge |

Only a `Fact` with `match: "judge"` becomes a `Claim`. The rest are checked by
`contains` or `regex` and never reach a model.
