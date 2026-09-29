# Node bridge (YojanaKhoj)

Two files to copy into the `govt-benefits-finder` repo, both under `scripts/`:

- `neverempty-export.js` — runs personas through `ruleEvaluator` and the rerank
  function in-process and writes Trace v1 JSONL.
- `neverempty-export.test.js` — exercises the pure parts against the real rule
  engine and scheme files. No LLM, no database, no network.

They live here rather than in that repo so neverempty ships them with tests it
runs itself; nothing in `govt-benefits-finder` is modified until you copy them.

## Running

```bash
node scripts/neverempty-export.js \
  --personas tests/personas.js \
  --schemes data/schemes \
  --out evals/yojanakhoj/ \
  --lang en,hi \
  --no-cache \
  --repeat 3

node scripts/neverempty-export.test.js
```

Then, on the Python side:

```bash
neverempty import evals/yojanakhoj/traces.jsonl \
  --cases evals/yojanakhoj/cases.jsonl
```

Exit codes: `0` ok, `2` rule engine error, `3` LLM error rate above 10%,
`4` validation failed.

## One change needed in `matchEngine.js`

The script calls `hardFilter` and `rerankWithLlm` directly and never
`matchSchemes`, because `matchSchemes` is what reads `MatchCache`. Those two are
currently module-private, so export them:

```js
module.exports = { matchSchemes, hardFilter, rerankWithLlm, /* ...existing */ };
```

The script fails with a clear message if they are missing, rather than silently
falling back to the cached path.

## Why `--no-cache` is mandatory

`buildCacheKey` hashes a tuple built from `ageBucket(age)` and
`incomeBucket(monthly_income)`. Two different personas in the same age and
income band therefore share a cache key, so a cached run can serve one persona an
explanation written for another. That would contaminate exactly the contradiction
rate being measured, so the flag is required rather than defaulted, and the
importer refuses any export that does not record the bypass.

## What is excluded, and why

`matchEngine` has two LLM-degraded fallback paths. When the rerank call throws,
it emits a fixed bilingual explanation, and sets `llm_confidence` to either
`Math.round(item.total_score * 100)` (a rescaled ranking score) or a hardcoded
`20`. Neither is a model answer or a stated confidence.

Those items are marked `llm_failed` and excluded from consistency scoring on both
sides. Scoring that text would measure the fallback string; scoring that number
as a confidence would measure the score function.

## The three-way rule result

`evaluateRule` returns `null` when a field is missing from the profile, and that
`null` is preserved all the way into `expect.items[].rule_result`. It is a real
third value: "the rules could not decide" is a different claim from "ineligible",
and collapsing them would destroy the most interesting case in the suite — the
one where the model is asked about something the rules could not settle.

A note on the design doc: it predicted that `eligibility_yes_no` might be a
strict boolean, which would force an overclaim on every null case. It is not —
`matchEngine` declares `enum: ['yes', 'no', 'check']` and its prompt says to use
`check` when unsure. So the schema *can* express uncertainty, and the measurable
question is whether the model actually uses it. That is what the
`overclaim_on_null` category counts.

## Trace v1 gotchas

Three things the schema enforces that are easy to get wrong, each of which was a
real bug in this script before its tests ran:

- `usage` and `cost` are required on every trace. Token counts stay `null` when
  the SDK did not report them, never `0`, which would price as free.
- `env` requires `neverempty_version`, `pricing_version` and `python_version`.
  The Node side prices nothing, so it names an unpriced table rather than
  inventing a version.
- Trace-level `status` is `ok` | `target_error` | `timeout` | `budget_abort`. A
  bare `"error"` is not a member; an LLM failure inside the target is a
  `target_error`. The *span* status is separately `error`, and that is what marks
  the LLM failure.
