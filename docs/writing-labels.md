# Writing labels

This is the part of the work that decides whether your published number means
anything, and it is the part most likely to be skipped. The code cannot do it for
you, and it deliberately refuses to pretend otherwise: `toolproof coverage` exits
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

`toolproof coverage` tells you exactly what is missing:

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
toolproof judge calibrate evals/calibration/judge.v1.jsonl
```

It prints Cohen's kappa, the 3×3 matrix, precision and recall for
`contradicted`, and a per-language slice.

**Kappa rather than raw agreement**, and the reason is worth internalising: on a
set that is 80% `supported`, a judge that answers `supported` to everything
scores 80% agreement and kappa 0.000. Raw agreement is inflated by the base rate;
kappa is not.

Below kappa 0.6, the judge-derived numbers are cut and only the deterministic
checks are published. `toolproof readme` enforces that — it will not print them.

The per-language slice earns its place immediately. A judge that is 90% overall
can be 100% in English and 67% in Hindi, and the aggregate hides it completely.

## A worked starting point

`evals/nextrole/routing.jsonl` and `evals/nextrole/failure.jsonl` ship with a few
worked examples covering the unambiguous case, the ambiguous case, an injected
fault, a Hinglish fault, and a genuine-empty mirror. Copy their shape.
