# Reading a report

Below is real output from `toolproof render`, annotated with what each number is
allowed to claim. The design rule throughout: a number that cannot support a
claim does not get printed as one.

## A rendered report

```text
# nextrole.routing (test)

Run 2026-09-29T10:00:00.000Z · suite version 1 · status **ok**

330 case(s), 3 repeat(s) each, 330 scored, 0 unscored, 11 unstable across repeats.

## Metrics

| metric | value | 95% CI | notes |
| --- | --- | --- | --- |
| facts | not measured | — | not measured: no case in this run declared the expectation |
| failure_handling | 6.1% | 1.7% to 19.8% | wilson; indicative only at n=33; collapse=any_hit |
| route | 86.3% | 82.2% to 89.6% | wilson; collapse=majority |

## Cost and latency

Total $1.4200, mean $0.0043 per trace (12 trace(s) of unknown cost excluded)

p50 820ms, p95 2140ms over 990 trace(s) at concurrency 8. Concurrency is stated
because p95 under 8-way concurrency is not production p95.

## Judge

Model `anthropic.claude-3-5-sonnet-20241022-v2:0`, prompt version `verify.v1`,
agreement with human labels (Cohen's kappa) 0.780.

## Reproducibility

| field | value |
| --- | --- |
| target commit | 9f2c1a0000000000000000000000000000000000 |
| resolved model | gpt-4o-mini-2024-07-18 |
| toolproof | 0.1.0 |
| pricing table | openai-2026-09-01 |
| python | 3.12.10 |
| mode | live |
| seed | 20260929 |
| concurrency | 8 |
```

## Line by line

### `330 case(s), 3 repeat(s) each`

**330, not 990.** Repeats collapse to one observation per case *before*
aggregation. Counting three repeats as three cases would shrink every interval by
a factor the sample never earned — a 330-case suite would report the confidence
of a 990-case one.

### `11 unstable across repeats`

Cases whose outcome differed between repeats. This is reported rather than
smoothed away, because high instability means no single-run delta can be trusted.
The gate treats an unstable rate above its threshold as **inconclusive** (exit 3),
not as a pass.

### `facts | not measured`

No case in this run declared a `facts` expectation. It prints **not measured**,
never `0%`. A metric with `applicable=0` carrying a value is rejected by the
`Metric` model itself, so this cannot be faked by accident.

This matters more than it looks: if deleting a scorer printed `0%`, a floor check
on it would pass and nobody would notice the measurement had stopped.

### `failure_handling | 6.1% | 1.7% to 19.8% | indicative only at n=33`

A 6.1% misreport rate whose interval spans **1.7% to 19.8%**. The point estimate
is not the finding; the interval is. At n=33 this says "somewhere between rare and
one in five", which is honest and is why the note says indicative.

Two suppression rules apply:

- Below **n=10**, no percentage at all — the row prints a count like `2/4`.
- Below **n=50**, the percentage prints but is labelled indicative.

A genuine zero still prints as `0.0%`, because 0% misreport is a *result*, not an
absence. That distinction is the whole library in one line.

### `collapse=any_hit` vs `collapse=majority`

Deliberately asymmetric, and the report states which was used:

- **majority** for capability metrics like `route`. Two passes in three is a
  pass, because one flake is not a broken capability.
- **any_hit** for safety metrics like `failure_handling` and `forbidden_tools`.
  One occurrence in three is a finding. For a failure mode you are trying to
  eliminate, the worst observed behaviour is the honest summary.
- **median** for continuous metrics like `facts` and `calibration`.

One rule for everything would be wrong in both directions at once: it would hide
a rare unsafe behaviour, or call a flaky capability broken.

### `(12 trace(s) of unknown cost excluded)`

Twelve traces had a model with no pricing entry, or missing usage. Their cost is
`null` with a reason — **never `0`**. A zero would make those calls look free and
quietly deflate the total. The count is printed so the mean can be read for what
it is: a mean over 978 traces, not 990.

### `p95 2140ms ... at concurrency 8`

The concurrency is printed beside the latency because p95 under 8-way concurrency
is not production p95. A report that omitted it would invite exactly that
confusion.

### `Cohen's kappa 0.780`

Judge agreement with human labels. Above 0.6, so judge-derived numbers may be
published; below it, they are cut and only deterministic checks are published.

Kappa rather than raw agreement, because raw agreement is inflated by the base
rate: on a set that is 80% `supported`, a judge answering `supported` to
everything scores 80% agreement and kappa 0.000.

### The reproducibility block

Not decoration. `compare` refuses to compare two reports whose resolved models
differ unless you pass `--allow-model-change`, and refuses outright when either
report has `complete=false`. The seed is recorded because all randomness —
bootstrap resampling, any sampling — comes from one `random.Random(seed)`.

## Statuses

| status | Meaning |
| --- | --- |
| `ok` | every case scored |
| `degraded` | judge error rate above 2%. Still complete, still gates on deterministic metrics; judge-derived ones are excluded |
| `incomplete` | at least one case unscored. The gate treats this as a failure of the run, never as a smaller sample |
| `aborted_budget` | the cost cap was hit. Writes a valid incomplete report rather than a partial one marked complete |

`incomplete` beats `degraded` when both apply, because it is the stronger claim.

## Calibration section

When a case states a confidence, the report also carries a reliability curve:

```text
## Calibration

Expected calibration error (ECE): **0.093** over 10 buckets, n=210. The agent is
overconfident by 7.1% on average (stated 85.2%, accurate 78.1%).

| confidence | n | accuracy | stated | gap | note |
| --- | --- | --- | --- | --- | --- |
| 0.4-0.5 | 5 | 3/5 | 45.0% | 0.150 | too few to report a rate (n=5) |
| 0.5-0.6 | 10 | 70.0% | 55.0% | 0.150 | — |
| 0.8-0.9 | 45 | 73.3% | 85.0% | 0.117 | — |
| 0.9-1.0 | 110 | 87.3% | 95.0% | 0.077 | — |
```

Read it as: when this agent says 0.9, it is right 87% of the time. Close enough
to act on. The bucket count travels with the ECE because ECE is a function of the
bucketing — two curves bucketed differently are not comparable.

Empty buckets are omitted rather than printed as `0%`. A band nobody predicted
into has no accuracy at all.

## What the gate does with this

```bash
toolproof gate baseline.json candidate.json
```

- Pairs on `case_id` and requires matching `suite_version`.
- Uses an **exact one-sided McNemar test** on the discordant pairs. One-sided so
  an improvement never fails a build.
- Checks `must_pass` cases regardless of statistics (exit 2).
- Refuses an incomplete run (exit 4).
- Reports floors as warnings when the metric was not measured — deleting a scorer
  must not look like passing.

Exit codes are ordered by severity: `4 > 2 > 3 > 1 > 0`.
