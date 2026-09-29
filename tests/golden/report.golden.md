# golden.suite (dev)

Run 2026-09-23T10:00:00.000Z · suite version 1 · status **ok**

12 case(s), 1 repeat(s) each, 12 scored, 0 unscored, 0 unstable across repeats.

## Metrics

| metric | value | 95% CI | notes |
| --- | --- | --- | --- |
| arguments | not measured | — | not measured: no case in this run declared the expectation |
| facts | 0/1 | — | too few to report a rate (n=1) |
| forbidden_tools | 1/1 | — | too few to report a rate (n=1) |
| route | 75.0% | 46.8% to 91.1% | wilson; indicative only at n=12; collapse=majority |

## Confusion matrix

Rows are labels, columns are predictions. Rows with fewer than 10 cases show counts only.

| label | blog_search | clarify | email_draft | followup | general | job_search | resume_query | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blog_search | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| clarify | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| email_draft | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| followup | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 3 |
| general | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| job_search | 0 | 1 | 0 | 0 | 1 | 2 | 0 | 4 |
| resume_query | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |

## Cost and latency

Cost unknown for 12 trace(s); no total is reported, because an unknown cost is never counted as zero.

p50 0ms, p95 0ms over 12 trace(s) at concurrency 4. Concurrency is stated because p95 under 4-way concurrency is not production p95.

## Reproducibility

| field | value |
| --- | --- |
| target commit | abc1234 |
| resolved model | not recorded |
| neverempty | 0.0.1-golden |
| pricing table | empty-2026-09-23 |
| python | 3.12.0 |
| mode | live |
| seed | 20260921 |
| concurrency | 4 |
