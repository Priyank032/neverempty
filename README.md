# toolproof

Evaluate and trace tool-calling LLM agents, with the one guarantee most
harnesses miss: **a tool failure can never look like an empty result.**

> **Status: 0.0.1, name reservation only.** There is no public API yet. This
> repository is the skeleton described in M0 of the design doc; the first
> usable release is 0.1.0. Do not depend on it.

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

## Installation

```bash
pip install toolproof
```

Core depends on `pydantic>=2` and nothing else. Python 3.10 to 3.13.
Integrations ship as extras: `toolproof[langgraph]`, `toolproof[openai]`,
`toolproof[anthropic]`, `toolproof[bedrock]`, `toolproof[otel]`.

## Numbers

This README will carry measured results for two real agents. Every number will
show its sample size, confidence interval, resolved model snapshot, date and
git sha, and will link to the committed report JSON it was rendered from.

None of those numbers exist yet, so none are printed here.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security policy in
[SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
