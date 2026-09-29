# toolproof documentation

| Page | Read it when |
| --- | --- |
| [Getting started](getting-started.md) | You want to go from an untraced tool to a CI gate. Start here. |
| [Writing labels](writing-labels.md) | Before you write your first label. This is the part that decides whether your numbers mean anything. |
| [Reading a report](reading-a-report.md) | You have a report and want to know what each number is allowed to claim. |
| [Architecture](architecture.md) | You are wondering why a piece is not simpler. Each answer is a specific failure mode. |

## Reference

- **Public API** — exactly what `toolproof/__init__.py` exports via `__all__`.
  Anything reachable but unnamed there is private and may change in any release.
- **Trace schema** — [`schemas/trace.v1.json`](../schemas/trace.v1.json),
  generated from the pydantic models and committed. An agent in any language that
  validates against it is a first-class citizen.
- **Case schema** — [`schemas/case.v1.json`](../schemas/case.v1.json).
- **Eval suites** — [`evals/README.md`](../evals/README.md) for the label files
  and what still needs writing.
- **Node bridge** — [`integrations/node/README.md`](../integrations/node/README.md)
  for exporting traces from a JavaScript agent.

## The four rules

Everything in this codebase follows from these, and every page above is an
elaboration of one of them:

1. **Failure is never representable as empty.** No falsiness inference, strict
   mode on by default, and the model-facing renderer states failure explicitly.
2. **Missing never looks like zero.** Unknown cost is `null` with a reason.
3. **Not-applicable never looks like failure.** A scorer that cannot decide
   returns `None`, never `passed=False`.
4. **Cancellation propagates untouched.** `CancelledError`, `KeyboardInterrupt`
   and `SystemExit` are never swallowed, anywhere, including inside a predicate or
   a judge backend.
