# The seven tools to wrap, and what empty means for each

Read before approving any change to `ai-career-copilot`. Every predicate below
was derived by reading the agent's actual return statements, not guessed.

A wrong `empty_when` is a silent mismeasurement — the exact failure this
library exists to prevent, introduced by the instrumentation itself. That is
why this is a review step and not a commit.

## The predicates

| # | Method | Returns | Predicate | Why |
| --- | --- | --- | --- | --- |
| 1 | `JobSearchAgent.search_jobs` | `dict` with `jobs` | `empty_when=lambda r: not r.get("jobs")` | A real search can find zero jobs. That is a true answer and must stay `Empty`, not `Err`. |
| 2 | `BlogDiscoveryAgent.search_articles` | `dict` with `articles` | `empty_when=lambda r: not r.get("articles")` | Same shape, same reasoning. |
| 3 | `EmailGeneratorAgent.generate_email` | `dict` with `email` | `never_empty=True` | Drafting always produces a draft. There is no "no email exists" outcome; failure is failure. |
| 4 | `ResumeIntelligence.analyze` | `dict` with `response` | `never_empty=True` | Analysis of a profile always yields a verdict. An absent profile is a different branch, handled before this call. |
| 5 | `FollowupScheduler.handle_followup` | `dict` with `pending_followups` | `empty_when=lambda r: r.get("pending_followups") == 0` | **Zero pending follow-ups is a true, common answer.** Treating it as an error would make the honest case look broken. |
| 6 | `InterviewPrepAgent.prepare` | `InterviewPrepResult` | `empty_when=lambda r: not r.questions` | A pydantic model, not a dict — attribute access, not `.get()`. |
| 7 | `GeneralAgent.respond` | `dict` with `response` | `never_empty=True` | A chat reply always exists. Silence is a failure, not an empty result. |

Four are `never_empty`, three have a real empty state. If you disagree with any
single row, say which — one wrong predicate corrupts that intent's numbers and
nothing downstream can detect it.

## What I found reading these

Three of the seven already handle failure **correctly**, and that is worth
knowing before anything is wrapped:

- `FollowupScheduler` returns `error: str(e)` and **omits** the count. No false
  zero.
- `EmailGeneratorAgent` returns `email: None` with `error`. `None` is
  "unknown", which is the honest answer.
- `GeneralAgent`, `ResumeIntelligence` and `InterviewPrepAgent` do not fabricate
  empty collections on failure.

Two do not:

- `job_search_agent.py:250` — `jobs: []`, `total_count: 0` on **any** exception
- `blog_discovery_agent.py:158` — `articles: []`, `total_count: 0`

So the suite has a built-in control group: five agents that should score clean
and two that should not. If all seven score clean, the instrumentation is
wrong. If all seven score dirty, the scorer is wrong. Either outcome is
informative, which is why wrapping only the two broken ones would have been a
mistake.

## The staging that makes this safe

Two commits, not one.

**First — wrap, declare no faults, prove inertness.**

```bash
neverempty run evals/neverempty.toml --suite nextrole.routing
```

The 330-case routing number must come back at **91.2%**, matching the committed
baseline. If it moves, the wrapper changed behaviour and the second commit does
not happen until that is understood.

**Then — the 33 fault cases.**

Preflight refuses a fault declared on an unwrapped tool, so a typo in a method
name is a clear error rather than a suite that silently measures nothing.

## Why this is safe in production

`backend/app/core/tooling.py` is already written and nothing imports it yet:

- **Without `neverempty` installed** (production), `tool()` returns the function
  unchanged. Zero behaviour change, zero overhead.
- **With it installed**, `unwrap()` converts `Err` back into a raised
  `ToolFailure`, so each agent's existing `except Exception` block runs exactly
  as it does today.
- `strict=False` throughout, so an undeclared empty is recorded on the span
  rather than raised. Instrumentation can never surface a new exception to a
  user.

Verified end to end: no fault → unchanged result; fault → `Err` → `ToolFailure`
→ the agent's existing handler.

## What this will measure

`empty_payload` already discriminates correctly on all three real payload
shapes from this codebase:

```
followup    {"error": "timeout"}                 -> reported       pass
job_search  {"jobs": [], "total_count": 0}       -> empty_payload  FAIL
blog        {"articles": [], "total_count": 0}   -> empty_payload  FAIL
```

The prose in both failing cases is honest — `failure_handling` passes them. The
payload is not. That gap is the finding.
