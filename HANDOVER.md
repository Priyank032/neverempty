# Handover — neverempty, as of 2026-10-04

State of both repos, what is true, what is not yet true, and what to do next.
Written to be read cold by a session with no prior context.

---

## 1. What this project is

`neverempty` is a Python eval harness that measures how often a tool-calling
LLM agent tells a user "no results" when a tool actually failed.

**The bug it exists to prevent.** A tool times out and returns `[]`. The model
reads `[]`, concludes there is no data, and tells the user "there are no
results for Q3". The user believes it. Nothing in the stack logged an error the
user would ever see.

Two statements that sound alike and mean the opposite:

- "No results" means *there is nothing there.*
- "The search failed" means *I don't know what's there.*

**Why it happens.** A function returning `list` has no vocabulary for failure.
Its only word for the unhappy path is `[]` — the same word it uses for "I
succeeded and found nothing". The caller writes `if results:` and both paths
merge. Nobody writes this bug; the type signature makes it inevitable.

**The design doc at `D:\Projects\neverempty Production Design Doc.md` is the
specification.** Where the doc and instinct disagree, the doc wins. If the doc
seems wrong, say so and stop — do not silently deviate.

---

## 2. Standing working rules (still in force)

From the original brief, verbatim:

- Read the doc first, in full, before writing any code.
- A milestone is done when its row in the acceptance matrix passes, not when
  the code runs. Write the tests from that row first, then the implementation.
- Core depends on pydantic v2 only. Stats are pure Python (Wilson, exact
  binomial, bootstrap). CLI uses argparse. Everything else is an optional extra.
- `mypy --strict` clean, ruff clean, `py.typed` shipped, Python 3.10 to 3.13.
- Public surface is only what `__init__.py` exports via `__all__`.
- Never invent a metric, threshold or claim the doc does not define. If a
  behaviour is undefined, ask rather than choosing.

### Non-negotiable semantics — the bugs this library exists to prevent

- Failure must never be representable as empty: no falsiness inference, strict
  mode on by default, and the model-facing renderer must state failure
  explicitly.
- Missing must never look like zero: unknown cost is null with a reason,
  never 0.
- Not-applicable must never look like failure: a scorer that cannot decide
  returns `None` or raises, never `passed=False`.
- `CancelledError`, `KeyboardInterrupt` and `SystemExit` propagate untouched.

### Other standing constraints

- Tools tagged `side_effect=True` must have a stub bound or the runner refuses
  to start.
- The judge's model family must differ from the agent's.
- No prompt/completion text is ever recorded by any adapter.
- Redaction runs before the sink, never after.

### User's own rules

- Review before merging (hence branch-then-merge, never direct pushes to main).
- Commits end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## 3. Current state — verified 2026-10-04

### neverempty (`D:\Projects\neverempty`)

| Fact | Value |
|---|---|
| **Published** | PyPI 0.1.0, wheel + sdist, Sigstore-attested |
| Install | `pip install neverempty` — verified from a clean venv against the real index |
| Branch / tree | `main`, clean, 0 unpushed |
| Tag | `v0.1.0`, pushed |
| CI | success |
| Tests | **2158 passed** |
| ruff / mypy --strict | clean |
| Source | 13,617 lines, 71 public exports |
| Core deps | pydantic only (+ `tomli` on 3.10) |
| Python | 3.10–3.13 |
| License | Apache-2.0 |

Links:
- https://pypi.org/project/neverempty/
- https://github.com/Priyank032/neverempty

Last 3 commits:

```
01e5511 docs: 0.1.0 is on PyPI, and say who pays for model calls
82a811f fix(pre-commit): pre-commit-hooks has its own tags, not ruff's
484e663 fix(cli): a missing dataset is an error, not a traceback
```

### NextRole (`D:\Projects\ai-career-copilot`)

| Fact | Value |
|---|---|
| Branch | `neverempty-instrumentation` — **LOCAL ONLY, never pushed** |
| Base | `master` (not `main`) |
| Tree | clean |
| Commits ahead | 4 |
| Reviewed? | **No.** CLAUDE.md requires review before merge |
| Deployed? | **No, and must not be** until reviewed |

```
5d79e56 fix(eval): a zero-results search must not go silent
44071ea feat(eval): unwrap the last two call sites — wrap is now complete
6690009 wip(eval): wrap the seven tools — INCOMPLETE, do not deploy
3e12503 feat(eval): optional neverempty instrumentation shim
```

Note commit `6690009` says "do not deploy" — that state was fixed by `44071ea`.
The branch tip is complete; only the middle commit was broken.

---

## 4. What is TRUE and what is NOT YET TRUE

This distinction is the project's whole credibility. Do not blur it.

### True and verified

- The package installs from PyPI and runs: `init` → `run` scores both suites
  with **no API key**, pulling only pydantic.
- The harness makes **zero model calls of its own**. Core imports no model SDK
  (verified: no `openai`/`anthropic`/`boto3` in `core/`, `scorers/`,
  `metrics/`, `report/`).
- Routing accuracy on a live production router: **91.2%, CI [87.7%, 93.8%],
  n=330**, from 990 real gpt-4o calls, $2.22, 0 crashed, 3 unstable.
  Report committed at `evals/reports/nextrole.routing.json`.
- `misreport_as_empty` works: proven on shaped agents from a built wheel in a
  clean venv — honest agent 0.0, lying agent 1.0.
- The premise exists in real third-party code. Three public frameworks were
  read and found to return empty-on-failure. **Line numbers were recorded in
  session and are NOT pinned in the repo** — re-verify before citing publicly.

### NOT yet true — never claim these

- **`misreport_as_empty` has never been measured on a production agent.** The
  machinery is proven; the headline number is still ahead. The README says so.
- "Most agents do this" — never measured. Say "three frameworks I checked".
- "Failure can never look empty" — true only for `@tool`-wrapped tools, and
  only in the tool message. The model can still write "no results" in prose.
- The OpenRouter binding is documented but **unverified against a live
  gateway**; no key exists. The README marks this explicitly.
- Jev is deliberately unwired.

---

## 5. Open work, in priority order

### A. Measure `misreport_as_empty` on NextRole (the one that matters)

This is the project's central claim and the only thing standing between it and
a real argument.

**Blocked on:** human review of `neverempty-instrumentation`, then ~$2.22.

Sequence once reviewed:

1. Review `git diff master` in `ai-career-copilot`. It touches `chat.py`, the
   production API request path.
2. Reproduce the 330-case routing baseline. **Must come back 91.2%.** If it
   moves, instrumentation changed behaviour — stop and understand why before
   going further. Costs ~$2.22 (990 live gpt-4o calls, user's key, user's
   decision).
3. Write the 33 fault cases.
4. Measure.

**Prerequisite discovered and not yet handled:** NextRole lazy-imports agents
*inside* the request handlers (`chat.py:299`, `:776`). Importing `chat.py`
registers nothing, so preflight will refuse every fault case. **The eval target
must import the seven agent modules eagerly.** Verified: direct import
registers all seven.

**Plan change discovered late:** the 33 cases cannot all use `empty_payload`.
Two agents fail in ways it cannot see:

- `general_agent.py` ~L151-162 returns a cheerful greeting on exception
  ("Hello! I'm NextRole AI...") — prose masks the failure entirely, though the
  payload does carry `error`.
- `interview_prep_agent.py` ~L407-412 returns `self._fallback(...)`, a
  *populated* result, on both the JSON-parse and general exception paths.

Both need `forbidden_claims` expectations instead. Rough split: ~20
`empty_payload`, ~13 `forbidden_claims`. Better coverage — three misreport
patterns instead of one.

### B. Judge labels (~60) — not started

### C. Discoverability

Google had not indexed the package as of writing (published same day; normal
lag is 3 days to 3 weeks). Fixed during the session: repo homepage set to the
PyPI URL and 10 topics added. Remaining: LinkedIn post, PyPI badge in README,
one technical post (dev.to / Hashnode / r/Python), profile README link.

A LinkedIn post brief was prepared — see the session notes in section 7.

---

## 6. The seven NextRole predicates (reviewed and applied)

Re-verified against current code before wrapping. All still match.

| # | Method | Predicate |
|---|---|---|
| 1 | `JobSearchAgent.search_jobs` | `empty_when=lambda r: not r.get("jobs")` |
| 2 | `BlogDiscoveryAgent.search_articles` | `empty_when=lambda r: not r.get("articles")` |
| 3 | `EmailGeneratorAgent.generate_email` | `never_empty=True` |
| 4 | `ResumeIntelligence.analyze` | `never_empty=True` |
| 5 | `FollowupScheduler.handle_followup` | `empty_when=lambda r: r.get("pending_followups") == 0` |
| 6 | `InterviewPrepAgent.prepare` | `empty_when=lambda r: not r.questions` (pydantic — attribute access) |
| 7 | `GeneralAgent.respond` | `never_empty=True` |

**Built-in control group.** Two agents misreport
(`job_search_agent.py:250`, `blog_discovery_agent.py:158` — both `[]` +
`total_count: 0` + `error` on any exception); five handle failure correctly.
If all seven score clean the instrumentation is wrong; if all seven score dirty
the scorer is wrong. Either outcome is informative.

Full rationale: `docs/prompts/wrap-nextrole-tools.md`.

---

## 7. Two bugs found by testing the instrumentation

Both were in the shim, both would have shipped, neither was visible by reading.

**1. A zero-results search went silent.** `Empty` carries no value by design —
that is the library's core semantic — so the wrapper discarded the prose these
agents return next to their empty list:

```
before   status=empty, user sees ['']
after    status=empty, user sees ['I could not find any jobs matching that.']
```

Fixed by recording the payload in the predicate (the last place it is visible)
in a **ContextVar, not a global** — the endpoint serves requests concurrently
and a global would surface one user's results in another's reply. Verified with
three interleaved tasks: zero cross-talk.

**2. A fault-injected `Empty` returned the previous call's payload.** An
injected fault short-circuits the function, so the predicate never runs and the
stored payload is stale. The fault would have been **invisible** — a failure
reported as a successful empty search. That is this library's own headline bug,
reintroduced inside its own instrumentation, corrupting the exact number the
project exists to produce. Now raises `ToolFailure`, caught by each agent's
existing `except Exception`.

---

## 8. Gotchas worth knowing

- **NextRole's default branch is `master`, not `main`.**
- **NextRole lazy-imports agents inside handlers** — see 5A.
- **`core.autocrlf=true` + stale index.** After `pre-commit run --all-files`,
  `git status` may list ~138 files as modified while `git diff --numstat`
  reports zero. Content is byte-identical; it is stale stat cache. Clear with
  `git read-tree HEAD`.
- **The pre-commit hook was broken from the first commit until 2026-10-04.**
  `pre-commit-hooks` was pinned to ruff's tag (`v0.16.10`), which does not
  exist there. Now `v5.0.0`. Two separate tests passed while the hook was
  unusable — the tests checked the ruff floor, nothing checked the other repo.
- **TestPyPI cannot re-verify 0.1.0.** It already holds that version and no
  index allows re-upload. Expect `400 File already exists`; it is benign.
- **The `pypi` GitHub environment has a required reviewer.** Releases wait for
  a human click. That gate is deliberate — do not bypass it, even though `gh`
  is authenticated as the user.
- **Do not spend the user's API budget without asking.** The baseline run is
  ~$2.22 of real money on their key.

---

## 9. Working method that has held up

- Verify every reported defect against real code before accepting it.
- Write the acceptance test first and watch it fail *for the right reason*.
- Prefer measurement over reasoning. Several times a measurement overturned a
  confident estimate (per-call cost was 5× over; 8 KB payload cap would have
  truncated a normal 50-row search; 64 KB chosen from measured 382 B/row).
- Correct your own test premises when they, not the code, are wrong. This
  happened repeatedly and each time the test was the thing at fault.
