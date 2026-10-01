# Prompt: draft the NextRole eval labels

Paste everything below the line into Claude Code, running in the
`ai-career-copilot` repo (the NextRole agent), with `neverempty` installed.

---

I need eval labels for my routing agent. You will draft them; I will verify
every one before they are used. Read this whole brief before writing anything.

## Context

`neverempty` is an eval harness for tool-calling agents. A label is one JSONL
line: an input, and the branch the agent should take for it. The harness runs
my real agent against each case and scores whether it routed correctly.

My agent (NextRole, a career copilot) routes to exactly these 11 intents:

    job_search        search for job listings
    email_draft       write an email on the user's behalf
    resume_query      questions about the user's own resume or profile
    blog_search       find articles or written guidance
    followup          follow up on a previous application or conversation
    profile_update    change something in the user's stored profile
    interview_prep    practice questions, interview advice
    salary_estimate   compensation ranges and negotiation
    skill_gap         what the user is missing for a target role
    career_planning   longer-horizon direction, switching fields
    general           anything else, including small talk and refusals

## What to produce

Three files. Write them in this order and stop after each for my review.

**1. `evals/nextrole/routing.jsonl` — 330 cases, 30 per intent**

One line per case:

```json
{"schema_version":1,"id":"nr-route-0001","suite":"nextrole.routing","split":"test",
 "input":{"messages":[{"role":"user","content":"any backend python roles in Pune?"}]},
 "expect":{"route":{"label":"job_search"}},
 "provenance":{"method":"llm_drafted_human_verified","labeller":"REPLACE_ME",
               "labelled_at":"2026-10-02","note":"drafted by Claude Code, verified by hand"}}
```

**2. `evals/nextrole/failure.jsonl` — 33 cases, 3 per intent**

Same shape, plus an injected tool failure. These are the most important cases
in the whole suite: they measure whether the agent, when a tool breaks, tells
the user the truth or claims no data exists.

```json
{"schema_version":1,"id":"nr-fault-0001","suite":"nextrole.failure","split":"test",
 "input":{"messages":[{"role":"user","content":"any python jobs in Pune?"}]},
 "faults":[{"tool":"search_jobs","kind":"timeout","after_calls":0}],
 "expect":{"route":{"label":"job_search"}},
 "provenance":{"method":"llm_drafted_human_verified","labeller":"REPLACE_ME",
               "labelled_at":"2026-10-02"}}
```

Fault kinds: `timeout`, `upstream`, `rate_limit`, `empty`, `truncated`. Spread
them; do not make all 33 timeouts. Use the real tool names from my codebase —
grep for `@tool` to find them. Do not invent tool names.

**3. `evals/nextrole/judge.jsonl` — 60 cases**

Only after I have confirmed the first two. Ask me before starting this one; its
shape depends on decisions I have not made yet.

## Format rules the loader enforces

These will reject the file if you get them wrong. Check them yourself with
`neverempty validate` before showing me anything.

| Field | Rule |
|---|---|
| `id` | `^[a-z0-9][a-z0-9._-]{2,63}$` — lowercase, 3–64 chars. Sequential: `nr-route-0001`. |
| `suite` | Dotted lowercase, exactly `nextrole.routing` or `nextrole.failure`. |
| `split` | `test` for all of these. |
| `input` | Exactly one of `messages` or `payload`. Never both, never neither. |
| `expect` | At least one expectation. An empty one is refused. |
| `provenance` | Required on the `test` split. |

`expect.route.acceptable` is a lenient set for genuinely ambiguous inputs, and
must not contain the `label`. Use it sparingly — see below.

## How to write good cases

**Realism beats coverage.** Write what a real user types: lowercase, typos,
missing punctuation, Hinglish, half-sentences. "pune me backend job", "resume
update kaise karu", "can u check my application status". A suite of clean
grammatical English measures a system nobody uses.

**Language mix.** Roughly 60% English, 25% Hinglish (Roman script), 15%
Devanagari. My users write all three, often in one sentence.

**Hard cases are the point.** For each intent, aim for:

- ~15 clear, unambiguous cases
- ~10 that are realistic but harder — vague wording, context from a previous
  turn, a request that sounds like one intent and is another
- ~5 near the boundary with a neighbouring intent

The pairs that actually confuse a router: `job_search` vs `blog_search` ("how
do I find backend jobs" is advice, not a search), `resume_query` vs
`profile_update` (reading vs writing), `skill_gap` vs `career_planning`
(specific gap vs direction), `followup` vs `email_draft` (status vs composing),
`salary_estimate` vs `career_planning`.

**Multi-turn where it matters.** Some cases should carry 2–3 messages, because
routing often depends on what came before. Put the earlier turns in `messages`
with the right roles.

**Ambiguity is a last resort, not a convenience.** If a case genuinely has two
defensible answers, use `acceptable`. If you are using it because you are
unsure, stop and ask me instead — that is a question about my product, and
guessing puts a wrong answer into the ground truth permanently.

## Rules for you

1. **Do not touch the agent's code.** Labels are a description of what it
   *should* do. Reading the code to find tool names and intent definitions is
   expected; changing anything so a case passes is not. If you think the agent
   routes something wrongly, write the label for the correct behaviour and tell
   me — that is a finding, not a problem with the label.

2. **Do not run the eval and then adjust labels to match.** Ground truth comes
   first. A label changed to agree with the agent measures nothing.

3. **Stop and ask when you are unsure what the right answer is.** You are
   drafting; I am deciding. A case you guessed at is worse than a case we
   discussed, because nothing downstream will ever flag it.

4. **Work in batches of 30 and show me each batch.** One intent at a time, in
   the order listed above. After each batch:
   - run `neverempty validate evals/nextrole/routing.jsonl`
   - show me the 30 lines
   - wait for me before continuing

   Do not write all 330 and then ask. I cannot verify 330 lines in one sitting,
   and unverified labels are the one thing that makes every number downstream
   meaningless.

5. **Keep a list of what you were unsure about** and give it to me at the end of
   each batch. Those are the cases I will look at hardest.

## Why this matters

Every number this harness publishes rests on these labels. If a label is wrong,
the agent gets blamed for being right, or credited for being wrong, and nothing
in the pipeline can detect it — the harness has no way to know the ground truth
is bad.

That is also why the provenance says `llm_drafted_human_verified` rather than
`human`. It is honest about how these were made, and it stops anyone later
reading the numbers as if a human had written every case from scratch.

Start with `job_search`. Show me 30 cases.
