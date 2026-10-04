# Prompt to start a new chat

Copy everything inside the fence into the first message of a new session.

---

```
Read D:\Projects\neverempty\HANDOVER.md in full before doing anything. It is
the state of this project as of 2026-10-04 and it is accurate — every fact in
it was verified against the running code, not recalled.

Then read the specification: D:\Projects\neverempty Production Design Doc.md.
It is the spec. Where the doc and your instincts disagree, the doc wins. If you
think the doc is wrong, say so and stop — do not silently deviate.

Two repos are in play:
  D:\Projects\neverempty          the library, published to PyPI as 0.1.0
  D:\Projects\ai-career-copilot   NextRole, the agent being measured

Standing rules, which the handover repeats in full:
- A milestone is done when its row in the acceptance matrix passes, not when
  the code runs. Write the test from that row first, then the implementation.
- Core depends on pydantic v2 only. Stats are pure Python. CLI is argparse.
- mypy --strict clean, ruff clean, Python 3.10 to 3.13.
- Never invent a metric, threshold or claim the doc does not define. If a
  behaviour is undefined, ask me rather than choosing.
- Review before merge. Branch, never push straight to main/master.
- Commits end with:
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>

The non-negotiable semantics, which are the bugs this library exists to
prevent, are in section 2 of the handover. Read them before touching core.

Two things matter more than speed here:

1. Verify, do not assume. Reproduce every claimed defect against real code
   before accepting it. Prefer measurement to reasoning — several confident
   estimates in this project were overturned by measuring. If I ask "is it
   tested, verified, not assumed?", the honest answer must already be yes.

2. Never blur what is true with what is not yet true. Section 4 of the
   handover splits these explicitly. The headline metric
   (misreport_as_empty) has NEVER been measured on a production agent. The
   README says so. Do not let that slip in any doc, commit or post.

Do not spend my API budget without asking. The pending baseline run is about
$2.22 of real money on my key.

Start by telling me what you understand the current state to be and what you
think the next step is. Do not start work until I confirm.
```

---

## If you want to go straight to the main task instead

Replace the last paragraph with:

```
The next task is section 5A of the handover: measure misreport_as_empty on
NextRole. It is blocked on my review of the local-only branch
`neverempty-instrumentation` in ai-career-copilot.

Begin by showing me `git diff master` for that branch and walking me through
what it changes in chat.py, which is my production API request path. Flag
anything you would not deploy. Do not run the baseline until I approve both
the diff and the spend.
```

## If you want the LinkedIn post instead

```
Read D:\Projects\neverempty\HANDOVER.md, sections 1, 3 and 4 only.

I want a LinkedIn post announcing neverempty 0.1.0. Hook first — the reader
should picture this bug in their own product before they know I am
announcing anything.

Hard rules:
- Never claim misreport_as_empty is measured in production. It is not.
- Say "three frameworks I checked", never "most agents".
- Do not cite file line numbers for third-party repos; they were recorded in
  an earlier session and are not pinned.
- Leave the 91.2% routing number out. It is a routing number, not the honesty
  number, and in a feed it reads as the thing the library measures.

Stating the gap openly is the point, not a weakness to hide.
```
