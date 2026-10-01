"""The files ``neverempty init`` writes into a new project.

Kept out of ``cli`` because they are content, not argument parsing, and because
every one of them is executed by a test: a scaffold that does not run is worse
than none, since it fails in the one place a newcomer has no way to debug.

What is deliberately absent is labels. The example cases are marked
``example-`` and exist to show the shape; ground truth about someone's own
agent is theirs to write, and a generated one would make every published number
a measure of one model agreeing with another.
"""

# ruff: noqa: E501 - JSONL is one case per line; wrapping breaks the format.

from __future__ import annotations

CONFIG = """# neverempty configuration. Run it with:
#
#     neverempty run evals/neverempty.toml

[project]
name = "my-agent"

[target]
# The async callable the runner invokes once per case: (Case, Tracer) -> None.
entrypoint = "evals.target:run_agent"
# Tools that touch the outside world. A run refuses to start unless each one
# has a stub bound, so an eval can never send a real email.
side_effect_tools = []

[[suite]]
name = "myagent.routing"
path = "datasets/routing.jsonl"
split = "dev"
scorers = ["route"]

[[suite]]
name = "myagent.failure"
path = "datasets/failure.jsonl"
split = "dev"
# failure_handling measures the headline number: when a tool fails, does the
# answer report it, or claim no data exists? The second is the bug this
# library exists to catch.
scorers = ["route", "failure_handling"]

[run]
repeats = 3          # instability is reported, not hidden
concurrency = 4
seed = 20260101      # the harness's randomness; not your agent's

[gate]
max_unstable_rate = 0.10
# Below roughly 200 cases the paired test alone detects almost nothing, so set
# a hard floor as well once you have real labels:
# floors = { route = 0.95 }
"""

ROUTING = """{"schema_version":1,"id":"example-0001","suite":"myagent.routing","split":"dev","input":{"messages":[{"role":"user","content":"show me backend jobs in Pune"}]},"expect":{"route":{"label":"job_search"}}}
{"schema_version":1,"id":"example-0002","suite":"myagent.routing","split":"dev","input":{"messages":[{"role":"user","content":"draft a follow-up email"}]},"expect":{"route":{"label":"email_draft"}}}
{"schema_version":1,"id":"example-0003","suite":"myagent.routing","split":"dev","input":{"messages":[{"role":"user","content":"what is the weather"}]},"expect":{"route":{"label":"general","acceptable":["job_search"]}}}
"""

FAILURE = """{"schema_version":1,"id":"example-0101","suite":"myagent.failure","split":"dev","input":{"messages":[{"role":"user","content":"any python jobs in Pune?"}]},"faults":[{"tool":"search_jobs","kind":"timeout","after_calls":0}],"expect":{"route":{"label":"job_search"}}}
{"schema_version":1,"id":"example-0102","suite":"myagent.failure","split":"dev","input":{"messages":[{"role":"user","content":"jobs in Mumbai?"}]},"faults":[{"tool":"search_jobs","kind":"upstream","after_calls":0}],"expect":{"route":{"label":"job_search"}}}
"""

TARGET = '''\
"""Your agent, as a plain async callable.

The runner calls this once per case. No framework is required -- the LangGraph
adapter is a helper that builds such a callable, not a dependency.
"""

from __future__ import annotations

from typing import Any

from neverempty import tool


@tool(empty_when=lambda rows: len(rows) == 0, timeout_s=8.0)
async def search_jobs(city: str) -> list[dict[str, Any]]:
    """Replace this with your real tool.

    ``empty_when`` is the point of the whole library: it declares what "no
    results" means for *this* tool, so an empty list can never be inferred
    from a failure. A tool that returns ``[]`` without declaring it raises,
    rather than letting the model read absence into an error.
    """
    if city.lower() == "pune":
        return [{"title": "Backend Engineer", "city": "Pune"}]
    return []


async def run_agent(case: Any, tracer: Any) -> None:
    """Route the request, call your tools, and record what you answered.

    ``set_output`` is what the scorers read. ``route`` is the branch you took;
    ``answer`` is what the user would see.
    """
    query = case.input.messages[-1].content.lower()

    if "job" in query:
        route = "job_search"
        result = await search_jobs(city="Pune" if "pune" in query else "Nowhere")
        # Three outcomes, three different answers. This is the part the
        # failure_handling scorer measures: an error must never be reported
        # to the user as an absence.
        if result.status == "error":
            answer = "The job search failed, so I could not check."
        elif result.status == "empty":
            answer = "No jobs found."
        else:
            answer = f"Found {len(result.value)} job(s)."
    elif "email" in query or "draft" in query:
        route, answer = "email_draft", "Here is a draft."
    else:
        route, answer = "general", "I can help with jobs and emails."

    tracer.current_run.set_output(answer=answer, route=route)
'''

FILES: dict[str, str] = {
    "evals/neverempty.toml": CONFIG,
    "evals/datasets/routing.jsonl": ROUTING,
    "evals/datasets/failure.jsonl": FAILURE,
    "evals/target.py": TARGET,
    "evals/__init__.py": "",
}

NEXT_STEPS = """\
Created a working eval setup.

  evals/neverempty.toml          the config
  evals/datasets/routing.jsonl   3 example routing cases
  evals/datasets/failure.jsonl   2 cases with an injected tool failure
  evals/target.py                your agent, as the runner calls it

Try it now, before changing anything:

  neverempty validate evals/datasets/*.jsonl
  neverempty run evals/neverempty.toml

Then make it yours:

  1. Point evals/target.py at your real agent and tools.
  2. Replace the example- cases with your own. Those four show the shape;
     they are not ground truth about your agent, and nobody else's can be.
  3. neverempty coverage evals/neverempty.toml tells you how many more each
     branch needs before a number from it is worth publishing.

docs/writing-labels.md has the full case format and every expectation type.
"""
