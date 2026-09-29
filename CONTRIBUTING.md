# Contributing

Thanks for looking. This library exists to make one class of bug impossible,
so the rules below are stricter than the average repo in exactly the places
where that bug hides.

## Setup

Development uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Priyank032/neverempty
cd neverempty
uv sync --group dev --all-extras
uv run pytest
```

Everything runs through `uv run`, so there is no virtualenv to activate.

## The checks CI runs

Run all four before pushing; they are the same commands CI uses.

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy                  # --strict, configured in pyproject.toml
uv run pytest --cov          # tests with coverage
```

`mypy --strict` must be clean. `# type: ignore` needs an error code and a
comment saying why. New public functions are fully annotated, and the package
ships `py.typed`, so a user's type checker sees our types.

## Non-negotiable semantics

These four are the reason the library exists. A pull request that weakens one
will be declined even if the tests pass.

1. **Failure is never representable as empty.** No falsiness inference: `0`,
   `False` and `""` are legitimate values. Strict mode stays on by default, and
   the model-facing renderer states failure explicitly in words.
2. **Missing never looks like zero.** An unknown cost is `null` with a reason
   attached, never `0`. A metric that could not be computed prints as "not
   measured", never as `0`.
3. **Not-applicable never looks like failure.** A scorer that cannot decide
   returns `None` or raises `ScorerError`. It never returns `passed=False`
   because data was missing.
4. **Cancellation propagates.** `asyncio.CancelledError`, `KeyboardInterrupt`
   and `SystemExit` are re-raised untouched, never converted into an error
   result. Turning a budget abort into an agent failure would corrupt every
   number downstream.

## Tests come first

A module is done when its row in the acceptance test matrix passes, not when
the code runs. Write the test from that row, watch it fail, then implement.

Coverage targets: 90% lines on `core`, `metrics`, `report` and `gate`; 80%
elsewhere. Every public function has at least one test.

Test layout:

| Directory | Holds |
| --- | --- |
| `tests/unit` | one module, no I/O, no network |
| `tests/property` | hypothesis property tests, mostly for the statistics |
| `tests/golden` | fixed input in, byte-identical JSON out |
| `tests/integration` | needs an optional extra; marked `@pytest.mark.integration` |

No test may make a network call. Provider interactions are tested against
fakes and recorded fixtures.

## Dependencies

Core depends on `pydantic>=2` and nothing else. The CLI uses `argparse`.
Statistics are pure Python: Wilson intervals, the exact binomial tail and a
seeded bootstrap. Everything else is an optional extra, and an optional import
must fail with a message naming the extra to install.

Adding a core dependency needs a strong argument in the pull request. "It is
already installed for most users" is not one.

## Schemas

`schemas/trace.v1.json` and `schemas/case.v1.json` are generated from the
pydantic models and committed. CI regenerates them and fails on any diff, so
regenerate and commit the result in the same pull request as a model change.

Schema versions are independent of the package version. A breaking change to a
schema is a new version file, not an edit to an existing one: readers must keep
accepting the previous version.

## Commits and pull requests

- [Conventional commits](https://www.conventionalcommits.org/): `feat:`,
  `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`.
- One logical change per pull request.
- Add a `CHANGELOG.md` entry under `## [Unreleased]` for anything a user would
  notice.
- Public API changes reference the milestone or issue they come from.

## Numbers in documentation

Any number in the README or the docs comes from a committed report file and
carries its sample size, confidence interval, resolved model id, date and git
sha. Tables are rendered from reports, never typed by hand. A number without
that provenance will be removed.

## Releasing

Maintainers only. Bump the version in `pyproject.toml`, move
`## [Unreleased]` to a dated version section in `CHANGELOG.md`, then tag
`vX.Y.Z`. The release workflow verifies that the tag, the project version and
the changelog section agree, then publishes over OIDC trusted publishing.

## Code of conduct

Be straightforward and assume good faith. Technical disagreement is welcome;
personal attacks are not. Report conduct problems through the contact in
`SECURITY.md`.
