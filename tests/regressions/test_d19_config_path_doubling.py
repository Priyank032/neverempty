"""D19: a config's dataset paths were resolved relative to the config file,
doubling the directory that both the design doc and every real config repeat.

The doc's own example (line 870) is::

    [[suite]]
    path = "evals/nextrole/routing.jsonl"

and that file lives at ``evals/nextrole/routing.jsonl`` from the repo root
(line 577), with the config itself at ``evals/neverempty.toml``. Anchoring to
the config's directory turns that into ``evals/evals/nextrole/routing.jsonl``.

This repo's own config had the bug, and ``neverempty coverage`` crashed on it.
I had already met the doubling once, through ``init``, and "fixed" it by
writing the scaffold's paths relative to the config -- treating the symptom and
leaving every hand-written config broken, including the one in this repository
and the one in the design doc.

The resolution rule is now: try the config's directory, and fall back to the
directory the config's own tree is rooted in. A path that exists under either
is used; one that exists under neither is reported against the project root,
because that is what the doc tells people to write.

``Config.resolve``'s docstring promise still holds -- running the CLI from
another directory must not change which dataset a suite names -- because
neither candidate depends on the working directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neverempty.config import load_config

CONFIG = """\
[project]
name = "demo"

[target]
entrypoint = "evals.target:run_agent"

[[suite]]
name = "demo.routing"
path = "{path}"
split = "dev"
scorers = ["route"]
"""

CASE = {
    "schema_version": 1,
    "id": "demo-0001",
    "suite": "demo.routing",
    "split": "dev",
    "input": {"messages": [{"role": "user", "content": "q"}]},
    "expect": {"route": {"label": "job_search"}},
}


def _project(tmp_path: Path, *, declared: str, actual: str) -> Path:
    """A project whose config declares ``declared`` for a file at ``actual``."""
    config = tmp_path / "evals" / "neverempty.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(CONFIG.format(path=declared), encoding="utf-8")

    dataset = tmp_path / actual
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(json.dumps(CASE) + "\n", encoding="utf-8")
    return config


class TestRepoRootRelativePathsResolve:
    """The convention the design doc writes and every hand-written config uses."""

    def test_a_doc_style_path_finds_its_file(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path,
            declared="evals/nextrole/routing.jsonl",
            actual="evals/nextrole/routing.jsonl",
        )
        resolved = load_config(str(config)).resolve("evals/nextrole/routing.jsonl")
        assert resolved.is_file()

    def test_it_does_not_double_the_directory(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path,
            declared="evals/nextrole/routing.jsonl",
            actual="evals/nextrole/routing.jsonl",
        )
        resolved = load_config(str(config)).resolve("evals/nextrole/routing.jsonl")
        assert "evals/evals" not in resolved.as_posix()


class TestConfigRelativePathsStillResolve:
    """What ``init`` writes, and what the docstring originally promised."""

    def test_a_config_relative_path_finds_its_file(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path, declared="datasets/routing.jsonl", actual="evals/datasets/routing.jsonl"
        )
        assert load_config(str(config)).resolve("datasets/routing.jsonl").is_file()


class TestItDoesNotDependOnTheWorkingDirectory:
    """``Config.resolve``'s whole reason for existing."""

    @pytest.mark.parametrize("declared", ["evals/nextrole/routing.jsonl", "datasets/r.jsonl"])
    def test_the_same_path_resolves_from_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: str
    ) -> None:
        actual = (
            "evals/nextrole/routing.jsonl"
            if declared.startswith("evals/")
            else "evals/datasets/r.jsonl"
        )
        config = _project(tmp_path, declared=declared, actual=actual)

        here = load_config(str(config)).resolve(declared)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert load_config(str(config)).resolve(declared) == here


class TestAMissingFileIsStillReportedUsefully:
    """Neither candidate exists, so this is an error message, not a lookup.

    The config's own directory is reported rather than its parent: a config
    kept at the project root has no parent worth naming, and a path above the
    project would send a reader looking outside their own repository.
    """

    def test_the_reported_path_ends_with_what_was_declared(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path, declared="evals/nextrole/routing.jsonl", actual="evals/nextrole/routing.jsonl"
        )
        resolved = load_config(str(config)).resolve("nextrole/absent.jsonl")
        assert resolved.as_posix().endswith("nextrole/absent.jsonl")

    def test_it_stays_inside_the_project(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path, declared="evals/nextrole/routing.jsonl", actual="evals/nextrole/routing.jsonl"
        )
        resolved = load_config(str(config)).resolve("nextrole/absent.jsonl")
        assert tmp_path.resolve() in resolved.parents


class TestAnAbsolutePathIsUntouched:
    def test_it_is_returned_as_given(self, tmp_path: Path) -> None:
        config = _project(
            tmp_path, declared="evals/nextrole/routing.jsonl", actual="evals/nextrole/routing.jsonl"
        )
        absolute = (tmp_path / "evals" / "nextrole" / "routing.jsonl").resolve()
        assert load_config(str(config)).resolve(str(absolute)) == absolute


class TestThisRepositorysOwnConfigWorks:
    """It had the bug. `coverage` crashed on it, which is how it surfaced."""

    def test_every_declared_suite_path_exists(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = load_config(str(root / "evals" / "neverempty.toml"))
        for suite in config.suites:
            resolved = config.resolve(suite.path)
            assert "evals/evals" not in resolved.as_posix(), suite.path
