"""The TOML config loader.

The config is the file a target repo commits, so its validation is the first
thing a new adopter meets. Two rules:

- An unknown key is an error, not a warning. A misspelled `max_cost_usd` that
  loaded silently would remove the budget cap from a live run.
- The doc's own TOML must load unchanged. A config someone copies out of the
  design doc and cannot run is a broken spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from toolproof.config import Config, ConfigError, load_config

DOC_CONFIG = """
[project]
name = "nextrole"
toolproof_version = ">=0.1,<0.2"

[target]
entrypoint = "evals.targets:run_nextrole"
git_sha_from = "git"
prompt_modules = ["app.agents.intent_router:INTENT_CLASSIFICATION_PROMPT"]
side_effect_tools = ["send_gmail", "save_application"]
stubs = "evals.targets:STUBS"

[[suite]]
name = "nextrole.routing"
path = "evals/nextrole/routing.jsonl"
split = "test"
suite_version = 1
scorers = ["route", "tool_selection", "forbidden_claims"]

[[suite]]
name = "nextrole.failure"
path = "evals/nextrole/failure.jsonl"
split = "test"
suite_version = 1
scorers = ["route", "failure_handling", "forbidden_claims"]

[run]
repeats = 3
concurrency = 8
case_timeout_s = 120
max_cost_usd = 5.0
seed = 20260921
cache = "evals/.cache"
mode = "live"

[judge]
provider = "bedrock"
model = "anthropic.claude-3-5-sonnet-20241022-v2:0"
agent_family = "openai"
calibration = "evals/calibration/judge.v1.jsonl"

[gate]
baseline = "evals/baselines/nextrole-routing.json"
must_pass = "fail_on_any"
paired_alpha = 0.05
floors = { route_strict = 0.75, misreport_as_empty_max = 0.10 }
warn = { p95_latency_increase = 0.25, cost_increase = 0.20 }
max_unstable_rate = 0.10
"""


def write(tmp_path: Path, text: str, name: str = "toolproof.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTheDocsConfig:
    def test_the_docs_config_loads_unchanged(self, tmp_path: Path) -> None:
        """Copied verbatim out of the design doc. If this fails, the spec and
        the implementation disagree about the file a user is told to write."""
        config = load_config(write(tmp_path, DOC_CONFIG))
        assert config.project.name == "nextrole"
        assert len(config.suites) == 2

    def test_both_floor_directions_are_read(self, tmp_path: Path) -> None:
        """``route_strict`` is a minimum and ``misreport_as_empty_max`` a
        maximum, distinguished by the suffix the doc's TOML uses."""
        config = load_config(write(tmp_path, DOC_CONFIG))
        assert config.gate.floors["route_strict"] == 0.75
        assert config.gate.floors["misreport_as_empty_max"] == 0.10

    def test_the_run_block_is_read(self, tmp_path: Path) -> None:
        run = load_config(write(tmp_path, DOC_CONFIG)).run
        assert run.repeats == 3
        assert run.concurrency == 8
        assert run.max_cost_usd == 5.0
        assert run.seed == 20260921
        assert run.mode == "live"

    def test_the_side_effect_tools_are_read(self, tmp_path: Path) -> None:
        """The runner's preflight needs these before any case runs."""
        target = load_config(write(tmp_path, DOC_CONFIG)).target
        assert target.side_effect_tools == ["send_gmail", "save_application"]

    def test_the_suites_keep_their_scorer_lists(self, tmp_path: Path) -> None:
        config = load_config(write(tmp_path, DOC_CONFIG))
        assert config.suites[1].scorers == ["route", "failure_handling", "forbidden_claims"]

    def test_the_warn_thresholds_are_read(self, tmp_path: Path) -> None:
        gate = load_config(write(tmp_path, DOC_CONFIG)).gate
        assert gate.warn["p95_latency_increase"] == 0.25
        assert gate.warn["cost_increase"] == 0.20


class TestStrictness:
    def test_an_unknown_top_level_key_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path, DOC_CONFIG + "\n[unknown]\nkey = 1\n")
        with pytest.raises(ConfigError, match="unknown"):
            load_config(path)

    def test_a_misspelled_run_key_is_rejected(self, tmp_path: Path) -> None:
        """The case that matters: a silently ignored ``max_cost_usd`` typo would
        remove the budget cap from a live run."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n'
            "[run]\nmax_cost_used = 5.0\n",
        )
        with pytest.raises(ConfigError, match="max_cost_used"):
            load_config(path)

    def test_a_missing_file_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path: Path) -> None:
        path = write(tmp_path, "[project\nname = broken")
        with pytest.raises(ConfigError) as exc:
            load_config(path)
        assert "toolproof.toml" in str(exc.value)

    def test_a_config_with_no_suite_is_rejected(self, tmp_path: Path) -> None:
        """A config that declares nothing to run measures nothing."""
        path = write(tmp_path, '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n')
        with pytest.raises(ConfigError, match="suite"):
            load_config(path)

    def test_duplicate_suite_names_are_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n'
            '[[suite]]\nname = "a.b"\npath = "y.jsonl"\nsplit = "test"\n',
        )
        with pytest.raises(ConfigError, match="duplicate"):
            load_config(path)

    def test_an_entrypoint_without_a_colon_is_rejected(self, tmp_path: Path) -> None:
        """``module:attribute`` is the documented form; without the colon there
        is nothing to import and the failure would come mid-run."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "evals.targets"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        with pytest.raises(ConfigError, match="entrypoint"):
            load_config(path)

    def test_an_unknown_scorer_name_is_rejected(self, tmp_path: Path) -> None:
        """A typo in a scorer name is an expectation nobody measures, and an
        unmeasured expectation cannot fail."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n'
            'scorers = ["rout"]\n',
        )
        with pytest.raises(ConfigError, match="rout"):
            load_config(path)

    def test_the_error_lists_the_known_scorers(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n'
            'scorers = ["rout"]\n',
        )
        with pytest.raises(ConfigError, match="route"):
            load_config(path)


class TestDefaults:
    def test_a_minimal_config_loads_with_documented_defaults(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        config = load_config(path)
        assert config.run.repeats == 1
        assert config.run.concurrency == 8
        assert config.run.case_timeout_s == 120
        assert config.run.mode == "live"
        assert config.gate.paired_alpha == 0.05
        assert config.gate.max_unstable_rate == 0.10

    def test_the_default_repeats_is_one_not_three(self, tmp_path: Path) -> None:
        """The doc's example sets 3, but a default of 3 would triple the cost of
        a first run without the user asking for it."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        assert load_config(path).run.repeats == 1

    def test_no_budget_is_set_by_default(self, tmp_path: Path) -> None:
        """``None`` means uncapped, which is different from a cap of 0. A default
        cap would abort a legitimate first run halfway through."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        assert load_config(path).run.max_cost_usd is None

    def test_a_suite_with_no_scorers_gets_none_rather_than_a_guess(self, tmp_path: Path) -> None:
        """Choosing scorers for the user would measure something they did not
        ask for and publish it under their suite name."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        assert load_config(path).suites[0].scorers == []


class TestGateConfigBridge:
    def test_the_gate_block_converts_to_a_gate_config(self, tmp_path: Path) -> None:
        """One definition of the thresholds, so the file and the gate cannot
        drift apart."""
        config = load_config(write(tmp_path, DOC_CONFIG))
        gate_config = config.gate.to_gate_config(seed=config.run.seed)
        assert gate_config.paired_alpha == 0.05
        assert gate_config.max_unstable_rate == 0.10
        assert gate_config.floors["route_strict"] == 0.75
        assert gate_config.seed == 20260921


class TestHashing:
    def test_the_config_hash_is_stable(self, tmp_path: Path) -> None:
        """``config_hash`` goes in the report, so two runs of one config have to
        produce the same hash or every report looks like a different setup."""
        first = load_config(write(tmp_path, DOC_CONFIG))
        second = load_config(write(tmp_path, DOC_CONFIG, name="copy.toml"))
        assert first.config_hash() == second.config_hash()

    def test_reformatting_does_not_change_the_hash(self, tmp_path: Path) -> None:
        """The hash is over the parsed values, not the file bytes, so adding a
        comment does not read as changing the configuration."""
        base = load_config(write(tmp_path, DOC_CONFIG))
        commented = load_config(
            write(tmp_path, "# a comment\n" + DOC_CONFIG, name="commented.toml")
        )
        assert base.config_hash() == commented.config_hash()

    def test_changing_a_value_changes_the_hash(self, tmp_path: Path) -> None:
        base = load_config(write(tmp_path, DOC_CONFIG))
        changed = load_config(
            write(tmp_path, DOC_CONFIG.replace("repeats = 3", "repeats = 5"), name="c.toml")
        )
        assert base.config_hash() != changed.config_hash()

    def test_the_hash_is_a_sha256_hex_digest(self, tmp_path: Path) -> None:
        digest = load_config(write(tmp_path, DOC_CONFIG)).config_hash()
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


class TestPathResolution:
    def test_suite_paths_resolve_relative_to_the_config_file(self, tmp_path: Path) -> None:
        """Running the CLI from a different directory must not change which
        dataset a suite names."""
        nested = tmp_path / "evals"
        nested.mkdir()
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "evals/routing.jsonl"\nsplit = "test"\n',
        )
        config = load_config(path)
        assert config.resolve(config.suites[0].path) == tmp_path / "evals" / "routing.jsonl"

    def test_an_absolute_path_is_left_alone(self, tmp_path: Path) -> None:
        absolute = (tmp_path / "abs.jsonl").as_posix()
        path = write(
            tmp_path,
            f'[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            f'[[suite]]\nname = "a.b"\npath = "{absolute}"\nsplit = "test"\n',
        )
        config = load_config(path)
        assert config.resolve(config.suites[0].path) == Path(absolute)

    def test_the_config_records_its_own_location(self, tmp_path: Path) -> None:
        path = write(tmp_path, DOC_CONFIG)
        assert load_config(path).source == path


class TestVersionConstraint:
    def test_a_satisfied_version_constraint_loads(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            '[project]\nname = "x"\ntoolproof_version = ">=0.0,<1"\n'
            '[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        assert load_config(path).project.toolproof_version == ">=0.0,<1"

    def test_the_constraint_is_recorded_rather_than_enforced_silently(self, tmp_path: Path) -> None:
        """The constraint travels into the report so a reader can see which
        library version the config expected, even when the check is advisory."""
        config = load_config(write(tmp_path, DOC_CONFIG))
        assert config.project.toolproof_version == ">=0.1,<0.2"


class TestSuiteModel:
    def test_a_suite_name_must_be_dotted_lowercase(self, tmp_path: Path) -> None:
        """Same rule as the dataset's ``suite`` field, so a config and its data
        cannot disagree about the suite's identity."""
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "NextRole"\npath = "x.jsonl"\nsplit = "test"\n',
        )
        with pytest.raises(ConfigError, match="name"):
            load_config(path)

    def test_an_invalid_split_is_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path,
            '[project]\nname = "x"\n[target]\nentrypoint = "m:f"\n'
            '[[suite]]\nname = "a.b"\npath = "x.jsonl"\nsplit = "train"\n',
        )
        with pytest.raises(ConfigError, match="split"):
            load_config(path)

    def test_a_config_object_is_frozen(self, tmp_path: Path) -> None:
        """A loaded config is a record of what was asked for. Mutating it after
        the report records its hash would make the hash a lie."""
        config = load_config(write(tmp_path, DOC_CONFIG))
        with pytest.raises((ValueError, TypeError)):
            config.run.repeats = 9


class TestConfigIsImportable:
    def test_config_is_constructible_without_a_file(self) -> None:
        """Programmatic use must not require writing TOML to a temp directory."""
        config = Config.model_validate(
            {
                "project": {"name": "x"},
                "target": {"entrypoint": "m:f"},
                "suite": [{"name": "a.b", "path": "x.jsonl", "split": "test"}],
            }
        )
        assert config.suites[0].name == "a.b"
