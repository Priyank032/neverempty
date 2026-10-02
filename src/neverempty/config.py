"""The ``neverempty.toml`` loader.

This is the first thing a new adopter writes, so validation is strict in the one
direction that matters: an unknown key is an error. A misspelled ``max_cost_usd``
that loaded silently would remove the budget cap from a live run, and the first
sign of it would be the bill.

``tomllib`` is standard library from 3.11. On 3.10 the loader falls back to
``tomli`` if present and otherwise says plainly that TOML config needs one of
them, rather than failing with an import error from three frames down.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from neverempty.report.gate import GateConfig
from neverempty.tracer.pricing import Pricing

_SUITE_NAME = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")
_ENTRYPOINT = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

Split: TypeAlias = Literal["dev", "test"]
Mode: TypeAlias = Literal["live", "replay", "record"]

KNOWN_SCORERS = frozenset(
    {
        "route",
        "tool_selection",
        "forbidden_tools",
        "arguments",
        "facts",
        "forbidden_claims",
        "failure_handling",
        "false_alarm",
        "calibration",
    }
)
"""Scorer names a config may request.

Validated here rather than at run time: a typo in a scorer name is an
expectation nobody measures, and an unmeasured expectation cannot fail.
"""


class ConfigError(ValueError):
    """The config file is missing, malformed, or asks for something unknown."""


class _Strict(BaseModel):
    """Frozen and closed. A loaded config is a record of what was asked for, and
    mutating it after the report records its hash would make the hash a lie."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Strict):
    name: str = Field(min_length=1)
    neverempty_version: str | None = None
    """Recorded, not enforced. It travels into the report so a reader can see
    which library version the config expected."""


class TargetConfig(_Strict):
    entrypoint: str
    git_sha_from: Literal["git", "none"] = "git"
    prompt_modules: list[str] = Field(default_factory=list)
    side_effect_tools: list[str] = Field(default_factory=list)
    stubs: str | None = None

    @field_validator("entrypoint")
    @classmethod
    def _importable(cls, value: str) -> str:
        if not _ENTRYPOINT.match(value):
            raise ValueError(
                f"entrypoint must be 'module.path:attribute', got {value!r}; "
                f"without the colon there is nothing to import, and the failure "
                f"would surface mid-run rather than at load time"
            )
        return value


class SuiteConfig(_Strict):
    name: str
    path: str = Field(min_length=1)
    split: Split
    suite_version: int = Field(default=1, ge=1)
    scorers: list[str] = Field(default_factory=list)
    """Empty means none. Choosing scorers for the user would measure something
    they did not ask for and publish it under their suite name."""

    @field_validator("name")
    @classmethod
    def _shape(cls, value: str) -> str:
        if not _SUITE_NAME.match(value):
            raise ValueError(
                f"suite name must be dotted lowercase like 'nextrole.routing', got {value!r}"
            )
        return value

    @field_validator("scorers")
    @classmethod
    def _known(cls, value: list[str]) -> list[str]:
        unknown = [name for name in value if name not in KNOWN_SCORERS]
        if unknown:
            raise ValueError(
                f"unknown scorer(s) {unknown}; known scorers are "
                f"{sorted(KNOWN_SCORERS)}. A typo here is an expectation nobody "
                f"measures, and an unmeasured expectation cannot fail."
            )
        return value


class RunConfig(_Strict):
    repeats: int = Field(default=1, ge=1)
    """Defaults to 1, not the doc example's 3: a default of 3 would triple the
    cost of a first run without the user asking."""
    concurrency: int = Field(default=8, ge=1)
    case_timeout_s: float = Field(default=120, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    """``None`` is uncapped, which is different from a cap of 0."""
    seed: int = 20260921
    cache: str | None = None
    mode: Mode = "live"


class JudgeConfig(_Strict):
    provider: str | None = None
    model: str | None = None
    agent_family: str | None = None
    calibration: str | None = None
    prompt_version: str | None = None


class GateBlock(_Strict):
    baseline: str | None = None
    must_pass: Literal["fail_on_any"] = "fail_on_any"  # noqa: S105 - a policy, not a secret
    paired_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    floors: dict[str, float] = Field(default_factory=dict)
    warn: dict[str, float] = Field(default_factory=dict)
    max_unstable_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    primary: str | None = None

    def to_gate_config(self, *, seed: int) -> GateConfig:
        """One definition of the thresholds, so the file and the gate cannot
        drift apart."""
        return GateConfig(
            paired_alpha=self.paired_alpha,
            max_unstable_rate=self.max_unstable_rate,
            floors=dict(self.floors),
            warn=dict(self.warn),
            must_pass=self.must_pass,
            primary=self.primary,
            seed=seed,
        )


class Config(_Strict):
    """A parsed ``neverempty.toml``."""

    project: ProjectConfig
    target: TargetConfig
    suites: list[SuiteConfig] = Field(min_length=1, alias="suite")
    run: RunConfig = Field(default_factory=RunConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    gate: GateBlock = Field(default_factory=GateBlock)
    pricing: Pricing | None = None
    """Prices for the models this run will call, or ``None``.

    Optional, but required in practice by ``run.max_cost_usd``: with no table
    every cost is null, so the budget cannot be measured against and the runner
    refuses to start rather than pretend it is enforcing one. ``ModelPrice``'s
    own rules carry over, so an undated or uncited price is still refused here.
    """
    source: Path | None = None
    """Where this config was loaded from, so relative paths can be resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("suites")
    @classmethod
    def _unique_names(cls, value: list[SuiteConfig]) -> list[SuiteConfig]:
        names = [suite.name for suite in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate suite name(s) {duplicates}; one report describes one "
                f"suite, so two suites sharing a name would overwrite each other"
            )
        return value

    def resolve(self, path: str) -> Path:
        """A path from the config, found without consulting the shell.

        Two conventions are in use and both have to work:

        - **Repo-root relative**, which the design doc writes and every
          hand-written config follows: a config at ``evals/neverempty.toml``
          naming ``evals/nextrole/routing.jsonl``. Anchoring that to the
          config's own directory produced ``evals/evals/nextrole/...``.
        - **Config relative**, which ``neverempty init`` writes: the same
          config naming ``datasets/routing.jsonl``.

        Whichever one exists wins. When neither does, the config's own
        directory is reported: a config kept at the project root has no parent
        worth naming, and a path above the project would send a reader looking
        outside their own repository.

        Never the working directory: running the CLI from somewhere else must
        not change which dataset a suite names, which is the whole reason this
        method exists rather than passing the string through.
        """
        candidate = Path(path)
        if candidate.is_absolute() or self.source is None:
            return candidate

        config_dir = self.source.parent
        project_root = config_dir.parent

        beside_config = (config_dir / candidate).resolve()
        from_root = (project_root / candidate).resolve()

        if beside_config.exists():
            return beside_config
        if from_root.exists():
            return from_root
        # Neither exists, so this is an error message rather than a lookup.
        # The config's own directory is the better guess: a config kept at the
        # project root has no parent worth naming, and reporting a path above
        # the project would send a reader looking outside their own repo.
        return beside_config

    def config_hash(self) -> str:
        """SHA-256 over the parsed values, not the file bytes.

        Adding a comment or reformatting does not read as changing the
        configuration, but changing any value does.
        """
        payload = self.model_dump(mode="json", exclude={"source"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _toml_module() -> Any:
    """``tomllib`` on 3.11+, ``tomli`` on 3.10.

    Resolved at call time rather than at import, so a 3.10 user without ``tomli``
    gets a sentence saying what to install instead of an ImportError from three
    frames down.
    """
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:  # pragma: no cover - only on 3.10
        pass
    try:
        import tomli
    except ModuleNotFoundError as exc:  # pragma: no cover - bare 3.10
        raise ConfigError(
            "reading a TOML config needs Python 3.11+ (tomllib) or the 'tomli' package on 3.10"
        ) from exc
    return tomli


def _load_toml(path: Path) -> dict[str, Any]:
    toml = _toml_module()
    try:
        with path.open("rb") as handle:
            loaded: dict[str, Any] = toml.load(handle)
    except toml.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    return loaded


def load_config(path: str | Path) -> Config:
    """Load and validate a config file.

    Every failure names the file and the offending key, because this is the
    file a new adopter is most likely to get wrong.
    """
    target = Path(path)
    if not target.is_file():
        raise ConfigError(f"config file not found: {target}")

    raw = _load_toml(target)
    try:
        return Config.model_validate({**raw, "source": target})
    except ValidationError as exc:
        raise ConfigError(_explain(target, exc)) from exc


def _explain(path: Path, error: ValidationError) -> str:
    """A message that names the file and each bad key."""
    problems: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "(root)"
        message = detail["msg"]
        if detail["type"] == "extra_forbidden":
            message = (
                "unknown key. A misspelled key that loaded silently would change "
                "what the run does without saying so"
            )
        problems.append(f"  {location}: {message}")
    return f"{path.name} is invalid:\n" + "\n".join(problems)


__all__ = [
    "KNOWN_SCORERS",
    "Config",
    "ConfigError",
    "GateBlock",
    "JudgeConfig",
    "Mode",
    "ProjectConfig",
    "RunConfig",
    "Split",
    "SuiteConfig",
    "TargetConfig",
    "load_config",
]
