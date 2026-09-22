# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is below 1.0.0, the public API may change in any minor
release. The public API is exactly what `toolproof/__init__.py` exports via
`__all__`; nothing else is public. The trace and case JSON Schemas are
versioned independently of the package.

## [Unreleased]

### Added

- Nothing yet.

## [0.0.1] - 2026-09-22

Name reservation and repository skeleton. No public API yet.

### Added

- Repository skeleton: `pyproject.toml` (hatchling), ruff, `mypy --strict`,
  pytest with coverage.
- `ci.yml`: ruff, ruff format, `mypy --strict`, pytest on Python 3.10 to 3.13,
  coverage gate, wheel build, and a clean-venv install check.
- `release.yml`: PyPI trusted publishing over OIDC with no stored API token,
  plus a GitHub release rendered from this file.
- Empty typed package that imports cleanly, ships `py.typed`, and depends only
  on pydantic v2.
- `toolproof --version` console script.

[Unreleased]: https://github.com/priyank-agrawal/toolproof/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/priyank-agrawal/toolproof/releases/tag/v0.0.1
