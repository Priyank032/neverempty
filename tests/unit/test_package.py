"""M0 acceptance: the package installs, imports and types cleanly.

These are deliberately about the packaging contract, not behaviour: the doc's
M0 row is done when CI is green on an empty package.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import toolproof


def test_imports_cleanly() -> None:
    assert toolproof.__version__


def test_version_matches_installed_distribution() -> None:
    assert importlib.metadata.version("toolproof") == toolproof.__version__


def test_ships_py_typed_marker() -> None:
    marker = Path(toolproof.__file__).parent / "py.typed"
    assert marker.is_file()


def test_public_surface_is_exactly_dunder_all() -> None:
    """Everything not named in ``__all__`` is private (doc: API stability).

    Submodules bound by the import system (``toolproof.cli``) are excluded,
    but a module listed in ``__all__`` (``redact``) is deliberate public API
    and must stay in the comparison.
    """
    exported = set(toolproof.__all__)
    public = {
        name
        for name, value in vars(toolproof).items()
        if not name.startswith("_") and (name in exported or not isinstance(value, ModuleType))
    }
    assert public == set(toolproof.__all__) - {"__version__"}


def test_all_names_are_importable() -> None:
    for name in toolproof.__all__:
        assert hasattr(toolproof, name), name


def test_console_script_runs() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "toolproof.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert toolproof.__version__ in proc.stdout


def test_core_import_does_not_pull_optional_extras() -> None:
    """Core depends on pydantic v2 only; importing must not need an extra."""
    code = (
        "import sys; import toolproof; "
        "banned = {'langgraph', 'langchain_core', 'openai', 'anthropic', 'boto3', "
        "'opentelemetry'}; "
        "hit = banned & {m.split('.')[0] for m in sys.modules}; "
        "assert not hit, hit"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
