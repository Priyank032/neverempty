"""M0 acceptance: the package installs, imports and types cleanly.

These are deliberately about the packaging contract, not behaviour: the doc's
M0 row is done when CI is green on an empty package.
"""

from __future__ import annotations

import importlib
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


def test_no_model_field_warns_on_any_supported_pydantic() -> None:
    """A field named ``model_*`` must declare ``protected_namespaces=()``.

    pydantic reserves the ``model_`` prefix and warns on any field using it. The
    warning fires on some 2.x versions and not others, and this project turns
    warnings into errors, so an undeclared field makes the whole package fail to
    import on a pydantic well inside the declared ``>=2.7,<3`` range -- and only
    on that version, so the local suite stays green while a CI leg goes red.
    """
    import pkgutil

    from pydantic import BaseModel

    import toolproof

    offenders: list[str] = []
    for info in pkgutil.walk_packages(toolproof.__path__, f"{toolproof.__name__}."):
        module = importlib.import_module(info.name)
        for name in dir(module):
            attribute = getattr(module, name)
            if not (isinstance(attribute, type) and issubclass(attribute, BaseModel)):
                continue
            if attribute.__module__ != info.name:
                continue
            reserved = attribute.model_config.get("protected_namespaces", ("model_",))
            # pydantic allows compiled patterns here as well as prefixes; only a
            # string prefix is checked, which is the form this project uses.
            prefixes = tuple(item for item in reserved if isinstance(item, str))
            for field in attribute.model_fields:
                if field.startswith(prefixes):
                    offenders.append(f"{info.name}.{attribute.__name__}.{field}")

    assert offenders == [], (
        f"these fields collide with a pydantic protected namespace: {offenders}. "
        f"Either rename the field or set protected_namespaces=() on the model."
    )
