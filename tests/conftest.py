"""Shared fixtures.

The side-effect registry is process-global by design: the runner's preflight has
to know every side-effecting tool that has been imported, wherever it came from.
That makes it leak across tests, so it is snapshotted and restored per test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from neverempty.core import stubs


@pytest.fixture(autouse=True)
def _isolate_side_effect_registry() -> Iterator[None]:
    saved = stubs.registered_side_effect_tools()
    stubs.clear_side_effect_registry()
    try:
        yield
    finally:
        stubs.clear_side_effect_registry()
        for name in saved:
            stubs.register_side_effect(name)
