"""toolproof: evaluate and trace tool-calling LLM agents.

The public surface of this library is exactly what ``__all__`` names below.
Anything reachable but unnamed here is private and may change in any release.

Nothing is exported yet: this is the M0 skeleton.
"""

# No ``from __future__ import annotations`` here: it would bind ``annotations``
# as a public module attribute, which the public-surface test rejects. Other
# modules in the package are free to use it.

__all__ = ["__version__"]

__version__ = "0.0.1"
