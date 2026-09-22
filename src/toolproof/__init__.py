"""toolproof: evaluate and trace tool-calling LLM agents.

The public surface of this library is exactly what ``__all__`` names below.
Anything reachable but unnamed here is private and may change in any release.
"""

# No ``from __future__ import annotations`` here: it would bind ``annotations``
# as a public module attribute, which the public-surface test rejects. Other
# modules in the package are free to use it.

from toolproof.core.faults import FaultSpec, fault_scope
from toolproof.core.results import AnyToolResult, Empty, Err, ErrorKind, Ok, ToolResult
from toolproof.core.tool import AmbiguousEmptyError, tool

__all__ = [
    "AmbiguousEmptyError",
    "AnyToolResult",
    "Empty",
    "Err",
    "ErrorKind",
    "FaultSpec",
    "Ok",
    "ToolResult",
    "__version__",
    "fault_scope",
    "tool",
]

__version__ = "0.0.1"
