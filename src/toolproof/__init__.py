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
from toolproof.core.trace import Cost, Env, FinalOutput, Span, Trace, Usage
from toolproof.tracer import redact
from toolproof.tracer.pricing import ModelPrice, Pricing
from toolproof.tracer.sinks import JsonlSink, MemorySink, MultiSink, NullSink, Sink
from toolproof.tracer.tracer import SpanHandle, Tracer, TraceRun

__all__ = [
    "AmbiguousEmptyError",
    "AnyToolResult",
    "Cost",
    "Empty",
    "Env",
    "Err",
    "ErrorKind",
    "FaultSpec",
    "FinalOutput",
    "JsonlSink",
    "MemorySink",
    "ModelPrice",
    "MultiSink",
    "NullSink",
    "Ok",
    "Pricing",
    "Sink",
    "Span",
    "SpanHandle",
    "ToolResult",
    "Trace",
    "TraceRun",
    "Tracer",
    "Usage",
    "__version__",
    "fault_scope",
    "redact",
    "tool",
]

__version__ = "0.0.1"
