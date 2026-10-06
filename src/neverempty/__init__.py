"""neverempty: evaluate and trace tool-calling LLM agents.

The public surface of this library is exactly what ``__all__`` names below.
Anything reachable but unnamed here is private and may change in any release.
"""

# No ``from __future__ import annotations`` here: it would bind ``annotations``
# as a public module attribute, which the public-surface test rejects. Other
# modules in the package are free to use it.

from neverempty import scorers
from neverempty.config import Config, ConfigError, load_config
from neverempty.core.faults import FaultSpec, fault_scope
from neverempty.core.results import AnyToolResult, Empty, Err, ErrorKind, Ok, ToolResult
from neverempty.core.stubs import stub_scope
from neverempty.core.tool import AmbiguousEmptyError, tool
from neverempty.core.trace import Cost, Env, FinalOutput, Span, Trace, Usage
from neverempty.dataset.case import Case, Expect, Provenance
from neverempty.dataset.loader import Dataset, DatasetError
from neverempty.judge.calibration import (
    CalibrationCase,
    CalibrationResult,
    calibrate,
    cohens_kappa,
    load_calibration,
)
from neverempty.judge.judge import Claim, ClaimJudge, Verdict
from neverempty.judge.model import JudgeError, JudgeModel, ScriptedJudge
from neverempty.metrics.reliability import (
    Bucket,
    ReliabilityCurve,
    curve_from_outcomes,
    reliability_curve,
)
from neverempty.metrics.stats import Interval, McNemarResult
from neverempty.report.gate import CompareResult, GateConfig, GateResult, compare, gate
from neverempty.report.render import render_gate, render_markdown, render_reliability
from neverempty.report.report import CaseOutcome, Metric, Report, Score
from neverempty.runner.runner import PreflightError, Runner, Scorer, ScorerError
from neverempty.tracer import redact
from neverempty.tracer.pricing import ModelPrice, Pricing
from neverempty.tracer.sinks import JsonlSink, MemorySink, MultiSink, NullSink, Sink
from neverempty.tracer.tracer import SpanHandle, Tracer, TraceRun

__all__ = [
    "AmbiguousEmptyError",
    "AnyToolResult",
    "Bucket",
    "CalibrationCase",
    "CalibrationResult",
    "Case",
    "CaseOutcome",
    "Claim",
    "ClaimJudge",
    "CompareResult",
    "Config",
    "ConfigError",
    "Cost",
    "Dataset",
    "DatasetError",
    "Empty",
    "Env",
    "Err",
    "ErrorKind",
    "Expect",
    "FaultSpec",
    "FinalOutput",
    "GateConfig",
    "GateResult",
    "Interval",
    "JsonlSink",
    "JudgeError",
    "JudgeModel",
    "McNemarResult",
    "MemorySink",
    "Metric",
    "ModelPrice",
    "MultiSink",
    "NullSink",
    "Ok",
    "PreflightError",
    "Pricing",
    "Provenance",
    "ReliabilityCurve",
    "Report",
    "Runner",
    "Score",
    "Scorer",
    "ScorerError",
    "ScriptedJudge",
    "Sink",
    "Span",
    "SpanHandle",
    "ToolResult",
    "Trace",
    "TraceRun",
    "Tracer",
    "Usage",
    "Verdict",
    "__version__",
    "calibrate",
    "cohens_kappa",
    "compare",
    "curve_from_outcomes",
    "fault_scope",
    "gate",
    "load_calibration",
    "load_config",
    "redact",
    "reliability_curve",
    "render_gate",
    "render_markdown",
    "render_reliability",
    "scorers",
    "stub_scope",
    "tool",
]

__version__ = "0.1.1"
