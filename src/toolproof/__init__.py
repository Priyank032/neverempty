"""toolproof: evaluate and trace tool-calling LLM agents.

The public surface of this library is exactly what ``__all__`` names below.
Anything reachable but unnamed here is private and may change in any release.
"""

# No ``from __future__ import annotations`` here: it would bind ``annotations``
# as a public module attribute, which the public-surface test rejects. Other
# modules in the package are free to use it.

from toolproof import scorers
from toolproof.config import Config, ConfigError, load_config
from toolproof.core.faults import FaultSpec, fault_scope
from toolproof.core.results import AnyToolResult, Empty, Err, ErrorKind, Ok, ToolResult
from toolproof.core.stubs import stub_scope
from toolproof.core.tool import AmbiguousEmptyError, tool
from toolproof.core.trace import Cost, Env, FinalOutput, Span, Trace, Usage
from toolproof.dataset.case import Case, Expect, Provenance
from toolproof.dataset.loader import Dataset, DatasetError
from toolproof.judge.calibration import (
    CalibrationCase,
    CalibrationResult,
    calibrate,
    cohens_kappa,
    load_calibration,
)
from toolproof.judge.judge import Claim, ClaimJudge, Verdict
from toolproof.judge.model import JudgeError, JudgeModel, ScriptedJudge
from toolproof.metrics.reliability import (
    Bucket,
    ReliabilityCurve,
    curve_from_outcomes,
    reliability_curve,
)
from toolproof.metrics.stats import Interval, McNemarResult
from toolproof.report.gate import CompareResult, GateConfig, GateResult, compare, gate
from toolproof.report.render import render_gate, render_markdown, render_reliability
from toolproof.report.report import CaseOutcome, Metric, Report, Score
from toolproof.runner.runner import PreflightError, Runner, Scorer, ScorerError
from toolproof.tracer import redact
from toolproof.tracer.pricing import ModelPrice, Pricing
from toolproof.tracer.sinks import JsonlSink, MemorySink, MultiSink, NullSink, Sink
from toolproof.tracer.tracer import SpanHandle, Tracer, TraceRun

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

__version__ = "0.0.1"
