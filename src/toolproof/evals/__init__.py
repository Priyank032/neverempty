"""Eval harness: suite definitions, coverage checks and target adapters.

Separate from ``toolproof.dataset`` on purpose. ``dataset`` validates one case
against the schema; this package decides whether a *collection* of cases is fit
to publish a number from, and holds the adapters that turn a real agent into the
runner's ``async (Case, Tracer) -> None`` target.
"""
