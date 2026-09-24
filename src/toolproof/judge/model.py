"""The judge's model interface.

Core ships a protocol and a deterministic offline fake, never a provider SDK.
Two reasons, and the second matters more:

- Core stays at one dependency. A judge that dragged boto3 into every install
  would make the trace layer unusable for someone who only wants tracing.
- Every acceptance requirement for the judge is about *its own logic*: retrying
  malformed output, refusing a matching family, resisting an injection, caching
  a verdict. Testing those against a mocked SDK would test the mock. Testing
  them against a scripted model tests the judge.

A real provider binding is a dozen lines against this protocol, and it belongs
behind an extra where an outage cannot turn into a red build on unrelated work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class JudgeError(RuntimeError):
    """The judge is misconfigured and the run must not start.

    Distinct from a ``judge_error`` *label*, which is a per-call outcome. This is
    a setup failure: a matching model family, or a family that cannot be
    determined at all.
    """


@runtime_checkable
class JudgeModel(Protocol):
    """One text completion, at the temperature the caller asks for.

    Deliberately minimal. The judge owns the prompts, the parsing, the retries
    and the cache; a backend only has to turn two strings into one string.
    """

    async def complete(self, *, system: str, user: str, temperature: float) -> str: ...


@dataclass(frozen=True)
class RecordedCall:
    """One call a fake received, for a test to assert against."""

    system: str
    user: str
    temperature: float


@dataclass
class ScriptedJudge:
    """A deterministic offline judge that returns queued responses in order.

    Used by the library's own tests and shipped so a downstream author can test
    a judge-backed scorer without credentials. It records every call, which is
    how the injection tests can assert that the payload was delimited as data
    rather than merely trusting that it was.

    Once the script is exhausted it repeats the last response. That keeps a test
    that makes one extra call from failing with an IndexError instead of the
    assertion it was actually making.
    """

    responses: Sequence[str]
    calls: list[RecordedCall] = field(default_factory=list)

    async def complete(self, *, system: str, user: str, temperature: float) -> str:
        index = len(self.calls)
        self.calls.append(RecordedCall(system=system, user=user, temperature=temperature))
        if not self.responses:
            raise JudgeError(
                "ScriptedJudge was called with no queued responses; the test "
                "expected the judge to make no call here"
            )
        return self.responses[min(index, len(self.responses) - 1)]


__all__ = ["JudgeError", "JudgeModel", "RecordedCall", "ScriptedJudge"]
