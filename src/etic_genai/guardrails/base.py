"""Guardrail building blocks.

A guardrail inspects a piece of text (the user question or the model answer) and
returns a `GuardrailResult`. It can pass, pass with a rewritten text (e.g. PII
redaction), or fail. A `GuardrailPipeline` runs guardrails in order and stops at
the first failure.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


@dataclass
class GuardrailContext:
    """Extra information output guardrails need to judge an answer."""

    question: str = ""
    contexts: list[str] = field(default_factory=list)


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    text: str
    reason: str = ""
    modified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self, stage: str) -> dict[str, Any]:
        event = asdict(self)
        event.pop("text")
        return {"stage": stage, **event}


class Guardrail(ABC):
    name: ClassVar[str]

    @abstractmethod
    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        """Inspect `text`; return the (possibly rewritten) text and a verdict."""

    def _pass(self, text: str, **metadata: Any) -> GuardrailResult:
        return GuardrailResult(self.name, True, text, metadata=metadata)

    def _fail(self, text: str, reason: str, **metadata: Any) -> GuardrailResult:
        return GuardrailResult(self.name, False, text, reason=reason, metadata=metadata)


@dataclass
class PipelineResult:
    passed: bool
    text: str
    results: list[GuardrailResult]

    @property
    def failure(self) -> GuardrailResult | None:
        return next((result for result in self.results if not result.passed), None)

    def events(self, stage: str) -> list[dict[str, Any]]:
        # Only report what is interesting: failures and rewrites.
        return [r.to_event(stage) for r in self.results if not r.passed or r.modified]


class GuardrailPipeline:
    def __init__(self, guardrails: list[Guardrail]) -> None:
        self.guardrails = guardrails

    def run(self, text: str, context: GuardrailContext | None = None) -> PipelineResult:
        results: list[GuardrailResult] = []
        for guardrail in self.guardrails:
            result = guardrail.check(text, context)
            results.append(result)
            if not result.passed:
                return PipelineResult(False, text, results)
            text = result.text
        return PipelineResult(True, text, results)
