"""Evaluator building blocks shared by online checks (inside the agent) and offline
evaluation runs (over a labelled dataset)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class EvalSample:
    question: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    # Labels, only available offline.
    reference: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    expected_status: str | None = None
    status: str | None = None


@dataclass
class EvalResult:
    name: str
    score: float
    passed: bool
    reason: str = ""
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "passed": self.passed,
            "reason": self.reason,
            "skipped": self.skipped,
        }


class Evaluator(ABC):
    """Scores one aspect of a RAG interaction between 0 (bad) and 1 (good)."""

    name: ClassVar[str]

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def evaluate(self, sample: EvalSample) -> EvalResult:
        skip_reason = self.skip_reason(sample)
        if skip_reason:
            return EvalResult(self.name, 0.0, True, skip_reason, skipped=True)
        score, reason = self.score(sample)
        return EvalResult(self.name, score, score >= self.threshold, reason)

    def skip_reason(self, sample: EvalSample) -> str | None:
        """Return a reason when this evaluator does not apply to the sample."""
        return None

    @abstractmethod
    def score(self, sample: EvalSample) -> tuple[float, str]:
        """Return `(score, explanation)`."""
