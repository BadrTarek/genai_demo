"""Guardrails applied to the generated answer before it reaches the user."""

import re

from etic_genai.evaluation.base import EvalSample, Evaluator
from etic_genai.guardrails.base import Guardrail, GuardrailContext, GuardrailResult

NOT_FOUND_ANSWER = "I could not find this in the knowledge base."
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> list[int]:
    return sorted({int(n) for n in CITATION_PATTERN.findall(answer)})


class CitationGuardrail(Guardrail):
    """Every answer must cite at least one passage, and only passages that exist."""

    name = "citations"

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        if NOT_FOUND_ANSWER.lower() in text.lower():
            return self._pass(text)
        available = len(context.contexts) if context else 0
        citations = cited_indices(text)
        if not citations:
            return self._fail(text, "The answer does not cite any source passage like [1].")
        invalid = [n for n in citations if not 1 <= n <= available]
        if invalid:
            return self._fail(
                text, f"The answer cites passages that do not exist: {invalid}.", invalid=invalid
            )
        return self._pass(text, citations=citations)


class GroundednessGuardrail(Guardrail):
    """Blocks answers that are not supported by the retrieved passages (hallucinations).

    Wraps a faithfulness evaluator, so the same judge is used online (here) and offline
    (in the evaluation runner).
    """

    name = "groundedness"

    def __init__(self, evaluator: Evaluator, threshold: float = 0.7) -> None:
        self.evaluator = evaluator
        self.threshold = threshold

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        context = context or GuardrailContext()
        result = self.evaluator.evaluate(
            EvalSample(question=context.question, answer=text, contexts=context.contexts)
        )
        if result.skipped:
            return self._pass(text, skipped=result.reason)
        metadata = {"score": result.score, "threshold": self.threshold}
        if result.score < self.threshold:
            return self._fail(
                text, f"The answer is not grounded in the sources: {result.reason}", **metadata
            )
        return self._pass(text, **metadata)
