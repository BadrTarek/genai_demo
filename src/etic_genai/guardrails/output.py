"""Guardrails applied to the generated answer before it reaches the user."""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from etic_genai.core.logging import get_logger
from etic_genai.guardrails.base import Guardrail, GuardrailContext, GuardrailResult
from etic_genai.prompts import load_template

logger = get_logger(__name__)

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


class GroundednessVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the score.")
    score: float = Field(ge=0.0, le=1.0, description="0 = unsupported, 1 = fully supported.")


class GroundednessGuardrail(Guardrail):
    """Blocks answers that are not supported by the retrieved passages (hallucinations),
    using an LLM judge (`prompts/templates/guard_groundedness.md`)."""

    name = "groundedness"

    def __init__(self, llm: BaseChatModel, threshold: float = 0.7) -> None:
        self.judge = llm.with_structured_output(GroundednessVerdict, method="function_calling")
        self.threshold = threshold

    def check(self, text: str, context: GuardrailContext | None = None) -> GuardrailResult:
        context = context or GuardrailContext()
        if not context.contexts:
            return self._pass(text, skipped="no passages to check against")
        prompt = load_template("guard_groundedness").format(
            question=context.question,
            answer=text,
            contexts="\n\n".join(f"[{i}] {c}" for i, c in enumerate(context.contexts, start=1)),
        )
        try:
            verdict = self.judge.invoke([HumanMessage(content=prompt)])
            if isinstance(verdict, dict):
                verdict = GroundednessVerdict.model_validate(verdict)
        except Exception as exc:
            # A broken judge must never let an unchecked answer through.
            logger.warning("Groundedness judge failed: %s", exc)
            return self._fail(text, f"The groundedness check could not run: {exc}", score=0.0)

        metadata = {"score": verdict.score, "threshold": self.threshold}
        if verdict.score < self.threshold:
            return self._fail(
                text, f"The answer is not grounded in the sources: {verdict.reasoning}", **metadata
            )
        return self._pass(text, **metadata)
