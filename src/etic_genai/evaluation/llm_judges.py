"""LLM-as-a-judge evaluators. Each one renders a prompt template from
`prompts/templates/eval_*.md` and asks the judge model for a structured verdict."""

from typing import ClassVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from etic_genai.core.logging import get_logger
from etic_genai.evaluation.base import EvalSample, Evaluator
from etic_genai.prompts import load_template

logger = get_logger(__name__)


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="One or two sentences explaining the score.")
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 (worst) and 1 (best).")


def format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {text}" for i, text in enumerate(contexts, start=1)) or "(none)"


class LLMJudgeEvaluator(Evaluator):
    template_name: ClassVar[str]

    def __init__(self, llm: BaseChatModel, threshold: float = 0.7) -> None:
        super().__init__(threshold)
        self.judge = llm.with_structured_output(JudgeVerdict, method="function_calling")

    def score(self, sample: EvalSample) -> tuple[float, str]:
        prompt = load_template(self.template_name).format(
            question=sample.question,
            answer=sample.answer,
            contexts=format_contexts(sample.contexts),
            reference=sample.reference or "",
        )
        try:
            verdict = self.judge.invoke([HumanMessage(content=prompt)])
        except Exception as exc:
            # A broken judge must never pass silently: score 0 and say why.
            logger.warning("Judge %s failed: %s", self.name, exc)
            return 0.0, f"judge error: {exc}"
        if isinstance(verdict, dict):
            verdict = JudgeVerdict.model_validate(verdict)
        return verdict.score, verdict.reasoning


class FaithfulnessEvaluator(LLMJudgeEvaluator):
    """Is every claim in the answer supported by the retrieved passages? (hallucination check)"""

    name = "faithfulness"
    template_name = "eval_faithfulness"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.answer and sample.contexts else "no answer or no contexts"


class AnswerRelevanceEvaluator(LLMJudgeEvaluator):
    """Does the answer actually address the question?"""

    name = "answer_relevance"
    template_name = "eval_answer_relevance"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.answer else "no answer"


class ContextRelevanceEvaluator(LLMJudgeEvaluator):
    """Did retrieval return passages that contain what is needed to answer?"""

    name = "context_relevance"
    template_name = "eval_context_relevance"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.contexts else "no contexts"


class AnswerCorrectnessEvaluator(LLMJudgeEvaluator):
    """Does the answer agree with the reference answer? (offline only, needs labels)"""

    name = "answer_correctness"
    template_name = "eval_answer_correctness"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.answer and sample.reference else "no answer or no reference"
