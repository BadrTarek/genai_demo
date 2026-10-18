"""Evaluators that need no LLM: cheap, reproducible, and safe to run in CI."""

from etic_genai.evaluation.base import EvalSample, Evaluator


class RetrievalRecallEvaluator(Evaluator):
    """Fraction of the expected source documents that retrieval returned."""

    name = "retrieval_recall"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.expected_sources else "no expected sources"

    def score(self, sample: EvalSample) -> tuple[float, str]:
        expected = set(sample.expected_sources)
        found = expected & set(sample.retrieved_sources)
        missing = sorted(expected - found)
        return len(found) / len(expected), f"missing: {missing}" if missing else "all found"


class RetrievalMRREvaluator(Evaluator):
    """Reciprocal rank of the first expected source (1.0 = it was the top hit)."""

    name = "retrieval_mrr"

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.expected_sources else "no expected sources"

    def score(self, sample: EvalSample) -> tuple[float, str]:
        for rank, source in enumerate(sample.retrieved_sources, start=1):
            if source in sample.expected_sources:
                return 1.0 / rank, f"first expected source at rank {rank}"
        return 0.0, "no expected source retrieved"


class ExpectedStatusEvaluator(Evaluator):
    """Checks the agent's behaviour: answered, blocked by a guardrail, or declined."""

    name = "expected_status"

    def __init__(self) -> None:
        super().__init__(threshold=1.0)

    def skip_reason(self, sample: EvalSample) -> str | None:
        return None if sample.expected_status else "no expected status"

    def score(self, sample: EvalSample) -> tuple[float, str]:
        ok = sample.status == sample.expected_status
        return float(ok), f"expected {sample.expected_status}, got {sample.status}"
