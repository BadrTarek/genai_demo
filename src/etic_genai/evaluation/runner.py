"""Offline evaluation: run the agent over a labelled dataset and score every answer."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.language_models import BaseChatModel

from etic_genai.agents.rag_agent import RAGAgent
from etic_genai.core.logging import get_logger
from etic_genai.evaluation.base import EvalSample, Evaluator
from etic_genai.evaluation.deterministic import (
    ExpectedStatusEvaluator,
    RetrievalMRREvaluator,
    RetrievalRecallEvaluator,
)
from etic_genai.evaluation.llm_judges import (
    AnswerCorrectnessEvaluator,
    AnswerRelevanceEvaluator,
    ContextRelevanceEvaluator,
    FaithfulnessEvaluator,
)

logger = get_logger(__name__)


@dataclass
class EvalCase:
    id: str
    question: str
    reference: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    expected_status: str = "ok"


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            cases.append(EvalCase(**json.loads(line)))
    return cases


def default_evaluators(judge_llm: BaseChatModel) -> list[Evaluator]:
    return [
        ExpectedStatusEvaluator(),
        RetrievalRecallEvaluator(threshold=1.0),
        RetrievalMRREvaluator(threshold=0.5),
        ContextRelevanceEvaluator(judge_llm),
        FaithfulnessEvaluator(judge_llm),
        AnswerRelevanceEvaluator(judge_llm),
        AnswerCorrectnessEvaluator(judge_llm),
    ]


class EvaluationRunner:
    def __init__(self, agent: RAGAgent, evaluators: list[Evaluator]) -> None:
        self.agent = agent
        self.evaluators = evaluators

    def run_case(self, case: EvalCase) -> dict[str, Any]:
        started = time.perf_counter()
        state = self.agent.invoke(case.question, thread_id=f"eval-{case.id}")
        latency = time.perf_counter() - started

        contexts = state.get("contexts", [])
        answered = state.get("status") == "ok"
        sample = EvalSample(
            question=case.question,
            # Only score the answer text when the agent actually answered.
            answer=state.get("answer", "") if answered else "",
            contexts=[c.get("content", "") for c in contexts] if answered else [],
            retrieved_sources=[c.get("source", "") for c in contexts],
            reference=case.reference,
            expected_sources=case.expected_sources,
            expected_status=case.expected_status,
            status=state.get("status"),
        )
        scores = {e.name: e.evaluate(sample).to_dict() for e in self.evaluators}
        return {
            "id": case.id,
            "question": case.question,
            "status": state.get("status"),
            "attempts": state.get("attempts", 0),
            "latency_s": round(latency, 2),
            "answer": state["messages"][-1].content,
            "guardrail_events": state.get("guardrail_events", []),
            "scores": scores,
        }

    def run(self, cases: list[EvalCase]) -> dict[str, Any]:
        rows = []
        for case in cases:
            logger.info("Evaluating %s", case.id)
            rows.append(self.run_case(case))
        return {"summary": self.summarize(rows), "cases": rows}

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for evaluator in self.evaluators:
            results = [
                row["scores"][evaluator.name]
                for row in rows
                if not row["scores"][evaluator.name]["skipped"]
            ]
            if results:
                metrics[evaluator.name] = {
                    "mean": round(mean(r["score"] for r in results), 3),
                    "pass_rate": round(mean(float(r["passed"]) for r in results), 3),
                    "n": len(results),
                }
        statuses: dict[str, int] = {}
        for row in rows:
            statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        return {
            "cases": len(rows),
            "statuses": statuses,
            "mean_latency_s": round(mean(r["latency_s"] for r in rows), 2) if rows else 0,
            "metrics": metrics,
        }


def format_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"Cases: {summary['cases']}   statuses: {summary['statuses']}   "
        f"mean latency: {summary['mean_latency_s']}s",
        "",
        f"{'metric':<20}{'mean':>8}{'pass rate':>12}{'n':>5}",
        "-" * 45,
    ]
    for name, metric in summary["metrics"].items():
        lines.append(
            f"{name:<20}{metric['mean']:>8.2f}{metric['pass_rate']:>12.0%}{metric['n']:>5}"
        )
    lines += ["", "Failures:"]
    failures = [
        f"  {row['id']}: {name} = {score['score']:.2f} ({score['reason']})"
        for row in report["cases"]
        for name, score in row["scores"].items()
        if not score["passed"] and not score["skipped"]
    ]
    lines += failures or ["  none"]
    return "\n".join(lines)
