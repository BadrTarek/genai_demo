"""LangSmith evaluation helpers for retrieval and generated answers."""

from functools import lru_cache
from typing import Any
from uuid import uuid4

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

from etic_genai.agents import AnswerJudgeAgent, RetrievalJudgeAgent
from etic_genai.api.dependencies import get_agent, get_ingestion_services
from etic_genai.api.services.chat import run_chat_agent
from etic_genai.auth.store import AuthUser

DATASET_NAME = "brd-retrieval-evaluation"


@lru_cache
def _get_retrieval_judge() -> RetrievalJudgeAgent:
    return RetrievalJudgeAgent()


@lru_cache
def _get_answer_judge() -> AnswerJudgeAgent:
    return AnswerJudgeAgent()


def retrieval_relevance_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, str | float]:
    del example
    outputs = run.outputs or {}
    evaluation = _get_retrieval_judge().judge(outputs["question"], outputs["context"])
    return {
        "key": "retrieval_relevance",
        "score": float(evaluation.score),
        "comment": evaluation.reasoning,
    }


def answer_correctness_evaluator(
    run: Run,
    example: Example | None = None,
) -> dict[str, str | float]:
    if example is None or not example.outputs:
        raise ValueError("Answer evaluation requires reference facts in the dataset outputs.")
    reference_answer = example.outputs.get("expected_context")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValueError(
            "Dataset examples must include non-empty 'expected_context' reference facts."
        )

    outputs = run.outputs or {}
    evaluation = _get_answer_judge().judge(
        question=outputs["question"],
        answer=outputs["answer"],
        context=outputs["context"],
        reference_answer=reference_answer,
    )
    return {
        "key": "answer_correctness",
        "score": float(evaluation.score),
        "comment": evaluation.reasoning,
    }


def retrieve_and_answer(inputs: dict[str, Any]) -> dict[str, str]:
    """Run the chat flow for the user ID included in an evaluation example."""
    services = get_ingestion_services()
    current_user = AuthUser(
        user_id=inputs["user_id"],
        email="evaluation@example.com",
    )
    result = run_chat_agent(
        message=inputs["question"],
        conversation_id=str(uuid4()),
        current_user=current_user,
        agent=get_agent(),
        services=services,
    )
    return {
        "question": inputs["question"],
        "context": result.context,
        "answer": result.answer,
    }


def run_dataset_evaluation(
    user_id: str,
    *,
    client: Client | None = None,
) -> dict[str, str | None]:
    """Evaluate the configured dataset against only the requesting admin's documents."""
    langsmith_client = client or Client()

    def evaluate_for_user(inputs: dict[str, Any]) -> dict[str, str]:
        return retrieve_and_answer({**inputs, "user_id": user_id})

    results = evaluate(
        evaluate_for_user,
        data=DATASET_NAME,
        evaluators=[retrieval_relevance_evaluator, answer_correctness_evaluator],
        experiment_prefix="brd-rag",
        description="Evaluate BRD retrieval relevance and answer correctness.",
        max_concurrency=1,
        blocking=True,
        client=langsmith_client,
    )
    return {
        "status": "completed",
        "experiment_name": results.experiment_name,
        "experiment_id": str(results.experiment_id),
        "experiment_url": results.url,
    }
