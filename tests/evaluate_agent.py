from uuid import uuid4

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

from etic_genai.agents import AnswerJudgeAgent, RetrievalJudgeAgent
from etic_genai.api.dependencies import get_agent, get_ingestion_services
from etic_genai.api.services.chat import run_chat_agent
from etic_genai.auth.store import AuthUser

client = Client()
retrieval_judge = RetrievalJudgeAgent()
answer_judge = AnswerJudgeAgent()


def retrieval_relevance_evaluator(run: Run, example: Example | None = None):
    del example
    outputs = run.outputs or {}
    question = outputs["question"]
    context = outputs["context"]
    evaluation = retrieval_judge.judge(question, context)

    return {
        "key": "retrieval_relevance",
        "score": float(evaluation.score),
        "comment": evaluation.reasoning,
    }


def answer_correctness_evaluator(run: Run, example: Example | None = None):
    if example is None or not example.outputs:
        raise ValueError("Answer evaluation requires reference facts in the dataset outputs.")
    reference_answer = example.outputs.get("expected_context")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValueError(
            "Dataset examples must include non-empty 'expected_context' reference facts."
        )

    outputs = run.outputs or {}
    evaluation = answer_judge.judge(
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


def retrieve_and_answer(inputs: dict) -> dict:
    """Run the same retrieval and assistant flow as the chat endpoint."""
    services = get_ingestion_services()
    question = inputs["question"]
    current_user = AuthUser(
        user_id=inputs["user_id"],
        email="evaluation@example.com",
    )
    result = run_chat_agent(
        message=question,
        conversation_id=str(uuid4()),
        current_user=current_user,
        agent=get_agent(),
        services=services,
    )

    return {
        "question": question,
        "context": result.context,
        "answer": result.answer,
    }


dataset_name = "brd-retrieval-evaluation"

# Create the dataset and example if they do not already exist.
try:
    dataset = client.read_dataset(dataset_name=dataset_name)
except Exception:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluate BRD document retrieval quality",
    )
    client.create_example(
        inputs={
            "question": "what is the audit requirment",
            "user_id": "0e11c41d-37cf-4620-8957-ad33f3b28951",
        },
        outputs={
            "expected_context": (
                "Customer creation must be recorded in the audit log. "
                "Customer updates must record both the previous and new "
                "values. Regular employees must not be able to delete "
                "audit records."
            )
        },
        dataset_id=dataset.id,
    )


results = evaluate(
    retrieve_and_answer,
    data=dataset_name,
    evaluators=[retrieval_relevance_evaluator, answer_correctness_evaluator],
    experiment_prefix="brd-rag",
)

print(results)
