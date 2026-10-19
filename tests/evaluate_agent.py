"""Run the retrieval and answer evaluation from the command line."""

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError

from etic_genai.evaluation import DATASET_NAME, run_dataset_evaluation

EVALUATION_USER_ID = "0e11c41d-37cf-4620-8957-ad33f3b28951"


def ensure_dataset(client: Client) -> None:
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
    except LangSmithNotFoundError:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Evaluate BRD document retrieval quality",
        )
        client.create_example(
            inputs={
                "question": "what is the audit requirment",
                "user_id": EVALUATION_USER_ID,
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


def main() -> None:
    client = Client()
    ensure_dataset(client)
    results = run_dataset_evaluation(EVALUATION_USER_ID, client=client)
    print(results)


if __name__ == "__main__":
    main()
