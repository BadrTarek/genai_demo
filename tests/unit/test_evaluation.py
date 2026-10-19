from types import SimpleNamespace

import etic_genai.evaluation as evaluation


def test_run_dataset_evaluation_limits_document_access_to_requesting_admin(monkeypatch):
    captured: dict[str, object] = {}

    def fake_retrieve_and_answer(inputs: dict) -> dict[str, str]:
        captured["inputs"] = inputs
        return {"question": inputs["question"], "context": "", "answer": "answer"}

    def fake_evaluate(target, **kwargs):
        captured["kwargs"] = kwargs
        target({"question": "test question", "user_id": "another-user"})
        return SimpleNamespace(
            experiment_name="brd-rag-test",
            experiment_id="experiment-id",
            url="https://smith.langchain.com/experiment",
        )

    monkeypatch.setattr(evaluation, "retrieve_and_answer", fake_retrieve_and_answer)
    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)

    result = evaluation.run_dataset_evaluation(
        "admin-user",
        client=SimpleNamespace(),
    )

    assert captured["inputs"] == {"question": "test question", "user_id": "admin-user"}
    assert captured["kwargs"]["data"] == evaluation.DATASET_NAME
    assert captured["kwargs"]["max_concurrency"] == 1
    assert captured["kwargs"]["blocking"] is True
    assert result == {
        "status": "completed",
        "experiment_name": "brd-rag-test",
        "experiment_id": "experiment-id",
        "experiment_url": "https://smith.langchain.com/experiment",
    }
