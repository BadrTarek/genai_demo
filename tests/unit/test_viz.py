import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
viz = pytest.importorskip("viz")

from etic_genai.agents import RAGAgent  # noqa: E402
from etic_genai.config import Settings  # noqa: E402


@pytest.fixture
def graph(knowledge_base, fake_llm_factory):
    agent = RAGAgent(
        knowledge_base, llm=fake_llm_factory(), settings=Settings(_env_file=None, llm_model="x")
    )
    return agent.graph.get_graph()


def test_graph_highlights_executed_path_and_failed_node(graph):
    dot = viz.graph_dot(graph, ["input_guardrails", "finalize"], failed_node="input_guardrails")

    assert f'"input_guardrails" [label="Input guardrails", color="{viz.CRITICAL}"' in dot
    assert f'"finalize" [label="Finalize", color="{viz.BLUE}"' in dot
    assert f'"input_guardrails" -> "finalize" [style=dashed, color="{viz.BLUE}"' in dot
    assert '"retrieve" [label="Retrieve"]' in dot  # not executed, not highlighted


def test_graph_counts_retries(graph):
    path = ["input_guardrails", "retrieve", "generate", "output_guardrails", "generate"]
    assert 'label="Generate ×2"' in viz.graph_dot(graph, path)


def test_describe_update():
    assert viz.describe_update("retrieve", {"contexts": [{}, {}]}) == "2 relevant passages"
    assert viz.describe_update(
        "output_guardrails",
        {"status": "retry", "feedback": "No citations.", "evaluations": {}},
    ) == "Rejected, regenerating: No citations."


def test_frames_for_charts():
    frame = viz.passages_frame([{"source": "a.md", "section": "S", "score": 0.5}], 0.3)
    assert frame.loc[0, "passage"] == "[1] a.md > S" and bool(frame.loc[0, "used"])

    report = {
        "cases": [
            {
                "id": "c1",
                "status": "ok",
                "attempts": 1,
                "latency_s": 1.0,
                "scores": {
                    "faithfulness": {"score": 0.5, "passed": False, "skipped": False},
                    "retrieval_recall": {"score": 0.0, "passed": True, "skipped": True},
                },
            }
        ]
    }
    cases = viz.cases_frame(report)
    assert cases.loc[0, "faithfulness"] == 0.5
    assert cases.loc[0, "retrieval_recall"] is None
    assert cases.loc[0, "failures"] == 1
