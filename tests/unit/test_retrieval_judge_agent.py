import pytest
from langchain_core.messages import AIMessage

from etic_genai.agents import RetrievalJudgeAgent
from etic_genai.prompts import RetrievalJudgePrompt


def test_retrieval_judge_returns_validated_score_and_reason(fake_llm_factory):
    agent = RetrievalJudgeAgent(
        llm=fake_llm_factory('{"score": 1, "reasoning": "Context answers the question."}')
    )

    judgment = agent.judge("What must be logged?", "Customer creation must be logged.")

    assert judgment.score == 1
    assert judgment.reasoning == "Context answers the question."


def test_retrieval_judge_prompt_includes_retrieval_rubric():
    prompt = RetrievalJudgePrompt().system_template

    assert "retrieval only" in prompt.lower()
    assert "score 1" in prompt.lower()
    assert "score 0" in prompt.lower()


def test_retrieval_judge_passes_question_and_context(fake_llm_factory, monkeypatch):
    agent = RetrievalJudgeAgent(llm=fake_llm_factory('{"score": 0, "reasoning": "No evidence."}'))
    captured = {}

    def capture_input(input):
        captured.update(input)
        return {"messages": [AIMessage(content='{"score": 0, "reasoning": "No evidence."}')]}

    monkeypatch.setattr(agent, "invoke", capture_input)
    agent.judge("Question text", "Retrieved evidence")

    message = captured["messages"][0]
    assert "Question text" in message.content
    assert "Retrieved evidence" in message.content


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"score": 2, "reasoning": "Invalid score."}',
        '{"score": 1}',
    ],
)
def test_retrieval_judge_rejects_invalid_output(fake_llm_factory, response):
    agent = RetrievalJudgeAgent(llm=fake_llm_factory(response))

    with pytest.raises(ValueError, match="invalid JSON or an invalid score"):
        agent.judge("Question", "Context")
