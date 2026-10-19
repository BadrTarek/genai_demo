import pytest
from langchain_core.runnables import RunnableLambda

from etic_genai.agents import AnswerJudgeAgent, AnswerJudgment
from etic_genai.prompts import AnswerJudgePrompt


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.schema = None
        self.method = None
        self.prompt_value = None

    def with_structured_output(self, schema, *, method):
        self.schema = schema
        self.method = method

        def respond(prompt_value):
            self.prompt_value = prompt_value
            return self.response

        return RunnableLambda(respond)


def test_answer_judge_returns_typed_score_and_uses_json_schema():
    model = FakeStructuredModel(
        AnswerJudgment(score=1, reasoning="The answer matches the evidence.")
    )
    agent = AnswerJudgeAgent(llm=model)
    judgment = agent.judge(
        question="What does AUD-104 require?",
        answer="Audit events must be recorded.",
        context="AUD-104 requires audit logging.",
        reference_answer="The system must record audit events.",
    )

    assert judgment.score == 1
    assert judgment.reasoning == "The answer matches the evidence."
    assert model.schema is AnswerJudgment
    assert model.method == "json_schema"


def test_answer_judge_prompt_checks_correctness_and_evidence():
    prompt = AnswerJudgePrompt().system_template.lower()

    assert "factually correct" in prompt
    assert "retrieved context" in prompt
    assert "reference answer" in prompt
    assert "unsupported" in prompt


def test_answer_judge_passes_question_answer_context_and_reference():
    model = FakeStructuredModel(AnswerJudgment(score=1, reasoning="Supported."))
    agent = AnswerJudgeAgent(llm=model)
    agent.judge("Question", "Answer", "Context", "Reference")

    content = "\n".join(str(message.content) for message in model.prompt_value.to_messages())
    for value in ("Question", "Answer", "Context", "Reference"):
        assert value in content


def test_answer_judge_rejects_untyped_model_output():
    agent = AnswerJudgeAgent(llm=FakeStructuredModel({"score": 1, "reasoning": "Supported."}))

    with pytest.raises(TypeError, match="Structured LLM output must be AnswerJudgment"):
        agent.judge("Question", "Answer", "Context", "Reference")
