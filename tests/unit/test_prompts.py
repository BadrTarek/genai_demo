import pytest
from langchain_core.messages import HumanMessage

from etic_genai.core.exceptions import PromptNotFoundError
from etic_genai.prompts import AssistantPrompt, load_template


def test_assistant_prompt_formats():
    prompt = AssistantPrompt().to_chat_prompt(agent_name="bot")
    messages = prompt.invoke(
        {
            "messages": [HumanMessage(content="hi")],
            "retrieved_context": "Retrieved context",
        }
    ).to_messages()
    assert "You are bot" in messages[0].content
    assert "Retrieved context" in messages[0].content
    assert messages[-1].content == "hi"


def test_missing_template():
    with pytest.raises(PromptNotFoundError):
        load_template("does_not_exist")
