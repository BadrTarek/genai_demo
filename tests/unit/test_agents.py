from langchain_core.messages import AIMessage

from etic_genai.agents import AssistantAgent
from etic_genai.memory import get_checkpointer


def test_agent_answers_without_tools(fake_llm_factory):
    agent = AssistantAgent(llm=fake_llm_factory("hello!"), tools=[])
    result = agent.invoke("hi")
    assert result["messages"][-1].content == "hello!"


def test_agent_calls_tool_then_answers(fake_llm_factory):
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "current_datetime", "args": {"timezone": "UTC"}, "id": "call_1"}],
    )
    agent = AssistantAgent(llm=fake_llm_factory(tool_call, "It is now."))
    result = agent.invoke("what time is it?")

    types = [m.type for m in result["messages"]]
    assert types == ["human", "ai", "tool", "ai"]
    assert result["messages"][-1].content == "It is now."


def test_agent_remembers_thread(fake_llm_factory):
    agent = AssistantAgent(
        llm=fake_llm_factory("one", "two"), tools=[], checkpointer=get_checkpointer()
    )
    agent.invoke("first", thread_id="t1")
    result = agent.invoke("second", thread_id="t1")
    assert len(result["messages"]) == 4
