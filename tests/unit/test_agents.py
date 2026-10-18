from langchain_core.messages import AIMessage

from etic_genai.agents import AssistantAgent
from etic_genai.memory import get_checkpointer, get_sqlite_checkpointer


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


def test_agent_history_survives_reopening_sqlite_checkpointer(fake_llm_factory, tmp_path):
    database_path = str(tmp_path / "chat.db")
    first_saver = get_sqlite_checkpointer(database_path)
    first_agent = AssistantAgent(
        llm=fake_llm_factory("first reply"),
        tools=[],
        checkpointer=first_saver,
    )
    first_agent.invoke("first message", thread_id="user-1:conversation-1")

    first_saver.conn.close()
    get_sqlite_checkpointer.cache_clear()
    second_saver = get_sqlite_checkpointer(database_path)
    second_agent = AssistantAgent(
        llm=fake_llm_factory("second reply"),
        tools=[],
        checkpointer=second_saver,
    )
    result = second_agent.invoke("second message", thread_id="user-1:conversation-1")

    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "first message"
    second_saver.conn.close()
    get_sqlite_checkpointer.cache_clear()
