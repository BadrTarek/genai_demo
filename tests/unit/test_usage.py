import sqlite3

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from etic_genai.agents import AssistantAgent
from etic_genai.auth.store import AuthStore
from etic_genai.usage.callback import UsageTracker
from etic_genai.usage.store import UsageStore


def test_tracker_collects_usage_from_multiple_model_calls():
    tracker = UsageTracker()
    first_call = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="tool call",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        },
                    )
                )
            ]
        ]
    )
    second_call = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="answer",
                        usage_metadata={
                            "input_tokens": 150,
                            "output_tokens": 30,
                            "total_tokens": 180,
                        },
                    )
                )
            ]
        ]
    )

    tracker.on_llm_end(first_call)
    tracker.on_llm_end(second_call)

    assert tracker.input_tokens == 250
    assert tracker.output_tokens == 50
    assert tracker.total_tokens == 300
    assert tracker.model_calls == 2
    assert tracker.unreported_calls == 0


def test_agent_passes_tracker_to_model_callbacks(fake_llm_factory):
    tracker = UsageTracker()
    reply = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 3,
            "total_tokens": 15,
        },
    )
    agent = AssistantAgent(llm=fake_llm_factory(reply), tools=[])

    result = agent.invoke("question", callbacks=[tracker])

    assert result["messages"][-1].content == "answer"
    assert tracker.input_tokens == 12
    assert tracker.output_tokens == 3
    assert tracker.total_tokens == 15
    assert tracker.model_calls == 1


def test_tracker_marks_usage_missing_from_model_response():
    tracker = UsageTracker()
    tracker.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="answer"))]])
    )

    assert tracker.model_calls == 1
    assert tracker.unreported_calls == 1
    assert tracker.total_tokens == 0


def test_usage_store_accumulates_by_user_and_conversation(tmp_path):
    database_path = str(tmp_path / "usage.db")
    auth_store = AuthStore(database_path)
    user = auth_store.create_user("person@example.com", "StrongPassword1!")
    store = UsageStore(database_path)

    first_request = UsageTracker()
    first_request.input_tokens = 100
    first_request.output_tokens = 25
    first_request.total_tokens = 125
    first_request.model_calls = 2
    store.add_usage(user.user_id, "conversation-1", first_request)

    second_request = UsageTracker()
    second_request.input_tokens = 50
    second_request.output_tokens = 10
    second_request.total_tokens = 60
    second_request.model_calls = 1
    store.add_usage(user.user_id, "conversation-1", second_request)
    store.add_usage(user.user_id, "conversation-2", second_request)

    rows = store.list_all()
    first_conversation = next(row for row in rows if row.conversation_id == "conversation-1")
    second_conversation = next(row for row in rows if row.conversation_id == "conversation-2")

    assert len(rows) == 2
    assert first_conversation.input_tokens == 150
    assert first_conversation.output_tokens == 35
    assert first_conversation.total_tokens == 185
    assert first_conversation.model_calls == 3
    assert second_conversation.total_tokens == 60

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM conversation_usage").fetchone()[0]
    assert count == 2


def test_usage_store_does_not_store_unreported_zero_usage(tmp_path):
    database_path = str(tmp_path / "usage.db")
    user = AuthStore(database_path).create_user("person@example.com", "StrongPassword1!")
    store = UsageStore(database_path)
    tracker = UsageTracker()
    tracker.model_calls = 1
    tracker.unreported_calls = 1

    store.add_usage(user.user_id, "conversation-1", tracker)

    usage = store.list_all()[0]
    assert usage.total_tokens == 0
    assert usage.unreported_calls == 1
