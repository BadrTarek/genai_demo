from langchain_core.messages import AIMessage

from etic_genai.mcp.server import ask_assistant, call_registered_tool


def test_mcp_tool_adapter_calls_registered_tool():
    result = call_registered_tool("current_datetime", {"timezone": "UTC"})
    assert result.endswith("+00:00")


def test_mcp_assistant_adapter_returns_latest_message(fake_llm_factory, monkeypatch):
    from etic_genai.mcp import server

    class FakeAssistantAgent:
        def __init__(self) -> None:
            self.llm = fake_llm_factory("ok")

        def invoke(self, message: str, thread_id: str = "mcp") -> dict:
            return {"messages": [AIMessage(content=f"{thread_id}: {message}: ok")]}

    monkeypatch.setattr(server, "AssistantAgent", FakeAssistantAgent)

    assert ask_assistant("hello", thread_id="test") == "test: hello: ok"
