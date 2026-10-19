"""Graph state schemas. Extend `BaseAgentState` to add agent-specific fields."""

from typing import Any, NotRequired

from langgraph.graph import MessagesState


class BaseAgentState(MessagesState):
    """Conversation `messages` (with the add_messages reducer) plus free-form metadata."""

    metadata: dict[str, Any]
    retrieved_context: NotRequired[str]


class RAGState(BaseAgentState):
    """State of one guarded RAG turn. Everything except `messages` is reset each turn."""

    question: str
    contexts: list[dict[str, Any]]
    answer: str
    # ok | blocked | no_context | retry | ungrounded | error
    status: str
    attempts: int
    feedback: str
    guardrail_events: list[dict[str, Any]]
    # Guardrail scores, e.g. {"groundedness": {"score": 0.9, "passed": True, ...}}
    evaluations: dict[str, dict[str, Any]]
