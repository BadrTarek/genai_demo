"""Graph state schemas. Extend `BaseAgentState` to add agent-specific fields."""

from typing import Any

from langgraph.graph import MessagesState


class BaseAgentState(MessagesState):
    """Conversation `messages` (with the add_messages reducer) plus free-form metadata."""

    metadata: dict[str, Any]


class RAGState(BaseAgentState):
    """State of one RAG turn. Everything except `messages` is reset at the start of a turn."""

    question: str
    contexts: list[dict[str, Any]]
    answer: str
    # ok | blocked | no_context | retry | ungrounded
    status: str
    attempts: int
    feedback: str
    guardrail_events: list[dict[str, Any]]
    evaluations: dict[str, dict[str, Any]]
