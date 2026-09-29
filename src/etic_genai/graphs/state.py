"""Graph state schemas. Extend `BaseAgentState` to add agent-specific fields."""

from typing import Any

from langgraph.graph import MessagesState


class BaseAgentState(MessagesState):
    """Conversation `messages` (with the add_messages reducer) plus free-form metadata."""

    metadata: dict[str, Any]
