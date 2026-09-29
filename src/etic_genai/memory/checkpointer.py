"""Checkpointers persist graph state between turns (conversation memory).

In-memory is fine for development. Swap in a persistent saver
(e.g. langgraph-checkpoint-postgres / -sqlite) for production.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


def get_checkpointer() -> BaseCheckpointSaver:
    return InMemorySaver()
