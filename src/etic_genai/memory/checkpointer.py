"""Checkpointers persist graph state between turns (conversation memory).

In-memory is fine for development. Swap in a persistent saver
(e.g. langgraph-checkpoint-postgres / -sqlite) for production.
"""

import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


def get_checkpointer() -> BaseCheckpointSaver:
    return InMemorySaver()


@lru_cache
def get_sqlite_checkpointer(database_path: str) -> BaseCheckpointSaver:
    """Return a process-wide SQLite saver for persistent graph state."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False, timeout=30)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer
