"""SQLite persistence for cumulative token usage per user conversation."""

import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from etic_genai.usage.callback import UsageTracker


@dataclass(frozen=True)
class ConversationUsage:
    user_id: str
    email: str
    conversation_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_calls: int
    unreported_calls: int
    updated_at: int


class UsageStore:
    def __init__(self, database_path: str) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_usage (
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    conversation_id TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    model_calls INTEGER NOT NULL DEFAULT 0,
                    unreported_calls INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (user_id, conversation_id)
                )
                """
            )

    def add_usage(
        self,
        user_id: str,
        conversation_id: str,
        tracker: UsageTracker,
    ) -> None:
        if tracker.model_calls == 0:
            return
        now = int(time.time())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_usage (
                    user_id, conversation_id, input_tokens, output_tokens,
                    total_tokens, model_calls, unreported_calls, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                    input_tokens = conversation_usage.input_tokens + excluded.input_tokens,
                    output_tokens = conversation_usage.output_tokens + excluded.output_tokens,
                    total_tokens = conversation_usage.total_tokens + excluded.total_tokens,
                    model_calls = conversation_usage.model_calls + excluded.model_calls,
                    unreported_calls = conversation_usage.unreported_calls
                        + excluded.unreported_calls,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    conversation_id,
                    tracker.input_tokens,
                    tracker.output_tokens,
                    tracker.total_tokens,
                    tracker.model_calls,
                    tracker.unreported_calls,
                    now,
                ),
            )

    def list_all(self) -> list[ConversationUsage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    usage.user_id,
                    users.email,
                    usage.conversation_id,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.model_calls,
                    usage.unreported_calls,
                    usage.updated_at
                FROM conversation_usage AS usage
                JOIN users ON users.user_id = usage.user_id
                ORDER BY usage.updated_at DESC, users.email, usage.conversation_id
                """
            ).fetchall()
        return [ConversationUsage(**dict(row)) for row in rows]
