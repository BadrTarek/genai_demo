import sqlite3
from pathlib import Path

from etic_genai.config.settings import get_settings


def _db_path() -> Path:
    settings = get_settings()
    return Path(settings.microsoft_auth_db_path) / "microsoft_auth.db"


def _connect() -> sqlite3.Connection:
    db_path = _db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn= sqlite3.connect(db_path)
    conn.row_factory=sqlite3.Row
    return conn



def _init_db() -> None:
    """Create the SQLite schema needed to persist Microsoft auth tokens."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_token_cache (
                user_id TEXT PRIMARY KEY,
                cache_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_identity (
                user_id TEXT PRIMARY KEY,
                account TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_device_flow (
                user_id TEXT PRIMARY KEY,
                flow_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pending_device_flow_created_at
            ON pending_device_flow (created_at)
            """
        )


def get_user_cache(user_id: str) -> str | None:
    """Retrieve the token cache JSON for a given user ID."""
    _init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT cache_json FROM user_token_cache WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["cache_json"] if row else None


def upsert_user_cache(user_id: str, cache_json: str) -> None:
    """Insert or update the token cache JSON for a given user ID."""
    _init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_token_cache (user_id, cache_json)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                cache_json = excluded.cache_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, cache_json),
        )

def save_pending_flow(user_id: str, flow_json: str) -> None:
    """Save a pending device flow for a given user ID."""
    _init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pending_device_flow (user_id, flow_json)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                flow_json = excluded.flow_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (user_id, flow_json),
        )


def get_pending_flow(user_id: str) -> str | None:
    """Retrieve the pending device flow JSON for a given user ID."""
    _init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT flow_json FROM pending_device_flow WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["flow_json"] if row else None


def clear_pending_flow(user_id: str) -> None:
    """Delete the pending device flow for a given user ID."""
    _init_db()
    with _connect() as conn:
        conn.execute(
            "DELETE FROM pending_device_flow WHERE user_id = ?", (user_id,)
        )


def upsert_user_identity(user_id: str, account: str, tenant_id: str) -> None:
    """Insert or update the user identity for a given user ID."""
    _init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_identity (user_id, account, tenant_id)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                account = excluded.account,
                tenant_id = excluded.tenant_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, account, tenant_id),
        )
def get_user_identity(user_id: str) -> dict[str, str] | None:
    """Retrieve the user identity for a given user ID."""
    _init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT account, tenant_id FROM user_identity WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return {"account": row["account"], "tenant_id": row["tenant_id"]}
        return None
