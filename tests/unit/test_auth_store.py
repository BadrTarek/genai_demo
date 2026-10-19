import sqlite3

import pytest

from etic_genai.auth.store import (
    MAX_FAILED_ATTEMPTS,
    AccountLockedError,
    AuthStore,
    InvalidCredentialsError,
)


def test_auth_store_hashes_password_and_expires_sessions_after_inactivity(tmp_path, monkeypatch):
    current_time = 1_000_000
    monkeypatch.setattr("etic_genai.auth.store.time.time", lambda: current_time)
    store = AuthStore(str(tmp_path / "auth.db"))
    user = store.create_user("Person@Example.com", "StrongPassword1!")

    with sqlite3.connect(store.database_path) as connection:
        stored_hash = connection.execute(
            "SELECT password_hash FROM users WHERE user_id = ?",
            (user.user_id,),
        ).fetchone()[0]
    assert "StrongPassword1!" not in stored_hash

    session = store.sign_in("person@example.COM", "StrongPassword1!")
    assert store.get_active_session_user(session.access_token) == user

    current_time += session.expires_in + 1
    assert store.get_active_session_user(session.access_token) is None


def test_auth_store_locks_account_after_five_failed_signins(tmp_path):
    store = AuthStore(str(tmp_path / "auth.db"))
    store.create_user("person@example.com", "StrongPassword1!")

    for _ in range(MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(InvalidCredentialsError):
            store.sign_in("person@example.com", "WrongPassword1!")

    with pytest.raises(AccountLockedError):
        store.sign_in("person@example.com", "WrongPassword1!")

    with pytest.raises(AccountLockedError):
        store.sign_in("person@example.com", "StrongPassword1!")


def test_existing_users_table_is_migrated_with_non_admin_default(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?)",
            ("legacy-user", "legacy@example.com", "old-hash", 1),
        )

    AuthStore(str(database_path))

    with sqlite3.connect(database_path) as connection:
        migrated_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(users)").fetchall()
        }
        is_admin = connection.execute(
            "SELECT is_admin FROM users WHERE user_id = ?",
            ("legacy-user",),
        ).fetchone()[0]

    assert "is_admin" in migrated_columns
    assert is_admin == 0


def test_admin_flag_is_loaded_from_database_for_authenticated_sessions(tmp_path):
    database_path = str(tmp_path / "auth.db")
    store = AuthStore(database_path)
    user = store.create_user("admin@example.com", "StrongPassword1!")
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE users SET is_admin = 1 WHERE user_id = ?", (user.user_id,))

    session = store.sign_in("admin@example.com", "StrongPassword1!")

    assert session.user.is_admin is True
    assert store.get_active_session_user(session.access_token).is_admin is True
