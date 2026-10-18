"""SQLite persistence for users and expiring bearer-token sessions."""

import hashlib
import hmac
import secrets
import sqlite3
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SESSION_INACTIVITY_SECONDS = 30 * 60
FAILED_ATTEMPT_WINDOW_SECONDS = 15 * 60
ACCOUNT_LOCK_SECONDS = 30 * 60
MAX_FAILED_ATTEMPTS = 5
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LENGTH = 64
_DUMMY_SALT = b"etic-genai-auth-dummy"
_DUMMY_HASH = hashlib.scrypt(
    b"not-a-user-password",
    salt=_DUMMY_SALT,
    n=SCRYPT_N,
    r=SCRYPT_R,
    p=SCRYPT_P,
    dklen=SCRYPT_KEY_LENGTH,
)


class DuplicateEmailError(Exception):
    """Raised when a case-insensitive duplicate email is registered."""


class InvalidCredentialsError(Exception):
    """Raised when sign-in credentials are invalid."""


class AccountLockedError(Exception):
    """Raised when sign-in is attempted during an account lockout."""


@dataclass(frozen=True)
class AuthUser:
    user_id: str
    email: str
    is_admin: bool = False


@dataclass(frozen=True)
class AuthSession:
    user: AuthUser
    access_token: str
    expires_in: int


class AuthStore:
    def __init__(self, database_path: str) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
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
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS auth_failures (
                    email TEXT PRIMARY KEY COLLATE NOCASE,
                    failed_attempts INTEGER NOT NULL,
                    window_started_at INTEGER NOT NULL,
                    locked_until INTEGER
                );
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    created_at INTEGER NOT NULL,
                    last_seen_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS user_sessions_user_id_idx
                    ON user_sessions(user_id);
                """
            )
            user_columns = {row["name"] for row in connection.execute("PRAGMA table_info(users)")}
            if "is_admin" not in user_columns:
                connection.execute(
                    "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().casefold()

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        password_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=SCRYPT_KEY_LENGTH,
        )
        return f"scrypt${salt.hex()}${password_hash.hex()}"

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        try:
            algorithm, salt_hex, expected_hex = stored_hash.split("$", maxsplit=2)
            if algorithm != "scrypt":
                return False
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(expected_hex)
        except (ValueError, TypeError):
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=SCRYPT_KEY_LENGTH,
        )
        return hmac.compare_digest(actual, expected)

    def create_user(self, email: str, password: str) -> AuthUser:
        normalized_email = self.normalize_email(email)
        password_hash = self._hash_password(password)

        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users (user_id, email, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (str(uuid.uuid4()), normalized_email, password_hash, int(time.time())),
                )
                user_id = connection.execute(
                    "SELECT user_id FROM users WHERE email = ?",
                    (normalized_email,),
                ).fetchone()["user_id"]
                connection.execute("DELETE FROM auth_failures WHERE email = ?", (normalized_email,))
        except sqlite3.IntegrityError as exc:
            if "users.email" in str(exc).lower() or "unique constraint" in str(exc).lower():
                raise DuplicateEmailError from exc
            raise

        return AuthUser(user_id=user_id, email=normalized_email)

    def sign_in(self, email: str, password: str) -> AuthSession:
        normalized_email = self.normalize_email(email)
        now = int(time.time())
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        auth_error: type[Exception] | None = None
        user: AuthUser | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            lockout = connection.execute(
                "SELECT failed_attempts, window_started_at, locked_until "
                "FROM auth_failures WHERE email = ?",
                (normalized_email,),
            ).fetchone()
            if lockout and lockout["locked_until"] and lockout["locked_until"] > now:
                auth_error = AccountLockedError
            else:
                row = connection.execute(
                    "SELECT user_id, email, password_hash, is_admin FROM users WHERE email = ?",
                    (normalized_email,),
                ).fetchone()

                stored_hash = row["password_hash"] if row else None
                verified = (
                    self._verify_password(password, stored_hash)
                    if stored_hash
                    else self._verify_dummy_password(password)
                )
                if row is None or not verified:
                    locked = self._record_failed_attempt(connection, normalized_email, lockout, now)
                    auth_error = AccountLockedError if locked else InvalidCredentialsError
                else:
                    connection.execute(
                        "DELETE FROM auth_failures WHERE email = ?", (normalized_email,)
                    )
                    expires_at = now + SESSION_INACTIVITY_SECONDS
                    connection.execute(
                        "INSERT INTO user_sessions "
                        "(token_hash, user_id, created_at, last_seen_at, expires_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (token_hash, row["user_id"], now, now, expires_at),
                    )
                    user = AuthUser(
                        user_id=row["user_id"],
                        email=row["email"],
                        is_admin=bool(row["is_admin"]),
                    )

        if auth_error is AccountLockedError:
            raise AccountLockedError
        if auth_error is InvalidCredentialsError:
            raise InvalidCredentialsError
        if user is None:
            raise RuntimeError("Authentication completed without a user or an error.")

        return AuthSession(user=user, access_token=token, expires_in=SESSION_INACTIVITY_SECONDS)

    @staticmethod
    def _verify_dummy_password(password: str) -> bool:
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_DUMMY_SALT,
            n=SCRYPT_N,
            r=SCRYPT_R,
            p=SCRYPT_P,
            dklen=SCRYPT_KEY_LENGTH,
        )
        return hmac.compare_digest(candidate, _DUMMY_HASH)

    @staticmethod
    def _record_failed_attempt(
        connection: sqlite3.Connection,
        email: str,
        existing: sqlite3.Row | None,
        now: int,
    ) -> bool:
        if existing is None or now - existing["window_started_at"] >= FAILED_ATTEMPT_WINDOW_SECONDS:
            failures = 1
            window_started_at = now
        else:
            failures = existing["failed_attempts"] + 1
            window_started_at = existing["window_started_at"]

        locked_until = now + ACCOUNT_LOCK_SECONDS if failures >= MAX_FAILED_ATTEMPTS else None
        connection.execute(
            "INSERT INTO auth_failures (email, failed_attempts, window_started_at, locked_until) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(email) DO UPDATE SET "
            "failed_attempts=excluded.failed_attempts, "
            "window_started_at=excluded.window_started_at, "
            "locked_until=excluded.locked_until",
            (email, failures, window_started_at, locked_until),
        )
        return locked_until is not None

    def get_active_session_user(self, access_token: str) -> AuthUser | None:
        token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()
        now = int(time.time())
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            row = connection.execute(
                "SELECT users.user_id, users.email, users.is_admin, user_sessions.expires_at "
                "FROM user_sessions "
                "JOIN users ON users.user_id = user_sessions.user_id "
                "WHERE user_sessions.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute(
                    "DELETE FROM user_sessions WHERE token_hash = ?",
                    (token_hash,),
                )
                return None

            connection.execute(
                "UPDATE user_sessions SET last_seen_at = ?, expires_at = ? WHERE token_hash = ?",
                (now, now + SESSION_INACTIVITY_SECONDS, token_hash),
            )
            return AuthUser(
                user_id=row["user_id"],
                email=row["email"],
                is_admin=bool(row["is_admin"]),
            )
