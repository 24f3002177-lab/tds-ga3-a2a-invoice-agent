import hashlib
import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(file).with_name("tasks.db")
DB_LOCK = threading.RLock()


def canonical_json(value):
    """Serialize JSON consistently, regardless of key order."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def message_hash(message):
    """Hash the message only; ignore the separate configuration."""
    encoded = canonical_json(message).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def connect():
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=10,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database():
    """Create the database tables if they do not exist."""
    with DB_LOCK:
        with connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    principal TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    task_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (principal, task_id)
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS message_dedup (
                    principal TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    PRIMARY KEY (principal, message_id)
                )
            """)


def save_task(task_id, principal, context_id, batch_id, status, task):
    """Insert a task, or update its saved representation."""
    task_json = canonical_json(task)

    with DB_LOCK:
        with connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, principal, context_id,
                    batch_id, status, task_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    task_json = excluded.task_json
                WHERE tasks.principal = excluded.principal
                """,
                (
                    task_id,
                    principal,
                    context_id,
                    batch_id,
                    status,
                    task_json,
                ),
            )


def get_task(task_id, principal):
    """Return a task only when it belongs to the given principal."""
    with DB_LOCK:
        with connect() as connection:
            row = connection.execute(
                """
                SELECT task_json
                FROM tasks
                WHERE task_id = ? AND principal = ?
                """,
                (task_id, principal),
            ).fetchone()

    return json.loads(row["task_json"]) if row else None


def list_tasks(principal):
    """List only the tasks owned by this principal."""
    with DB_LOCK:
        with connect() as connection:
            rows = connection.execute(
                """
                SELECT task_json
                FROM tasks
                WHERE principal = ?
                ORDER BY created_at DESC, task_id
                """,
                (principal,),
            ).fetchall()

    return [json.loads(row["task_json"]) for row in rows]


initialize_database()
