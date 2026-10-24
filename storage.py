import hashlib
import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path(__file__).with_name("tasks.db")
DB_LOCK = threading.RLock()


def canonical_json(value):
    """Serialize JSON with consistent key ordering."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def message_hash(message):
    """Hash the message, excluding the separate configuration."""
    return hashlib.sha256(
        canonical_json(message).encode("utf-8")
    ).hexdigest()


def connect():
    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=15,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                status TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS message_dedup (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id),
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            )
        """)


def get_existing_message(principal, message_id):
    """Return the saved hash and task for a previous message."""
    with connect() as connection:
        row = connection.execute("""
            SELECT d.message_hash, t.task_json
            FROM message_dedup AS d
            JOIN tasks AS t ON t.task_id = d.task_id
            WHERE d.principal = ? AND d.message_id = ?
        """, (principal, message_id)).fetchone()

    if row is None:
        return None

    return {
        "message_hash": row["message_hash"],
        "task": json.loads(row["task_json"]),
    }


def create_task_once(
    principal,
    message_id,
    digest,
    task_id,
    context_id,
    batch_id,
    status,
    task,
):
    """Atomically create a task and register its message ID."""
    task_json = canonical_json(task)

    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        existing = connection.execute("""
            SELECT d.message_hash, t.task_json
            FROM message_dedup AS d
            JOIN tasks AS t ON t.task_id = d.task_id
            WHERE d.principal = ? AND d.message_id = ?
        """, (principal, message_id)).fetchone()

        if existing is not None:
            connection.rollback()

            if existing["message_hash"] != digest:
                return {
                    "kind": "conflict",
                }

            return {
                "kind": "replay",
                "task": json.loads(existing["task_json"]),
            }

        connection.execute("""
            INSERT INTO tasks (
                task_id, principal, context_id,
                batch_id, status, task_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            task_id,
            principal,
            context_id,
            batch_id,
            status,
            task_json,
        ))

        connection.execute("""
            INSERT INTO message_dedup (
                principal, message_id, message_hash, task_id
            )
            VALUES (?, ?, ?, ?)
        """, (principal, message_id, digest, task_id))

        connection.commit()

    return {
        "kind": "created",
        "task": task,
    }
def save_task(task_id, principal, context_id, batch_id, status, task):
    """Save an existing task only if its owner matches."""
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        cursor = connection.execute("""
            UPDATE tasks
            SET context_id = ?,
                batch_id = ?,
                status = ?,
                task_json = ?
            WHERE task_id = ? AND principal = ?
        """, (
            context_id,
            batch_id,
            status,
            canonical_json(task),
            task_id,
            principal,
        ))

        if cursor.rowcount != 1:
            connection.rollback()
            return False

        connection.commit()
        return True


def get_task(task_id, principal):
    with connect() as connection:
        row = connection.execute("""
            SELECT task_json
            FROM tasks
            WHERE task_id = ? AND principal = ?
        """, (task_id, principal)).fetchone()

    return json.loads(row["task_json"]) if row else None


def list_tasks(principal):
    with connect() as connection:
        rows = connection.execute("""
            SELECT task_json
            FROM tasks
            WHERE principal = ?
            ORDER BY created_at DESC, task_id
        """, (principal,)).fetchall()

    return [json.loads(row["task_json"]) for row in rows]


initialize_database()
