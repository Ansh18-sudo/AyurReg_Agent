"""SQLite access: one cached connection per thread, indexed schema, versioned migrations."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime

import config as cfg

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(cfg.DB_PATH), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS users(
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS conversations(
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        summarized_upto INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_conv_user_updated ON conversations(user_id, updated_at DESC);
    CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','assistant')),
        content TEXT NOT NULL,
        meta TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
    CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id);
    CREATE TABLE IF NOT EXISTS memories(
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        conversation_id TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
    CREATE TABLE IF NOT EXISTS preferences(
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY(user_id, key)
    );
    """,
    # Future schema changes: add 2: "ALTER TABLE ...;" here. Existing data is never dropped.
}


def init_db() -> None:
    """Create or upgrade the database. Safe to call on every start; never deletes data."""
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = get_conn()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for v in sorted(MIGRATIONS):
            if v > version:
                conn.executescript(MIGRATIONS[v] + f"\nPRAGMA user_version={v};")
        conn.commit()
        _initialised = True
