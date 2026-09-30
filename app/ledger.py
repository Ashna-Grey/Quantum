from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock


class Ledger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS key_registry (
                    key_id TEXT PRIMARY KEY,
                    authorized_verifiers TEXT NOT NULL,
                    copies_issued INTEGER NOT NULL DEFAULT 0,
                    used INTEGER NOT NULL DEFAULT 0
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def register_key(self, key_id: str, authorized_verifiers: str = "verifier-1"):
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO key_registry(key_id, authorized_verifiers) VALUES (?, ?)",
                (key_id, authorized_verifiers),
            )

    def issue_copy(self, key_id: str):
        with self._lock, self._conn() as c:
            c.execute("UPDATE key_registry SET copies_issued = copies_issued + 1 WHERE key_id = ?", (key_id,))

    def get(self, key_id: str):
        with self._conn() as c:
            row = c.execute("SELECT * FROM key_registry WHERE key_id = ?", (key_id,)).fetchone()
            return dict(row) if row else None

    def authorize(self, key_id: str, verifier_id: str) -> bool:
        row = self.get(key_id)
        if not row:
            return False
        allowed = {v.strip() for v in row["authorized_verifiers"].split(",") if v.strip()}
        return verifier_id in allowed

    def consume_session(self, session_id: str, key_id: str) -> bool:
        with self._lock, self._conn() as c:
            existing = c.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if existing:
                return False
            c.execute(
                "INSERT INTO sessions(session_id, key_id, created_at) VALUES (?, ?, datetime('now'))",
                (session_id, key_id),
            )
            c.execute("UPDATE key_registry SET used = used + 1 WHERE key_id = ?", (key_id,))
            return True
