"""SQLite cache for Jev responses and embedding vectors."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np


def request_key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT)")
            self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
            self._db.commit()

    def get_response(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT body FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_response(self, key: str, body: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(body))
            )
            self._db.commit()  # WAL with synchronous=NORMAL: a commit is an append, not an fsync

    def get_vector(self, key: str) -> np.ndarray | None:
        with self._lock:
            row = self._db.execute("SELECT vec FROM vectors WHERE key=?", (key,)).fetchone()
        return np.frombuffer(row[0], dtype=np.float32) if row else None

    def put_vector(self, key: str, vec: np.ndarray) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                (key, np.asarray(vec, dtype=np.float32).tobytes()),
            )
            self._db.commit()

    def flush(self) -> None:
        with self._lock:
            self._db.commit()

    def close(self) -> None:
        self.flush()
        self._db.close()
