from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from mlforge.errors import ForgeError


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid() -> str:
    return uuid.uuid4().hex


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ForgeError("Invalid resource identifier", "not_found", 404)
    return value


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Store:
    def __init__(self, home: Path):
        self.home = home
        home.mkdir(parents=True, exist_ok=True)
        self.path = home / "metadata.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                  kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                  created_at TEXT NOT NULL, PRIMARY KEY(kind,id));
                CREATE INDEX IF NOT EXISTS record_recent ON records(kind,created_at);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, kind: str, key: str) -> dict:
        identifier(key)
        with self.connect() as db:
            row = db.execute(
                "SELECT body FROM records WHERE kind=? AND id=?", (kind, key)
            ).fetchone()
        if row is None:
            raise ForgeError(f"{kind.capitalize()} not found", "not_found", 404)
        return json.loads(row[0])

    def list(self, kind: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?",
                (kind, min(limit, 1000), offset),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def save(self, kind: str, record: dict) -> dict:
        identifier(record["id"])
        with self.connect() as db:
            db.execute(
                "INSERT INTO records VALUES(?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body",
                (kind, record["id"], dumps(record), record["created_at"]),
            )
        return record

    def patch(self, kind: str, key: str, **changes: Any) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT body FROM records WHERE kind=? AND id=?", (kind, identifier(key))
            ).fetchone()
            if row is None:
                raise ForgeError("Resource not found", "not_found", 404)
            record = json.loads(row[0])
            record.update(changes)
            db.execute(
                "UPDATE records SET body=? WHERE kind=? AND id=?", (dumps(record), kind, key)
            )
        return record

    def remove(self, kind: str, key: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier(key)))

    def reserve_experiment(self, record: dict) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT body FROM records WHERE kind='experiment'"):
                if json.loads(row[0])["status"] in ("queued", "running"):
                    raise ForgeError(
                        "Another experiment is active. Wait or cancel it first.", "busy", 409
                    )
            db.execute(
                "INSERT INTO records VALUES('experiment',?,?,?)",
                (record["id"], dumps(record), record["created_at"]),
            )

    def highest_model_version(self, name: str) -> int:
        with self.connect() as db:
            row = db.execute(
                "SELECT max(json_extract(body,'$.version')) FROM records WHERE kind='model' AND json_extract(body,'$.name')=?",
                (name,),
            ).fetchone()
        return int(row[0] or 0)

    def overview(self) -> dict:
        with self.connect() as db:
            counts = {
                r[0]: r[1] for r in db.execute("SELECT kind,count(*) FROM records GROUP BY kind")
            }
            statuses = [
                json.loads(r[0])["status"]
                for r in db.execute("SELECT body FROM records WHERE kind='experiment'")
            ]
            model_count = sum(
                json.loads(r[0])["status"] == "registered"
                for r in db.execute("SELECT body FROM records WHERE kind='model'")
            )
        return {
            "datasets": counts.get("dataset", 0),
            "experiments": len(statuses),
            "succeeded": statuses.count("succeeded"),
            "failed": statuses.count("failed"),
            "active": sum(s in ("running", "queued") for s in statuses),
            "models": model_count,
            "requests_retained": counts.get("prediction", 0),
        }

    def retain_logs(self, count: int, days: int) -> None:
        with self.connect() as db:
            db.execute(
                "DELETE FROM records WHERE kind='prediction' AND (julianday(created_at) < julianday('now')-? OR id NOT IN (SELECT id FROM records WHERE kind='prediction' ORDER BY created_at DESC LIMIT ?))",
                (days, count),
            )
