"""Versioned schema migrations."""
from __future__ import annotations
import sqlite3
from semanticsd.db import schema


def _current_version(conn) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def _safe_run(conn, stmt: str) -> None:
    """Run a statement; swallow 'duplicate column'/'already exists' on re-runs."""
    try:
        conn.execute(stmt)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "duplicate column" in msg or "already exists" in msg:
            return
        raise


def apply(conn) -> None:
    """Apply all pending migrations. Idempotent."""
    current = _current_version(conn)
    if current >= schema.SCHEMA_VERSION:
        return
    if current < 1:
        for stmt in schema.DDL_V1:
            conn.execute(stmt)
    if current < 2:
        for stmt in schema.DDL_V2:
            conn.execute(stmt)
    if current < 3:
        for stmt in schema.DDL_V3:
            _safe_run(conn, stmt)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(schema.SCHEMA_VERSION),),
    )
