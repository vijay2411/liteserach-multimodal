"""SQLite connection factory. Uses stdlib sqlite3 (Python must be built with
--enable-loadable-sqlite-extensions; see the plan's tech-stack note) and loads
the sqlite-vec extension on every connection."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import sqlite_vec


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
