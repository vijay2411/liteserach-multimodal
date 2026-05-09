# SemanticsD — Plan 1: Foundation & Daemon Shell

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap a runnable SemanticsD daemon: project scaffold, config + paths + Keychain modules, SQLite schema (excluding vec0), FastAPI app with `/v1/health` behind bearer auth, `ssearch` CLI with `--status`, `semanticsd install/uninstall/serve` admin commands, `requirements.txt` + `install.sh` + `Makefile`. End state: `make install` produces a launchd-managed daemon that `curl` and `ssearch --status` can both talk to.

**Architecture:** Python 3.11+ package `semanticsd` with subpackages for `db/`, `server/`, `admin/`. CLI built with Typer. HTTP server is FastAPI + uvicorn. Bearer token in macOS Keychain via `keyring`. SQLite migrations run on startup; `sqlite-vec` and vector tables come in Plan 2. No file-watching, no embedders, no search engine yet — those are subsequent plans.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, Typer, pydantic v2, keyring, tomli (or stdlib `tomllib`), sqlite3 (stdlib), pytest, httpx (test client).

**Spec reference:** `/Users/vedantvijay/dev/side_projects/SemanticSearch/docs/superpowers/specs/2026-05-09-semanticsd-design.md`

---

## File Structure for This Plan

```
SemanticSearch/
├── semanticsd/                     # python package
│   ├── __init__.py                 # __version__
│   ├── __main__.py                 # `python -m semanticsd` → CLI
│   ├── cli.py                      # typer app: both `ssearch` + `semanticsd` entry points
│   ├── config.py                   # TOML loader, dataclass model
│   ├── paths.py                    # macOS-conventional paths
│   ├── keychain.py                 # `keyring` wrapper for tokens + API keys
│   ├── logging_setup.py            # structured log config
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.py               # versioned DDL (Plan 1 subset; vec0 added in Plan 2)
│   │   ├── migrations.py           # apply_migrations(conn)
│   │   └── connection.py           # get_connection(path)
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app factory
│   │   ├── auth.py                 # bearer-token dependency
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── health.py           # /v1/health
│   └── admin/
│       ├── __init__.py
│       ├── install.py              # install / uninstall / token print
│       └── launchd.py              # plist renderer
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # fixtures: tmp config dir, in-memory db, test client
│   ├── test_config.py
│   ├── test_paths.py
│   ├── test_keychain.py
│   ├── test_db.py
│   ├── test_auth.py
│   ├── test_health.py
│   └── test_launchd.py
├── scripts/
│   └── install.sh
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
└── README.md
```

---

## Task 1: Repo init + .gitignore + README stub

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `pyproject.toml`

- [ ] **Step 1: Initialize git repo and create .gitignore**

```bash
cd /Users/vedantvijay/dev/side_projects/SemanticSearch
git init
```

Create `.gitignore`:
```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
venv/
build/
dist/
.coverage
htmlcov/
*.log
.DS_Store
.env
.env.local
node_modules/
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "semanticsd"
version = "0.1.0"
description = "SemanticsD — local semantic search daemon for macOS"
requires-python = ">=3.11"
authors = [{ name = "Vedant Vijay" }]
readme = "README.md"

[project.scripts]
semanticsd = "semanticsd.cli:semanticsd_app"
ssearch = "semanticsd.cli:ssearch_app"

[tool.setuptools.packages.find]
include = ["semanticsd*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 3: Create README stub**

```markdown
# SemanticsD

Local semantic search daemon for macOS.

See `docs/superpowers/specs/2026-05-09-semanticsd-design.md` for the design.

## Quick start

    ./scripts/install.sh
    ssearch --status
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore README.md pyproject.toml
git commit -m "chore: project scaffold"
```

---

## Task 2: requirements.txt + Makefile

**Files:**
- Create: `requirements.txt`
- Create: `Makefile`

- [ ] **Step 1: Create requirements.txt with Plan-1-only deps (more added later)**

```
# Plan 1 (foundation)
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.6
typer>=0.12
keyring>=24.0

# Tests
pytest>=8.0
pytest-asyncio>=0.23
httpx>=0.27
```

(Plan 2-6 will add: `sqlite-vec`, `watchdog`, `sentence-transformers`, `torch`, `open-clip-torch`, `pillow`, `openai`, `voyageai`, `cohere`, `google-generativeai`, `google-cloud-aiplatform`, `boto3`, `tiktoken`, `pypdf`, `python-docx`, `openpyxl`, `xlrd`, `python-pptx`, `beautifulsoup4`, `ebooklib`, `striprtf`, `pytesseract`, `faster-whisper`, `pyobjc-framework-IOKit`.)

- [ ] **Step 2: Create Makefile**

```make
.PHONY: install dev test clean uninstall

PY := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev: install
	$(PYTHON) -m semanticsd serve

test: install
	$(PYTEST) -v

clean:
	rm -rf $(VENV) build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

uninstall:
	bash scripts/install.sh --uninstall
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt Makefile
git commit -m "chore: requirements.txt and Makefile"
```

---

## Task 3: Package skeleton

**Files:**
- Create: `semanticsd/__init__.py`
- Create: `semanticsd/__main__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create package init**

`semanticsd/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 2: Create `__main__.py` so `python -m semanticsd` works**

```python
from semanticsd.cli import semanticsd_app

if __name__ == "__main__":
    semanticsd_app()
```

- [ ] **Step 3: Create empty tests/__init__.py and a conftest stub**

`tests/__init__.py`: empty file.

`tests/conftest.py`:
```python
import os
import tempfile
from pathlib import Path
import pytest


@pytest.fixture
def tmp_app_support(monkeypatch) -> Path:
    """Redirect SemanticsD's Application Support dir to a tmp path for the test."""
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("SEMANTICSD_HOME", d)
        yield Path(d)
```

- [ ] **Step 4: Commit**

```bash
git add semanticsd/__init__.py semanticsd/__main__.py tests/__init__.py tests/conftest.py
git commit -m "feat: package skeleton + test conftest"
```

---

## Task 4: Paths module (TDD)

**Files:**
- Create: `tests/test_paths.py`
- Create: `semanticsd/paths.py`

- [ ] **Step 1: Write failing tests**

`tests/test_paths.py`:
```python
from pathlib import Path
from semanticsd import paths


def test_app_support_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    p = paths.app_support()
    assert p == Path.home() / "Library" / "Application Support" / "semanticsd"


def test_app_support_override(tmp_app_support):
    assert paths.app_support() == tmp_app_support


def test_logs_dir(monkeypatch):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    assert paths.logs_dir() == Path.home() / "Library" / "Logs" / "semanticsd"


def test_db_path_under_app_support(tmp_app_support):
    assert paths.db_path() == tmp_app_support / "index.db"


def test_config_path(tmp_app_support):
    assert paths.config_path() == tmp_app_support / "config.toml"


def test_launch_agent_plist_path(monkeypatch):
    monkeypatch.delenv("SEMANTICSD_HOME", raising=False)
    assert paths.launch_agent_plist() == Path.home() / "Library" / "LaunchAgents" / "com.semanticsd.plist"


def test_ensure_dirs_creates(tmp_app_support):
    paths.ensure_dirs()
    assert tmp_app_support.exists()
    assert (tmp_app_support / "models").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
.venv/bin/pytest tests/test_paths.py -v
```
Expected: ImportError or `module has no attribute`.

- [ ] **Step 3: Implement paths.py**

```python
"""macOS-conventional paths for SemanticsD."""
from __future__ import annotations
import os
from pathlib import Path


LAUNCHD_LABEL = "com.semanticsd"


def app_support() -> Path:
    """Application Support directory. Honors $SEMANTICSD_HOME for tests."""
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override)
    return Path.home() / "Library" / "Application Support" / "semanticsd"


def logs_dir() -> Path:
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override) / "logs"
    return Path.home() / "Library" / "Logs" / "semanticsd"


def db_path() -> Path:
    return app_support() / "index.db"


def config_path() -> Path:
    return app_support() / "config.toml"


def models_dir() -> Path:
    return app_support() / "models"


def launch_agent_plist() -> Path:
    override = os.environ.get("SEMANTICSD_HOME")
    if override:
        return Path(override) / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def ensure_dirs() -> None:
    """Create all directories the daemon needs. Idempotent."""
    for d in (app_support(), logs_dir(), models_dir(), launch_agent_plist().parent):
        d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 4: Run tests and verify pass**

```bash
.venv/bin/pytest tests/test_paths.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_paths.py semanticsd/paths.py
git commit -m "feat(paths): macOS-conventional paths module"
```

---

## Task 5: Config module (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `semanticsd/config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
from pathlib import Path
import textwrap
from semanticsd import config


def test_defaults_when_no_file(tmp_app_support):
    cfg = config.load()
    assert cfg.daemon.http_host == "127.0.0.1"
    assert cfg.daemon.http_port == 47600
    assert cfg.search.default_mode == "semantic"
    assert cfg.power.mode == "active"


def test_loads_overrides_from_file(tmp_app_support):
    (tmp_app_support / "config.toml").write_text(textwrap.dedent("""
        [daemon]
        http_port = 9999
        log_level = "debug"

        [search]
        default_mode = "filename"
    """))
    cfg = config.load()
    assert cfg.daemon.http_port == 9999
    assert cfg.daemon.log_level == "debug"
    assert cfg.search.default_mode == "filename"
    # Untouched defaults preserved:
    assert cfg.daemon.http_host == "127.0.0.1"


def test_invalid_mode_rejected(tmp_app_support):
    (tmp_app_support / "config.toml").write_text("[search]\ndefault_mode = 'bogus'\n")
    try:
        config.load()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_write_default(tmp_app_support):
    config.write_default()
    text = (tmp_app_support / "config.toml").read_text()
    assert "[daemon]" in text
    assert "http_port = 47600" in text
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement config.py**

```python
"""TOML config loader with defaults."""
from __future__ import annotations
import sys
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, field_validator
from semanticsd import paths

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class WatchConfig(BaseModel):
    directories: list[str] = ["~/"]
    ignore_patterns: list[str] = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
    max_file_size_mb: int = 50


class EmbeddingConfig(BaseModel):
    backend: str = "local"           # "local" | "openai" | provider id
    preset: str = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str = ""
    dimensions: int = 0
    batch_size: int = 128


class SearchConfig(BaseModel):
    default_mode: str = "semantic"
    max_results: int = 20

    @field_validator("default_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("semantic", "filename", "grep"):
            raise ValueError(f"default_mode must be semantic|filename|grep, got {v}")
        return v


class ChunkingConfig(BaseModel):
    strategy: str = "sliding"
    window_tokens: int = 512
    overlap_tokens: int = 64


class DaemonConfig(BaseModel):
    http_host: str = "127.0.0.1"
    http_port: int = 47600
    log_level: str = "info"


class PowerConfig(BaseModel):
    mode: str = "active"
    saver_reindex_interval: str = "1h"
    saver_pause_watcher: bool = True
    auto_saver_on_battery: bool = True


class IndexingConfig(BaseModel):
    max_attempts: int = 5
    worker_concurrency: int = 2


class Config(BaseModel):
    watch: WatchConfig = Field(default_factory=WatchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)


def load(path: Path | None = None) -> Config:
    """Load config, falling back to defaults if file missing or section absent."""
    p = path or paths.config_path()
    if not p.exists():
        return Config()
    raw = tomllib.loads(p.read_text())
    try:
        return Config(**raw)
    except ValidationError as e:
        raise ValueError(str(e)) from e


DEFAULT_TOML = """\
[watch]
directories = ["~/"]
ignore_patterns = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
max_file_size_mb = 50

[embedding]
backend = "local"
preset = "local"
model = "BAAI/bge-small-en-v1.5"
batch_size = 128

[search]
default_mode = "semantic"
max_results = 20

[chunking]
strategy = "sliding"
window_tokens = 512
overlap_tokens = 64

[daemon]
http_host = "127.0.0.1"
http_port = 47600
log_level = "info"

[power]
mode = "active"
saver_reindex_interval = "1h"
saver_pause_watcher = true
auto_saver_on_battery = true

[indexing]
max_attempts = 5
worker_concurrency = 2
"""


def write_default() -> Path:
    """Write a default config file if none exists. Returns the path."""
    paths.ensure_dirs()
    p = paths.config_path()
    if not p.exists():
        p.write_text(DEFAULT_TOML)
    return p
```

- [ ] **Step 4: Run tests, expect pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_config.py semanticsd/config.py
git commit -m "feat(config): TOML config loader with defaults"
```

---

## Task 6: Keychain module (TDD with mocked backend)

**Files:**
- Create: `tests/test_keychain.py`
- Create: `semanticsd/keychain.py`

- [ ] **Step 1: Write failing tests**

`tests/test_keychain.py`:
```python
import pytest
import keyring
from keyring.backends.fail import Keyring as FailKeyring
from keyring.backend import KeyringBackend
from semanticsd import keychain


class InMemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        self._store.pop((service, username), None)


@pytest.fixture(autouse=True)
def in_memory_keychain(monkeypatch):
    backend = InMemoryKeyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    yield backend


def test_set_and_get_token():
    keychain.set_auth_token("abc123")
    assert keychain.get_auth_token() == "abc123"


def test_get_token_missing_returns_none():
    assert keychain.get_auth_token() is None


def test_generate_or_get_creates_when_missing():
    tok = keychain.generate_or_get_auth_token()
    assert len(tok) >= 32
    # Stable on second call:
    assert keychain.generate_or_get_auth_token() == tok


def test_provider_api_key():
    keychain.set_provider_key("openai", "sk-xxx")
    assert keychain.get_provider_key("openai") == "sk-xxx"
    keychain.delete_provider_key("openai")
    assert keychain.get_provider_key("openai") is None
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_keychain.py -v
```

- [ ] **Step 3: Implement keychain.py**

```python
"""Keychain wrapper for the SemanticsD auth token and provider API keys."""
from __future__ import annotations
import secrets
import keyring

SERVICE = "semanticsd"
TOKEN_ACCOUNT = "api_token"


def set_auth_token(token: str) -> None:
    keyring.set_password(SERVICE, TOKEN_ACCOUNT, token)


def get_auth_token() -> str | None:
    return keyring.get_password(SERVICE, TOKEN_ACCOUNT)


def generate_or_get_auth_token() -> str:
    """Return existing token, or generate and store a new one."""
    existing = get_auth_token()
    if existing:
        return existing
    new_token = secrets.token_urlsafe(32)
    set_auth_token(new_token)
    return new_token


def delete_auth_token() -> None:
    try:
        keyring.delete_password(SERVICE, TOKEN_ACCOUNT)
    except keyring.errors.PasswordDeleteError:
        pass


def set_provider_key(provider_id: str, api_key: str) -> None:
    keyring.set_password(SERVICE, provider_id, api_key)


def get_provider_key(provider_id: str) -> str | None:
    return keyring.get_password(SERVICE, provider_id)


def delete_provider_key(provider_id: str) -> None:
    try:
        keyring.delete_password(SERVICE, provider_id)
    except keyring.errors.PasswordDeleteError:
        pass
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_keychain.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_keychain.py semanticsd/keychain.py
git commit -m "feat(keychain): wrapper for auth token + provider API keys"
```

---

## Task 7: SQLite schema + migrations (TDD; vec0 deferred to Plan 2)

**Files:**
- Create: `semanticsd/db/__init__.py`
- Create: `semanticsd/db/connection.py`
- Create: `semanticsd/db/schema.py`
- Create: `semanticsd/db/migrations.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write failing tests**

`tests/test_db.py`:
```python
import sqlite3
from semanticsd.db import connection, migrations, schema


def test_apply_migrations_creates_tables(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    for required in ("files", "chunks", "jobs", "usage", "meta", "embedding_meta"):
        assert required in tables, f"missing {required}"


def test_fts_virtual_tables_created(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'fts_%'")
    names = {r[0] for r in cur.fetchall()}
    assert "fts_chunks" in names
    assert "fts_paths" in names


def test_schema_version_recorded(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert v is not None
    assert int(v[0]) == schema.SCHEMA_VERSION


def test_apply_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    migrations.apply(conn)  # second call must not raise
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(v[0]) == schema.SCHEMA_VERSION


def test_wal_mode_enabled(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    assert mode == "wal"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_db.py -v
```

- [ ] **Step 3: Create db package init (empty)**

`semanticsd/db/__init__.py`: empty file.

- [ ] **Step 4: Implement db/connection.py**

```python
"""SQLite connection factory."""
from __future__ import annotations
import sqlite3
from pathlib import Path


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn
```

- [ ] **Step 5: Implement db/schema.py (Plan 1 subset; vec0 added in Plan 2 by bumping SCHEMA_VERSION)**

```python
"""Schema DDL. Plan 1 subset — no vector tables yet (those land in Plan 2)."""

SCHEMA_VERSION = 1

DDL_V1 = [
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE NOT NULL,
        modified_at INTEGER NOT NULL,
        size INTEGER NOT NULL,
        file_type TEXT NOT NULL,
        indexed_at INTEGER,
        last_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        byte_start INTEGER NOT NULL,
        byte_end INTEGER NOT NULL,
        UNIQUE(file_id, chunk_index)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)",
    """
    CREATE TABLE IF NOT EXISTS embedding_meta (
        chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        dim INTEGER NOT NULL,
        content_hash TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emb_meta_hash ON embedding_meta(content_hash, provider_id, model_id, dim)",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    """
    CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY,
        timestamp INTEGER NOT NULL,
        provider_id TEXT NOT NULL,
        model_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        input_tokens INTEGER NOT NULL,
        cost_usd REAL NOT NULL,
        chunk_count INTEGER NOT NULL,
        duration_ms INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_time ON usage(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(provider_id, model_id, timestamp)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        text, content='chunks', content_rowid='id'
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_paths USING fts5(
        path, content='files', content_rowid='id'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
]
```

- [ ] **Step 6: Implement db/migrations.py**

```python
"""Versioned schema migrations."""
from __future__ import annotations
import sqlite3
from semanticsd.db import schema


def _current_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def apply(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations. Idempotent."""
    current = _current_version(conn)
    if current >= schema.SCHEMA_VERSION:
        return
    if current < 1:
        for stmt in schema.DDL_V1:
            conn.execute(stmt)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(schema.SCHEMA_VERSION),),
    )
```

- [ ] **Step 7: Run tests, expect pass**

```bash
.venv/bin/pytest tests/test_db.py -v
```
Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add semanticsd/db/ tests/test_db.py
git commit -m "feat(db): SQLite schema + migrations (Plan 1 subset)"
```

---

## Task 8: Logging setup

**Files:**
- Create: `semanticsd/logging_setup.py`

- [ ] **Step 1: Implement logging_setup.py**

```python
"""Structured logging for the daemon."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from semanticsd import paths


def configure(level: str = "info", to_file: bool = True) -> None:
    """Configure root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Avoid duplicate handlers if called twice.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    root.addHandler(stderr)

    if to_file:
        paths.ensure_dirs()
        fh = logging.FileHandler(paths.logs_dir() / "semanticsd.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)
```

- [ ] **Step 2: Commit**

```bash
git add semanticsd/logging_setup.py
git commit -m "feat(logging): structured stderr+file logger"
```

---

## Task 9: Bearer-token auth dependency (TDD)

**Files:**
- Create: `semanticsd/server/__init__.py`
- Create: `semanticsd/server/auth.py`
- Create: `tests/test_auth.py`

- [ ] **Step 1: Write failing tests**

`tests/test_auth.py`:
```python
import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from semanticsd.server import auth


def make_app(token: str) -> FastAPI:
    app = FastAPI()
    auth._token_cache = token  # for test only

    @app.get("/protected", dependencies=[Depends(auth.require_token)])
    def protected():
        return {"ok": True}

    return app


def test_missing_header_returns_401(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_token_returns_401(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected", headers={"X-Auth-Token": "wrong"})
    assert r.status_code == 401


def test_correct_token_allows(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_auth.py -v
```

- [ ] **Step 3: Create server/__init__.py (empty) and implement auth.py**

`semanticsd/server/__init__.py`: empty file.

`semanticsd/server/auth.py`:
```python
"""Bearer-token auth dependency for FastAPI routes."""
from __future__ import annotations
from fastapi import Header, HTTPException, status
from semanticsd import keychain

_token_cache: str | None = None


def get_expected_token() -> str:
    """Lazy-load the token from Keychain on first use; cache thereafter."""
    global _token_cache
    if _token_cache is None:
        _token_cache = keychain.get_auth_token()
        if not _token_cache:
            raise RuntimeError(
                "No SemanticsD auth token in Keychain. Run `semanticsd install` first."
            )
    return _token_cache


def reload_token() -> None:
    """Force re-read from Keychain (used after rotation)."""
    global _token_cache
    _token_cache = None


def require_token(x_auth_token: str | None = Header(default=None)) -> None:
    expected = get_expected_token()
    if not x_auth_token or x_auth_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Auth-Token",
        )
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_auth.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/server/__init__.py semanticsd/server/auth.py tests/test_auth.py
git commit -m "feat(server): bearer-token auth dependency"
```

---

## Task 10: Health route + FastAPI app factory (TDD)

**Files:**
- Create: `semanticsd/server/routes/__init__.py`
- Create: `semanticsd/server/routes/health.py`
- Create: `semanticsd/server/app.py`
- Create: `tests/test_health.py`

- [ ] **Step 1: Write failing tests**

`tests/test_health.py`:
```python
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth


def test_health_unauthenticated_rejected(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 401


def test_health_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "doc_count" in body


def test_openapi_docs_available(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"]


def test_cors_allows_localhost(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app()
    client = TestClient(app)
    r = client.options(
        "/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-Auth-Token",
        },
    )
    assert r.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in r.headers}
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_health.py -v
```

- [ ] **Step 3: Create routes package**

`semanticsd/server/routes/__init__.py`: empty.

- [ ] **Step 4: Implement health route**

`semanticsd/server/routes/health.py`:
```python
"""Health endpoint."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd import __version__
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths

router = APIRouter()


@router.get("/health", dependencies=[Depends(require_token)])
def health() -> dict:
    db_ok = True
    doc_count = 0
    try:
        conn = connection.get_connection(paths.db_path())
        row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
        doc_count = int(row[0]) if row else 0
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "doc_count": doc_count,
        "vector_store": {"ok": db_ok},
        "embedder": {"ok": True, "message": "not configured (Plan 2)"},
    }
```

- [ ] **Step 5: Implement app factory**

`semanticsd/server/app.py`:
```python
"""FastAPI app factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from semanticsd import __version__
from semanticsd.server.routes import health


def create_app() -> FastAPI:
    app = FastAPI(
        title="SemanticsD",
        version=__version__,
        description="Local semantic search daemon for macOS.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/v1")
    return app
```

- [ ] **Step 6: Run tests, expect pass**

```bash
.venv/bin/pytest tests/test_health.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add semanticsd/server/app.py semanticsd/server/routes/ tests/test_health.py
git commit -m "feat(server): app factory + /v1/health route"
```

---

## Task 11: launchd plist renderer (TDD)

**Files:**
- Create: `semanticsd/admin/__init__.py`
- Create: `semanticsd/admin/launchd.py`
- Create: `tests/test_launchd.py`

- [ ] **Step 1: Write failing tests**

`tests/test_launchd.py`:
```python
import plistlib
from pathlib import Path
from semanticsd.admin import launchd


def test_render_returns_valid_plist(tmp_path):
    py = tmp_path / "python"
    py.write_text("#!/bin/sh\necho test\n")
    py.chmod(0o755)
    text = launchd.render_plist(python_executable=str(py), package_dir=str(tmp_path))
    parsed = plistlib.loads(text.encode())
    assert parsed["Label"] == "com.semanticsd"
    assert parsed["RunAtLoad"] is True
    assert parsed["KeepAlive"] is True
    assert parsed["ProgramArguments"][0] == str(py)
    assert parsed["ProgramArguments"][-2:] == ["-m", "semanticsd"]
    assert "serve" in parsed["ProgramArguments"]


def test_write_plist_to_path(tmp_path):
    plist_path = tmp_path / "com.semanticsd.plist"
    launchd.write_plist(plist_path, python_executable="/usr/bin/python3", package_dir=str(tmp_path))
    assert plist_path.exists()
    parsed = plistlib.loads(plist_path.read_bytes())
    assert parsed["Label"] == "com.semanticsd"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_launchd.py -v
```

- [ ] **Step 3: Implement admin/launchd.py**

`semanticsd/admin/__init__.py`: empty.

`semanticsd/admin/launchd.py`:
```python
"""Render and install the launchd plist for SemanticsD."""
from __future__ import annotations
import plistlib
from pathlib import Path
from semanticsd import paths


LABEL = paths.LAUNCHD_LABEL


def render_plist(python_executable: str, package_dir: str) -> str:
    paths.ensure_dirs()
    plist = {
        "Label": LABEL,
        "ProgramArguments": [python_executable, "-m", "semanticsd", "serve"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": package_dir,
        "StandardOutPath": str(paths.logs_dir() / "semanticsd.out.log"),
        "StandardErrorPath": str(paths.logs_dir() / "semanticsd.err.log"),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        },
        "ProcessType": "Background",
    }
    return plistlib.dumps(plist).decode()


def write_plist(target: Path, python_executable: str, package_dir: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_plist(python_executable, package_dir))
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_launchd.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/admin/ tests/test_launchd.py
git commit -m "feat(admin): launchd plist renderer"
```

---

## Task 12: Admin install/uninstall logic

**Files:**
- Create: `semanticsd/admin/install.py`

- [ ] **Step 1: Implement install.py (logic; CLI wiring in next task)**

```python
"""Install / uninstall SemanticsD as a launchd user agent."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from semanticsd import paths, config, keychain
from semanticsd.admin import launchd
from semanticsd.db import connection, migrations


def install() -> dict:
    """Idempotent install. Returns a dict of what was done."""
    actions = []

    paths.ensure_dirs()
    actions.append(f"ensured dirs at {paths.app_support()}")

    cfg_path = config.write_default()
    actions.append(f"config at {cfg_path}")

    token = keychain.generate_or_get_auth_token()
    actions.append("auth token in Keychain (service=semanticsd, account=api_token)")

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    actions.append(f"db migrated at {paths.db_path()}")

    plist_path = paths.launch_agent_plist()
    package_dir = str(Path(__file__).resolve().parent.parent.parent)
    launchd.write_plist(plist_path, python_executable=sys.executable, package_dir=package_dir)
    actions.append(f"plist at {plist_path}")

    # Bootstrap into the user's launchd domain. Bootout first to make idempotent.
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True,
    )
    result = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        actions.append(f"launchctl bootstrap failed: {result.stderr.strip()}")
    else:
        actions.append("launchctl bootstrap ok")

    return {"actions": actions, "token_hint": "stored in Keychain", "plist": str(plist_path)}


def uninstall() -> dict:
    """Stop and remove launchd agent. Does not delete index/config."""
    actions = []
    plist_path = paths.launch_agent_plist()
    uid = os.getuid()
    subprocess.run(
        ["launchctl", "bootout", f"gui/{uid}", str(plist_path)],
        check=False, capture_output=True,
    )
    actions.append("launchctl bootout")
    if plist_path.exists():
        plist_path.unlink()
        actions.append(f"removed {plist_path}")
    return {"actions": actions}


def print_token() -> str:
    tok = keychain.get_auth_token()
    if not tok:
        raise RuntimeError("no token in Keychain — run `semanticsd install` first")
    return tok
```

- [ ] **Step 2: Commit**

```bash
git add semanticsd/admin/install.py
git commit -m "feat(admin): install / uninstall / token print"
```

---

## Task 13: CLI — `semanticsd` admin app + `serve`

**Files:**
- Create: `semanticsd/cli.py`

- [ ] **Step 1: Implement cli.py with both Typer apps**

```python
"""CLI entry points: `semanticsd` (admin) and `ssearch` (client)."""
from __future__ import annotations
import json
import sys
import typer
import httpx
import uvicorn
from semanticsd import config, keychain, logging_setup, paths
from semanticsd.admin import install as admin_install


semanticsd_app = typer.Typer(no_args_is_help=True, help="SemanticsD daemon admin")
ssearch_app = typer.Typer(no_args_is_help=True, help="SemanticsD client CLI")


# ---- semanticsd admin ----

@semanticsd_app.command()
def serve():
    """Run the daemon (normally invoked by launchd)."""
    cfg = config.load()
    logging_setup.configure(level=cfg.daemon.log_level, to_file=True)
    from semanticsd.server.app import create_app
    app = create_app()
    uvicorn.run(
        app,
        host=cfg.daemon.http_host,
        port=cfg.daemon.http_port,
        log_level=cfg.daemon.log_level,
        access_log=False,
    )


@semanticsd_app.command()
def install():
    """Install launchd plist + config + token. Idempotent."""
    result = admin_install.install()
    typer.echo("SemanticsD installed.")
    for a in result["actions"]:
        typer.echo(f"  - {a}")
    typer.echo(f"\nToken: {result['token_hint']}")
    typer.echo(f"Plist: {result['plist']}")


@semanticsd_app.command()
def uninstall():
    """Stop and remove launchd agent. Does NOT delete index/config."""
    result = admin_install.uninstall()
    typer.echo("SemanticsD uninstalled.")
    for a in result["actions"]:
        typer.echo(f"  - {a}")


token_app = typer.Typer(help="Auth-token management")
semanticsd_app.add_typer(token_app, name="token")


@token_app.command("print")
def token_print():
    """Print the current API auth token."""
    typer.echo(admin_install.print_token())


# ---- ssearch client ----

def _client() -> httpx.Client:
    cfg = config.load()
    tok = keychain.get_auth_token()
    if not tok:
        typer.echo("ERROR: no auth token in Keychain. Run `semanticsd install` first.", err=True)
        raise typer.Exit(2)
    return httpx.Client(
        base_url=f"http://{cfg.daemon.http_host}:{cfg.daemon.http_port}",
        headers={"X-Auth-Token": tok},
        timeout=10.0,
    )


@ssearch_app.callback(invoke_without_command=True)
def ssearch_root(
    ctx: typer.Context,
    status: bool = typer.Option(False, "--status", help="Show daemon status."),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
):
    if status:
        try:
            with _client() as c:
                r = c.get("/v1/health")
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            typer.echo(f"status:    {body['status']}")
            typer.echo(f"version:   {body['version']}")
            typer.echo(f"doc_count: {body['doc_count']}")
            typer.echo(f"embedder:  {body['embedder']['message']}")
        return

    if ctx.invoked_subcommand is None:
        typer.echo("Usage: ssearch [QUERY] | --status | <subcommand>")
        typer.echo("Search subcommands land in Plan 5.")
        raise typer.Exit(0)
```

- [ ] **Step 2: Commit**

```bash
git add semanticsd/cli.py
git commit -m "feat(cli): semanticsd admin app + ssearch --status"
```

---

## Task 14: install.sh bootstrap script

**Files:**
- Create: `scripts/install.sh`

- [ ] **Step 1: Implement install.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
LAUNCHD_LABEL="com.semanticsd"
PLIST_PATH="$HOME/Library/LaunchAgents/${LAUNCHD_LABEL}.plist"

case "${1:-}" in
  --uninstall)
    echo "Uninstalling SemanticsD…"
    if [[ -f "$PLIST_PATH" ]]; then
      launchctl bootout "gui/$(id -u)" "$PLIST_PATH" || true
      rm -f "$PLIST_PATH"
      echo "  removed $PLIST_PATH"
    fi
    if [[ "${2:-}" == "--purge" ]]; then
      rm -rf "$HOME/Library/Application Support/semanticsd"
      rm -rf "$HOME/Library/Logs/semanticsd"
      echo "  purged Application Support + Logs"
    fi
    echo "Done."
    exit 0
    ;;
esac

# 1. macOS check
if [[ "$(uname)" != "Darwin" ]]; then
  echo "ERROR: SemanticsD is macOS-only." >&2
  exit 1
fi

# 2. Python 3.11+
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." >&2
  exit 1
fi
PYV=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(printf '%s\n' "3.11" "$PYV" | sort -V | head -n1)" != "3.11" ]]; then
  echo "ERROR: Python 3.11+ required (have $PYV)." >&2
  exit 1
fi

# 3. venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR…"
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip >/dev/null
"$VENV_DIR/bin/pip" install -r "$REPO_ROOT/requirements.txt"
"$VENV_DIR/bin/pip" install -e "$REPO_ROOT"

# 4. install launchd agent + token + config
"$VENV_DIR/bin/python" -m semanticsd install

# 5. wait for /v1/health
PORT=$("$VENV_DIR/bin/python" -c "from semanticsd import config; print(config.load().daemon.http_port)")
TOKEN=$("$VENV_DIR/bin/python" -m semanticsd token print)
echo "Waiting for daemon on :$PORT…"
for i in {1..30}; do
  if curl -s -H "X-Auth-Token: $TOKEN" "http://127.0.0.1:$PORT/v1/health" >/dev/null 2>&1; then
    echo "  ready."
    break
  fi
  sleep 1
done

echo
echo "SemanticsD is installed."
echo "  Try: ssearch --status"
echo "  Logs: ~/Library/Logs/semanticsd/"
echo "  Config: ~/Library/Application Support/semanticsd/config.toml"
```

- [ ] **Step 2: Make executable + commit**

```bash
chmod +x scripts/install.sh
git add scripts/install.sh
git commit -m "feat(install): bash bootstrap script"
```

---

## Task 15: End-to-end smoke test

**Files:**
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write a test that boots the FastAPI app in-process and exercises the CLI client logic**

`tests/test_e2e_smoke.py`:
```python
"""In-process end-to-end test: app + auth + health, no real launchd."""
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth
from semanticsd.db import connection, migrations
from semanticsd import paths


def test_smoke_install_then_query(tmp_app_support, monkeypatch):
    # Apply migrations the way `install` would.
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    monkeypatch.setattr(auth, "_token_cache", "smoke-token")
    app = server_app.create_app()
    client = TestClient(app)

    r = client.get("/v1/health", headers={"X-Auth-Token": "smoke-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["doc_count"] == 0
    assert body["version"]

    # Wrong token rejected:
    r = client.get("/v1/health", headers={"X-Auth-Token": "wrong"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/pytest -v
```
Expected: all tests pass (across all test files).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: e2e smoke test for install + health"
```

---

## Task 16: Manual install verification (one-time check; not committed)

This task is the human-driven smoke test of `install.sh`. No code change.

- [ ] **Step 1: Run install.sh**

```bash
./scripts/install.sh
```
Expected:
- Venv created at `.venv/`.
- "SemanticsD installed." printed.
- "Waiting for daemon on :47600… ready." printed.

- [ ] **Step 2: Hit the daemon directly with curl**

```bash
TOKEN=$(.venv/bin/python -m semanticsd token print)
curl -s -H "X-Auth-Token: $TOKEN" http://127.0.0.1:47600/v1/health | python3 -m json.tool
```
Expected: JSON with `"status": "ok"`, `"doc_count": 0`.

- [ ] **Step 3: Hit it via the CLI**

```bash
.venv/bin/ssearch --status
```
Expected:
```
status:    ok
version:   0.1.0
doc_count: 0
embedder:  not configured (Plan 2)
```

- [ ] **Step 4: Tail the launchd log**

```bash
tail -n 20 ~/Library/Logs/semanticsd/semanticsd.err.log
```
Expected: uvicorn startup line, no tracebacks.

- [ ] **Step 5: Verify launchctl sees it**

```bash
launchctl list | grep semanticsd
```
Expected: a line with `com.semanticsd` and a non-error exit code.

- [ ] **Step 6: Uninstall and reverify**

```bash
./scripts/install.sh --uninstall
launchctl list | grep semanticsd || echo "gone"
```
Expected: `gone`.

If anything in Steps 1-6 fails, fix and re-run. Do **not** proceed to Plan 2 until all six steps pass.

---

## What's Next (Plan 2 preview)

Plan 2 will add:
- `sqlite-vec` extension load + `vec_embeddings` virtual table (schema bump to v2).
- `semanticsd/embedders/`: base, registry, factory, local (sentence-transformers), openai_compatible, openai, ollama, voyage, cohere, gemini, vertex, bedrock.
- Provider preset registry endpoint (`GET /v1/presets`).
- Test-connection endpoint (`POST /v1/embedder/test`).
- Added requirements: `sqlite-vec`, `sentence-transformers`, `torch`, `openai`, `voyageai`, `cohere`, `google-generativeai`, `google-cloud-aiplatform`, `boto3`, `tiktoken`.
- `ssearch --presets` and `ssearch --test-embedder` CLI flags.
- `/v1/health` updated to actually exercise the configured embedder.

Plans 3-6 cover document pipeline, watcher+power, search engine, and cost tracking — each producing working software on its own.
