# SemanticsD — Plan 2: Embedder Layer + sqlite-vec + Sandbox Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the pluggable embedder layer (base + LocalEmbedder + OpenAICompatibleEmbedder + OpenAIEmbedder + OllamaEmbedder), bump SQLite schema to v2 with `sqlite-vec`'s `vec_embeddings(dim=384)` virtual table, expose `/v1/presets` and `/v1/embedder/test`, upgrade `/v1/health` to exercise the configured embedder, add `ssearch --presets` / `--test-embedder` CLI commands, and ship a `sandbox/` development harness with `make dev-sandbox`. After this plan: pick a provider, round-trip-test it, see real vectors land in SQLite — no indexing or search yet.

**Architecture:** One file per embedder provider in `semanticsd/embedders/`, all deriving from `Embedder` ABC. Registry-driven factory selects implementation by preset id. `sqlite-vec` extension loaded into the connection. Default config ships with `directories = []` so a fresh install never auto-indexes anything until the user opts in. Sandbox harness keeps development off the user's `$HOME`.

**Tech Stack:** Python 3.11+ **with sqlite-extension support** (Homebrew `python@3.13` recommended; pyenv builds need `PYTHON_CONFIGURE_OPTS="--enable-loadable-sqlite-extensions"`), FastAPI, sqlite-vec (vector store, loaded into stdlib sqlite3), sentence-transformers + torch (local embedder), OpenAI Python SDK (OpenAI-compatible HTTP transport), tiktoken (token estimation), pytest + monkeypatch (no respx — we mock SDK methods directly).

**Important — Python build requirement:** sqlite-vec requires `Connection.enable_load_extension`, which is only available if Python's sqlite3 module was compiled with `--enable-loadable-sqlite-extensions`. Apple's bundled Python and pyenv-default builds lack this. The fix during dev is recreating the venv with Homebrew Python 3.13 (`/opt/homebrew/bin/python3.13`). This requirement is documented in `install.sh` (Plan 2.5 will add the runtime check; for now, the controller has rebuilt the dev venv with brew python).

**Spec reference:** `/Users/vedantvijay/dev/side_projects/SemanticSearch/docs/superpowers/specs/2026-05-09-semanticsd-design.md`
**Plan 1 reference:** `/Users/vedantvijay/dev/side_projects/SemanticSearch/docs/superpowers/plans/2026-05-09-foundation.md` (foundation already shipped)

---

## File Structure for This Plan

```
SemanticSearch/
├── semanticsd/
│   ├── config.py                       # MODIFY: default directories = []; add [embedding] preset field semantics
│   ├── cli.py                          # MODIFY: add --presets, --test-embedder
│   ├── db/
│   │   ├── connection.py               # MODIFY: import pysqlite3 as sqlite3; load sqlite-vec extension
│   │   ├── schema.py                   # MODIFY: SCHEMA_VERSION=2, DDL_V2 with vec_embeddings(dim=384)
│   │   └── migrations.py               # MODIFY: apply DDL_V2 on upgrade
│   ├── embedders/
│   │   ├── __init__.py                 # NEW: public exports + get_active_embedder() cached factory
│   │   ├── base.py                     # NEW: Embedder ABC + EmbedResult model
│   │   ├── registry.py                 # NEW: PROVIDER_REGISTRY + build_embedder()
│   │   ├── local.py                    # NEW: LocalEmbedder (sentence-transformers)
│   │   ├── openai_compatible.py        # NEW: OpenAICompatibleEmbedder
│   │   ├── openai.py                   # NEW: OpenAIEmbedder (subclass with cost table)
│   │   └── ollama.py                   # NEW: OllamaEmbedder (subclass, localhost, no key)
│   └── server/
│       ├── app.py                      # MODIFY: include presets, embedder_test routers
│       └── routes/
│           ├── presets.py              # NEW: GET /v1/presets
│           ├── embedder_test.py        # NEW: POST /v1/embedder/test
│           └── health.py               # MODIFY: exercise configured embedder
├── tests/
│   ├── test_config.py                  # MODIFY: update test for new default directories=[]
│   ├── test_db.py                      # MODIFY: add v2 schema test for vec_embeddings
│   ├── test_embedders_base.py          # NEW
│   ├── test_embedders_local.py         # NEW
│   ├── test_embedders_openai_compatible.py # NEW
│   ├── test_embedders_openai.py        # NEW
│   ├── test_embedders_ollama.py        # NEW
│   ├── test_embedders_registry.py      # NEW
│   ├── test_presets_route.py           # NEW
│   ├── test_embedder_test_route.py     # NEW
│   ├── test_health.py                  # MODIFY: assert embedder fields when configured
│   └── test_e2e_vec.py                 # NEW: real LocalEmbedder → vec0 round-trip
├── sandbox/                            # NEW: dev harness
│   ├── README.md
│   ├── notes/alpha.md
│   ├── notes/beta.md
│   ├── code/hello.py
│   └── docs/design.txt
├── Makefile                            # MODIFY: add dev-sandbox target
└── requirements.txt                    # MODIFY: add sqlite-vec, sentence-transformers, torch, openai, tiktoken, pysqlite3-binary
```

---

## Pre-Task: Bootstrap dependencies (controller-run, before Task 1)

Before dispatching subagents, the controller must:

1. Verify Python supports sqlite extensions:
   ```bash
   .venv/bin/python -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True); print('ok')"
   ```
   If this errors with `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'`, the venv must be recreated with a Python build that has extensions enabled (e.g., `/opt/homebrew/bin/python3.13`).

2. Install Plan 2 deps:
   ```bash
   .venv/bin/pip install --upgrade pip
   .venv/bin/pip install \
       sqlite-vec>=0.1 \
       sentence-transformers>=2.7 \
       torch>=2.2 \
       openai>=1.30 \
       tiktoken>=0.7
   ```

This is done once per Plan-2 execution to avoid every subagent re-installing.

---

## Task 1: Update requirements.txt + Makefile (add deps + dev-sandbox)

**Files:**
- Modify: `requirements.txt`
- Modify: `Makefile`

- [ ] **Step 1: Append new deps to requirements.txt**

Append to the existing `requirements.txt`:
```
# Plan 2 additions
sqlite-vec>=0.1
sentence-transformers>=2.7
torch>=2.2
openai>=1.30
tiktoken>=0.7
```

(Note: `pysqlite3-binary` is NOT used. We rely on stdlib sqlite3 from a Python build with extension support — see plan tech-stack note above.)

- [ ] **Step 2: Add `dev-sandbox` target to Makefile**

Append to `Makefile`:
```make
.PHONY: dev-sandbox

dev-sandbox: install
	SEMANTICSD_HOME=$(PWD)/sandbox/.semanticsd \
	$(PYTHON) -c "from semanticsd import config, paths; paths.ensure_dirs(); \
	open(paths.config_path(),'w').write(config.DEFAULT_TOML.replace('directories = []','directories = [\"$(PWD)/sandbox\"]'))" && \
	SEMANTICSD_HOME=$(PWD)/sandbox/.semanticsd $(PYTHON) -m semanticsd serve
```

The recipe lines must use **tab** indentation (Makefile rule).

This target:
1. Sets `SEMANTICSD_HOME` to a sandbox-private state dir (`./sandbox/.semanticsd/`).
2. Generates a config there with `[watch] directories = ["./sandbox"]`.
3. Runs the daemon in foreground (so logs stream to terminal, Ctrl+C stops it).

- [ ] **Step 3: Verify Makefile parses**

```bash
make -n dev-sandbox
```
Expected: prints commands without error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt Makefile
git commit -m "chore(deps): add sqlite-vec + embedder libs + dev-sandbox target"
```

---

## Task 2: Sandbox fixtures

**Files:**
- Create: `sandbox/README.md`
- Create: `sandbox/notes/alpha.md`
- Create: `sandbox/notes/beta.md`
- Create: `sandbox/code/hello.py`
- Create: `sandbox/docs/design.txt`

- [ ] **Step 1: Create directory structure and seed files**

```bash
mkdir -p sandbox/notes sandbox/code sandbox/docs
```

`sandbox/README.md`:
```markdown
# SemanticsD Sandbox

This directory holds seed files for SemanticsD development. Run `make dev-sandbox`
from the repo root and the daemon will index this folder (and only this folder).

State for the sandboxed daemon (config, db, logs) lives in `./sandbox/.semanticsd/`,
not in `~/Library/Application Support/semanticsd/`. Untracked.
```

`sandbox/notes/alpha.md`:
```markdown
# Alpha Note

The alpha protocol describes how messages are exchanged between two parties.
Authentication uses a shared secret derived from a passphrase.

## Steps

1. Both parties derive a key from the shared passphrase.
2. The initiator sends a hello frame.
3. The responder replies with a challenge.
4. The initiator answers with the HMAC of the challenge.
```

`sandbox/notes/beta.md`:
```markdown
# Beta Note

Notes on the beta release of the indexing pipeline. Performance improved by
batching embedding calls. The chunker now respects code boundaries via
tree-sitter, falling back to a sliding window for unsupported languages.
```

`sandbox/code/hello.py`:
```python
"""A trivial example used as a sandbox fixture for SemanticsD tests."""


def greet(name: str) -> str:
    """Return a friendly greeting for the given name."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("world"))
```

`sandbox/docs/design.txt`:
```
SemanticsD design overview.

The daemon watches configured directories for changes. When a file is created or
modified, an extractor pulls text out of it. The text is chunked, hashed, and
embedded. Embeddings land in sqlite-vec for nearest-neighbor lookup. Filename
and full-text searches use SQLite FTS5 tables maintained alongside.
```

- [ ] **Step 2: Add `sandbox/.semanticsd/` to .gitignore**

Append to `.gitignore`:
```
sandbox/.semanticsd/
```

- [ ] **Step 3: Commit**

```bash
git add sandbox/ .gitignore
git commit -m "feat(sandbox): seed fixtures for development harness"
```

---

## Task 3: Default config — empty `directories` (no surprise indexing)

**Files:**
- Modify: `semanticsd/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Update the failing test first**

Edit `tests/test_config.py` — replace the `test_defaults_when_no_file` body and add a new assertion:

```python
def test_defaults_when_no_file(tmp_app_support):
    cfg = config.load()
    assert cfg.daemon.http_host == "127.0.0.1"
    assert cfg.daemon.http_port == 47600
    assert cfg.search.default_mode == "semantic"
    assert cfg.power.mode == "active"
    # Plan 2: no surprise auto-indexing on a fresh install.
    assert cfg.watch.directories == []
```

Add a new test below it:

```python
def test_default_toml_has_empty_directories(tmp_app_support):
    config.write_default()
    text = (tmp_app_support / "config.toml").read_text()
    assert "directories = []" in text
```

- [ ] **Step 2: Run tests — expect 1 failure (existing default is `["~/"]`)**

```bash
.venv/bin/pytest tests/test_config.py -v
```
Expected: `test_defaults_when_no_file` fails on the new assertion.

- [ ] **Step 3: Modify the model default and the DEFAULT_TOML in `semanticsd/config.py`**

In `semanticsd/config.py`, change:
```python
class WatchConfig(BaseModel):
    directories: list[str] = ["~/"]
```
to:
```python
class WatchConfig(BaseModel):
    directories: list[str] = []
```

And in the `DEFAULT_TOML` string, change:
```
directories = ["~/"]
```
to:
```
directories = []
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```
Expected: 5 passed (4 prior + 1 new).

- [ ] **Step 5: Run full suite — no regressions**

```bash
.venv/bin/pytest -q
```
Expected: 32 passed (31 prior + 1 new).

- [ ] **Step 6: Commit**

```bash
git add semanticsd/config.py tests/test_config.py
git commit -m "feat(config): default directories=[] — no surprise indexing on install"
```

---

## Task 4: Schema v2 — sqlite-vec + `vec_embeddings(dim=384)` (TDD)

**Files:**
- Modify: `semanticsd/db/connection.py`
- Modify: `semanticsd/db/schema.py`
- Modify: `semanticsd/db/migrations.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_db.py`:

```python
def test_vec_embeddings_table_exists(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE name='vec_embeddings'")
    assert cur.fetchone() is not None


def test_vec_embeddings_round_trip(tmp_path):
    """Insert a 384-dim vector, query it back."""
    import struct
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    vec = [0.1] * 384
    blob = struct.pack(f"{len(vec)}f", *vec)
    conn.execute("INSERT INTO vec_embeddings(rowid, embedding) VALUES (1, ?)", (blob,))
    row = conn.execute(
        "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT 1",
        (blob,),
    ).fetchone()
    assert row is not None
    assert row[0] == 1
    # distance to self ~ 0
    assert row[1] < 1e-3


def test_schema_version_is_2(tmp_path):
    db = tmp_path / "test.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    v = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    assert int(v[0]) == 2
```

- [ ] **Step 2: Run tests — expect fail**

```bash
.venv/bin/pytest tests/test_db.py -v
```
Expected: 3 new tests fail (no `vec_embeddings` table; schema_version still 1).

- [ ] **Step 3: Update `semanticsd/db/connection.py` to load sqlite-vec**

Replace the entire contents of `semanticsd/db/connection.py`:

```python
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
```

- [ ] **Step 4: Update `semanticsd/db/schema.py` to add v2 DDL**

Edit `semanticsd/db/schema.py` — bump `SCHEMA_VERSION` and add `DDL_V2`:

```python
"""Schema DDL.

Plan 1 created the relational tables. Plan 2 adds vec_embeddings (sqlite-vec
virtual table) at dim=384 to match the default LocalEmbedder
(BAAI/bge-small-en-v1.5).
"""

SCHEMA_VERSION = 2

DDL_V1 = [
    # ... (existing DDL_V1 list — leave unchanged)
]

DDL_V2 = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0(
        embedding FLOAT[384]
    )
    """,
]
```

(Keep the existing `DDL_V1` list verbatim. Only add `DDL_V2` and bump `SCHEMA_VERSION` to 2.)

- [ ] **Step 5: Update `semanticsd/db/migrations.py` to apply v2**

Replace the body of `apply()` in `semanticsd/db/migrations.py`:

```python
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
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(schema.SCHEMA_VERSION),),
    )
```

(Note the broader `except Exception` — pysqlite3's exception class differs slightly from stdlib sqlite3, so a narrow catch could miss. The intent is "if meta table doesn't exist, current is 0".)

- [ ] **Step 6: Run db tests — expect pass**

```bash
.venv/bin/pytest tests/test_db.py -v
```
Expected: 8 passed (5 prior + 3 new).

- [ ] **Step 7: Run full suite — no regressions**

```bash
.venv/bin/pytest -q
```
Expected: 35 passed.

- [ ] **Step 8: Commit**

```bash
git add semanticsd/db/ tests/test_db.py
git commit -m "feat(db): schema v2 — sqlite-vec + vec_embeddings(dim=384)"
```

---

## Task 5: Embedder base class + `EmbedResult` (TDD)

**Files:**
- Create: `semanticsd/embedders/__init__.py` (stub)
- Create: `semanticsd/embedders/base.py`
- Create: `tests/test_embedders_base.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_base.py`:
```python
"""Tests for the Embedder ABC and EmbedResult model."""
from semanticsd.embedders.base import Embedder, EmbedResult


def test_embed_result_has_required_fields():
    r = EmbedResult(vectors=[[0.1, 0.2]], input_tokens=4)
    assert r.vectors == [[0.1, 0.2]]
    assert r.input_tokens == 4
    assert r.output_tokens == 0
    assert r.raw_response is None


def test_embedder_is_abstract():
    """Abstract methods cannot be instantiated directly."""
    try:
        Embedder()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Embedder should be abstract")


def test_concrete_subclass_can_be_instantiated():
    class Stub(Embedder):
        provider_id = "stub"
        model_id = "stub-1"
        dim = 4
        supports_kind = False
        cost_per_million_input_tokens_usd = 0.0

        def embed(self, texts, kind):
            return EmbedResult(vectors=[[0.0] * self.dim for _ in texts], input_tokens=0)

        def health_check(self):
            return (True, "ok")

        def estimate_tokens(self, texts):
            return sum(len(t) // 4 for t in texts)

    s = Stub()
    out = s.embed(["hello", "world"], kind="doc")
    assert len(out.vectors) == 2
    assert s.health_check() == (True, "ok")
    assert s.estimate_tokens(["abcdefgh"]) == 2
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_base.py -v
```
Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Create `semanticsd/embedders/__init__.py` (stub for now)**

```python
"""Embedder layer — pluggable providers, one file per provider."""
```

- [ ] **Step 4: Implement `semanticsd/embedders/base.py`**

```python
"""Abstract Embedder base + result model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel


class EmbedResult(BaseModel):
    vectors: list[list[float]]
    input_tokens: int
    output_tokens: int = 0
    raw_response: dict | None = None


class Embedder(ABC):
    """Pluggable embedder. Subclass and register in registry.py.

    Required class attributes (set on the subclass):
      provider_id: str  — preset key, e.g. "openai" / "local" / "ollama"
      model_id: str
      dim: int
      supports_kind: bool  — True if the provider distinguishes doc vs query
      cost_per_million_input_tokens_usd: float
    """

    provider_id: str = ""
    model_id: str = ""
    dim: int = 0
    supports_kind: bool = False
    cost_per_million_input_tokens_usd: float = 0.0

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult: ...

    @abstractmethod
    def health_check(self) -> tuple[bool, str]: ...

    @abstractmethod
    def estimate_tokens(self, texts: list[str]) -> int: ...
```

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_base.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/embedders/ tests/test_embedders_base.py
git commit -m "feat(embedders): base ABC + EmbedResult model"
```

---

## Task 6: LocalEmbedder (sentence-transformers; mocked in unit tests) (TDD)

**Files:**
- Create: `semanticsd/embedders/local.py`
- Create: `tests/test_embedders_local.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_local.py`:
```python
"""LocalEmbedder unit tests — sentence-transformers is mocked.
The real-model integration test lives in tests/test_e2e_vec.py.
"""
import numpy as np
import pytest
from semanticsd.embedders.base import EmbedResult
from semanticsd.embedders.local import LocalEmbedder


class FakeST:
    """Fake SentenceTransformer that returns deterministic 384-d vectors."""
    def __init__(self, model_name, device=None, cache_folder=None):
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        # Deterministic: each text -> a 384-vec scaled by len(text).
        return np.array(
            [[float(len(t) % 7) / 10.0] * 384 for t in texts],
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(self):
        return 384


@pytest.fixture
def patched_st(monkeypatch):
    monkeypatch.setattr(
        "semanticsd.embedders.local.SentenceTransformer",
        FakeST,
    )


def test_local_embedder_metadata(patched_st):
    e = LocalEmbedder()
    assert e.provider_id == "local"
    assert e.model_id == "BAAI/bge-small-en-v1.5"
    assert e.dim == 384
    assert e.supports_kind is False
    assert e.cost_per_million_input_tokens_usd == 0.0


def test_local_embed_returns_correct_shape(patched_st):
    e = LocalEmbedder()
    out = e.embed(["alpha", "bravo charlie"], kind="doc")
    assert isinstance(out, EmbedResult)
    assert len(out.vectors) == 2
    assert all(len(v) == 384 for v in out.vectors)
    assert out.input_tokens > 0


def test_local_health_check_ok(patched_st):
    e = LocalEmbedder()
    ok, msg = e.health_check()
    assert ok is True


def test_local_estimate_tokens(patched_st):
    e = LocalEmbedder()
    n = e.estimate_tokens(["abcdefgh", "ijkl"])
    # Heuristic: chars / 4 — "abcdefgh"=2, "ijkl"=1 => 3
    assert n == 3


def test_local_lazy_loads_model(patched_st):
    """Model isn't loaded until first embed() call."""
    e = LocalEmbedder()
    assert e._model is None
    e.embed(["x"], kind="doc")
    assert e._model is not None


def test_custom_model_id(patched_st):
    e = LocalEmbedder(model_id="custom-model")
    assert e.model_id == "custom-model"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_local.py -v
```

- [ ] **Step 3: Implement `semanticsd/embedders/local.py`**

```python
"""LocalEmbedder — sentence-transformers in-process. Default zero-config provider.

The model is downloaded lazily on first embed() call to ~/Library/Application
Support/semanticsd/models/. No API key, no network at startup, costs $0.
"""
from __future__ import annotations
from typing import Literal
from sentence_transformers import SentenceTransformer
from semanticsd import paths
from semanticsd.embedders.base import Embedder, EmbedResult


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class LocalEmbedder(Embedder):
    provider_id = "local"
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def __init__(self, model_id: str = DEFAULT_MODEL, dim: int = DEFAULT_DIM):
        self.model_id = model_id
        self.dim = dim
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            paths.ensure_dirs()
            self._model = SentenceTransformer(
                self.model_id,
                cache_folder=str(paths.models_dir()),
            )
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim != self.dim:
                self.dim = actual_dim
        return self._model

    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult:
        model = self._ensure_model()
        arr = model.encode(texts, normalize_embeddings=True)
        # arr is np.ndarray; convert to plain Python list of lists.
        vectors = [v.tolist() for v in arr]
        return EmbedResult(
            vectors=vectors,
            input_tokens=self.estimate_tokens(texts),
        )

    def health_check(self) -> tuple[bool, str]:
        try:
            self._ensure_model()
            return (True, f"local model {self.model_id} loaded")
        except Exception as e:
            return (False, f"local model failed: {e}")

    def estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) // 4 for t in texts)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_local.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv/bin/pytest -q
```
Expected: 41 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/embedders/local.py tests/test_embedders_local.py
git commit -m "feat(embedders): LocalEmbedder (sentence-transformers, lazy-loaded)"
```

---

## Task 7: OpenAICompatibleEmbedder (TDD)

**Files:**
- Create: `semanticsd/embedders/openai_compatible.py`
- Create: `tests/test_embedders_openai_compatible.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_openai_compatible.py`:
```python
"""OpenAICompatibleEmbedder — covers any /v1/embeddings server.
The OpenAI Python SDK is monkey-patched; no network calls.
"""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


class FakeEmbeddings:
    """Replaces openai.OpenAI().embeddings."""
    def __init__(self, dim=384):
        self.dim = dim
        self.last_request = None

    def create(self, model, input, **kwargs):
        self.last_request = {"model": model, "input": input, "kwargs": kwargs}
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * self.dim) for _ in input],
            usage=SimpleNamespace(prompt_tokens=sum(len(t) // 4 for t in input)),
        )


class FakeClient:
    def __init__(self, dim=384):
        self.embeddings = FakeEmbeddings(dim=dim)


@pytest.fixture
def fake_openai(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        lambda **kwargs: fake,
    )
    return fake


def test_metadata(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="anything",
        model="some-embed-model",
        dim=384,
    )
    assert e.provider_id == "openai_compatible"
    assert e.model_id == "some-embed-model"
    assert e.dim == 384


def test_embed_calls_sdk(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    out = e.embed(["hello", "world"], kind="doc")
    assert len(out.vectors) == 2
    assert all(len(v) == 384 for v in out.vectors)
    assert fake_openai.embeddings.last_request["model"] == "m"
    assert fake_openai.embeddings.last_request["input"] == ["hello", "world"]


def test_embed_passes_dimensions_when_set(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
        dimensions=512,  # Matryoshka truncation
    )
    e.embed(["x"], kind="doc")
    assert fake_openai.embeddings.last_request["kwargs"].get("dimensions") == 512


def test_embed_does_not_pass_dimensions_when_zero(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
        dimensions=0,
    )
    e.embed(["x"], kind="doc")
    assert "dimensions" not in fake_openai.embeddings.last_request["kwargs"]


def test_input_tokens_from_sdk_usage(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    out = e.embed(["abcdefgh", "ij"], kind="doc")
    assert out.input_tokens == 2 + 0  # heuristic from FakeEmbeddings


def test_health_check_does_a_short_embed(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    ok, msg = e.health_check()
    assert ok is True


def test_estimate_tokens_heuristic(fake_openai):
    e = OpenAICompatibleEmbedder(
        base_url="http://localhost:1234/v1",
        api_key="x",
        model="m",
        dim=384,
    )
    assert e.estimate_tokens(["abcdefgh", "ij"]) == 2
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_openai_compatible.py -v
```

- [ ] **Step 3: Implement `semanticsd/embedders/openai_compatible.py`**

```python
"""OpenAICompatibleEmbedder — generic /v1/embeddings client.

Covers OpenAI, Ollama (with its /v1 compat endpoint), LM Studio, vLLM, llama.cpp
server, OpenRouter, Together, Groq, Fireworks, TEI, and any self-hosted server
that speaks the OpenAI embeddings API.

Constructor: (base_url, api_key, model, dim, dimensions=None).
"""
from __future__ import annotations
from typing import Literal
from openai import OpenAI
from semanticsd.embedders.base import Embedder, EmbedResult


class OpenAICompatibleEmbedder(Embedder):
    provider_id = "openai_compatible"
    supports_kind = False  # generic OpenAI-compat servers ignore input_type
    cost_per_million_input_tokens_usd = 0.0  # subclasses override

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        dimensions: int = 0,
    ):
        self.base_url = base_url
        self.api_key = api_key or "not-needed"  # local servers reject empty strings
        self.model_id = model
        self.dim = dim
        self.dimensions = dimensions
        self._client = OpenAI(base_url=base_url, api_key=self.api_key)

    def embed(
        self,
        texts: list[str],
        kind: Literal["doc", "query"],
    ) -> EmbedResult:
        kwargs = {}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embeddings.create(
            model=self.model_id, input=texts, **kwargs
        )
        vectors = [list(d.embedding) for d in resp.data]
        # Some servers (Ollama) don't populate usage; fall back to heuristic.
        usage_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        input_tokens = (
            int(usage_tokens) if usage_tokens is not None else self.estimate_tokens(texts)
        )
        return EmbedResult(vectors=vectors, input_tokens=input_tokens)

    def health_check(self) -> tuple[bool, str]:
        try:
            self.embed(["ping"], kind="query")
            return (True, f"{self.base_url} reachable, model {self.model_id} ok")
        except Exception as e:
            return (False, f"{self.base_url} unreachable: {e}")

    def estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) // 4 for t in texts)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_openai_compatible.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/embedders/openai_compatible.py tests/test_embedders_openai_compatible.py
git commit -m "feat(embedders): OpenAICompatibleEmbedder (generic /v1/embeddings)"
```

---

## Task 8: OpenAIEmbedder subclass (TDD)

**Files:**
- Create: `semanticsd/embedders/openai.py`
- Create: `tests/test_embedders_openai.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_openai.py`:
```python
"""OpenAIEmbedder — thin subclass of OpenAICompatibleEmbedder."""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.openai import OpenAIEmbedder, COST_PER_MILLION


class FakeEmbeddings:
    def create(self, model, input, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input],
            usage=SimpleNamespace(prompt_tokens=10),
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings()
        self.kwargs = kwargs


@pytest.fixture
def fake_openai(monkeypatch):
    """Patch the OpenAI symbol used by openai_compatible (its parent module)."""
    last = {}

    def factory(**kwargs):
        last["kwargs"] = kwargs
        c = FakeClient(**kwargs)
        return c

    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        factory,
    )
    return last


def test_default_base_url_is_openai(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.base_url == "https://api.openai.com/v1"
    assert fake_openai["kwargs"]["base_url"] == "https://api.openai.com/v1"


def test_default_model_is_3_small(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.model_id == "text-embedding-3-small"
    assert e.dim == 1536


def test_provider_id_is_openai(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.provider_id == "openai"


def test_cost_table_for_3_small(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test")
    assert e.cost_per_million_input_tokens_usd == COST_PER_MILLION["text-embedding-3-small"]


def test_cost_table_for_3_large(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
    assert e.cost_per_million_input_tokens_usd == COST_PER_MILLION["text-embedding-3-large"]


def test_cost_zero_for_unknown_model(fake_openai):
    e = OpenAIEmbedder(api_key="sk-test", model="some-future-model")
    assert e.cost_per_million_input_tokens_usd == 0.0
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_openai.py -v
```

- [ ] **Step 3: Implement `semanticsd/embedders/openai.py`**

```python
"""OpenAIEmbedder — OpenAI's hosted embeddings API.
Thin subclass of OpenAICompatibleEmbedder with cost table and sane defaults.
"""
from __future__ import annotations
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


# USD per 1M input tokens, as of 2026.
COST_PER_MILLION = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

DIM_BY_MODEL = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(OpenAICompatibleEmbedder):
    provider_id = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = 0,
    ):
        dim = dimensions if dimensions else DIM_BY_MODEL.get(model, 0)
        super().__init__(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            model=model,
            dim=dim,
            dimensions=dimensions,
        )
        self.cost_per_million_input_tokens_usd = COST_PER_MILLION.get(model, 0.0)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_openai.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/embedders/openai.py tests/test_embedders_openai.py
git commit -m "feat(embedders): OpenAIEmbedder subclass with cost table"
```

---

## Task 9: OllamaEmbedder subclass (TDD)

**Files:**
- Create: `semanticsd/embedders/ollama.py`
- Create: `tests/test_embedders_ollama.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_ollama.py`:
```python
"""OllamaEmbedder — local Ollama via its OpenAI-compatible /v1 endpoint."""
import pytest
from types import SimpleNamespace
from semanticsd.embedders.ollama import OllamaEmbedder


class FakeEmbeddings:
    def create(self, model, input, **kwargs):
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 768) for _ in input],
            usage=None,  # Ollama doesn't always populate usage
        )


class FakeClient:
    def __init__(self, **kwargs):
        self.embeddings = FakeEmbeddings()


@pytest.fixture
def fake_openai(monkeypatch):
    last = {}

    def factory(**kwargs):
        last["kwargs"] = kwargs
        return FakeClient(**kwargs)

    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        factory,
    )
    return last


def test_default_base_url_is_localhost(fake_openai):
    e = OllamaEmbedder()
    assert e.base_url == "http://localhost:11434/v1"


def test_default_model_is_nomic(fake_openai):
    e = OllamaEmbedder()
    assert e.model_id == "nomic-embed-text"
    assert e.dim == 768


def test_provider_id_is_ollama(fake_openai):
    e = OllamaEmbedder()
    assert e.provider_id == "ollama"


def test_no_api_key_required(fake_openai):
    e = OllamaEmbedder()
    # The internal SDK client gets a placeholder; the user passed nothing.
    assert e.api_key == "ollama"


def test_cost_is_zero(fake_openai):
    e = OllamaEmbedder()
    assert e.cost_per_million_input_tokens_usd == 0.0


def test_input_tokens_falls_back_to_heuristic_when_no_usage(fake_openai):
    e = OllamaEmbedder()
    out = e.embed(["abcdefgh"], kind="doc")  # 8 chars => 2 tokens
    assert out.input_tokens == 2
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_ollama.py -v
```

- [ ] **Step 3: Implement `semanticsd/embedders/ollama.py`**

```python
"""OllamaEmbedder — Ollama via its OpenAI-compat /v1 endpoint.
No API key, defaults to localhost:11434. Cost = 0 (local).
"""
from __future__ import annotations
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768


class OllamaEmbedder(OpenAICompatibleEmbedder):
    provider_id = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        dim: int = DEFAULT_DIM,
    ):
        super().__init__(
            base_url=base_url,
            api_key="ollama",  # placeholder; Ollama ignores it
            model=model,
            dim=dim,
        )
        self.cost_per_million_input_tokens_usd = 0.0
```

- [ ] **Step 4: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_ollama.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/embedders/ollama.py tests/test_embedders_ollama.py
git commit -m "feat(embedders): OllamaEmbedder subclass (localhost, free)"
```

---

## Task 10: Provider registry + `build_embedder()` factory (TDD)

**Files:**
- Create: `semanticsd/embedders/registry.py`
- Modify: `semanticsd/embedders/__init__.py`
- Create: `tests/test_embedders_registry.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedders_registry.py`:
```python
"""Provider registry + build_embedder() factory."""
import pytest
from semanticsd.embedders import registry
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


def test_registry_has_expected_keys():
    keys = set(registry.PROVIDER_REGISTRY.keys())
    expected = {"local", "openai", "ollama", "openai_compatible", "lmstudio", "vllm", "custom"}
    assert expected.issubset(keys)


def test_registry_entry_shape():
    for preset, entry in registry.PROVIDER_REGISTRY.items():
        assert "class" in entry
        assert "default_model" in entry or entry["default_model"] is None
        assert "needs_api_key" in entry


def test_build_local_embedder(monkeypatch):
    # Local should not require an API key or anything special.
    e = registry.build_embedder("local", config={})
    assert isinstance(e, LocalEmbedder)


def test_build_openai_requires_api_key():
    try:
        registry.build_embedder("openai", config={"api_key": ""})
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing OpenAI API key")


def test_build_openai_with_api_key():
    e = registry.build_embedder("openai", config={"api_key": "sk-test"})
    assert isinstance(e, OpenAIEmbedder)


def test_build_ollama_no_key_needed():
    e = registry.build_embedder("ollama", config={})
    assert isinstance(e, OllamaEmbedder)


def test_build_custom_requires_base_url():
    try:
        registry.build_embedder(
            "custom", config={"api_key": "x", "model": "m", "dim": 384}
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing base_url")


def test_build_custom_full():
    e = registry.build_embedder(
        "custom",
        config={
            "base_url": "http://localhost:1234/v1",
            "api_key": "anything",
            "model": "some-model",
            "dim": 384,
        },
    )
    assert isinstance(e, OpenAICompatibleEmbedder)


def test_unknown_preset_raises():
    try:
        registry.build_embedder("nonexistent", config={})
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedders_registry.py -v
```

- [ ] **Step 3: Implement `semanticsd/embedders/registry.py`**

```python
"""Provider registry + factory.

PROVIDER_REGISTRY maps preset id -> spec dict consumed by both the factory
(server-side embedder construction) and the /v1/presets endpoint
(frontend-side dropdown rendering).
"""
from __future__ import annotations
from typing import Any, Type
from semanticsd.embedders.base import Embedder
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder


PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "local": {
        "class": "LocalEmbedder",
        "default_model": "BAAI/bge-small-en-v1.5",
        "needs_api_key": False,
        "needs_base_url": False,
    },
    "ollama": {
        "class": "OllamaEmbedder",
        "default_model": "nomic-embed-text",
        "needs_api_key": False,
        "needs_base_url": False,
        "default_base_url": "http://localhost:11434/v1",
    },
    "lmstudio": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:1234/v1",
    },
    "vllm": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
        "default_base_url": "http://localhost:8000/v1",
    },
    "openai": {
        "class": "OpenAIEmbedder",
        "default_model": "text-embedding-3-small",
        "needs_api_key": True,
        "needs_base_url": False,
    },
    "openai_compatible": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
    },
    "custom": {
        "class": "OpenAICompatibleEmbedder",
        "default_model": None,
        "needs_api_key": False,
        "needs_base_url": True,
    },
}

_CLASS_BY_NAME: dict[str, Type[Embedder]] = {
    "LocalEmbedder": LocalEmbedder,
    "OllamaEmbedder": OllamaEmbedder,
    "OpenAIEmbedder": OpenAIEmbedder,
    "OpenAICompatibleEmbedder": OpenAICompatibleEmbedder,
}


def build_embedder(preset: str, config: dict[str, Any]) -> Embedder:
    """Construct an Embedder instance from a preset id + config dict.

    Required config keys depend on the preset (see PROVIDER_REGISTRY[preset]):
      - needs_api_key: include "api_key"
      - needs_base_url: include "base_url"
    Optional: "model", "dim", "dimensions".

    Raises ValueError on unknown preset or missing required config.
    """
    if preset not in PROVIDER_REGISTRY:
        raise ValueError(f"unknown embedder preset: {preset!r}")
    entry = PROVIDER_REGISTRY[preset]
    cls = _CLASS_BY_NAME[entry["class"]]

    if entry["needs_api_key"] and not config.get("api_key"):
        raise ValueError(f"preset {preset!r} requires an api_key")
    if entry["needs_base_url"] and not config.get("base_url") and not entry.get("default_base_url"):
        raise ValueError(f"preset {preset!r} requires a base_url")

    if cls is LocalEmbedder:
        return LocalEmbedder(model_id=config.get("model") or entry["default_model"])
    if cls is OpenAIEmbedder:
        return OpenAIEmbedder(
            api_key=config["api_key"],
            model=config.get("model") or entry["default_model"],
            dimensions=config.get("dimensions", 0),
        )
    if cls is OllamaEmbedder:
        base_url = config.get("base_url") or entry.get("default_base_url")
        return OllamaEmbedder(
            model=config.get("model") or entry["default_model"],
            base_url=base_url,
        )
    if cls is OpenAICompatibleEmbedder:
        base_url = config.get("base_url") or entry.get("default_base_url")
        if not base_url:
            raise ValueError(f"preset {preset!r} requires a base_url")
        if not config.get("model"):
            raise ValueError(f"preset {preset!r} requires a model")
        if not config.get("dim"):
            raise ValueError(f"preset {preset!r} requires a dim")
        return OpenAICompatibleEmbedder(
            base_url=base_url,
            api_key=config.get("api_key", ""),
            model=config["model"],
            dim=int(config["dim"]),
            dimensions=config.get("dimensions", 0),
        )
    raise ValueError(f"no factory branch for class {entry['class']!r}")
```

- [ ] **Step 4: Update `semanticsd/embedders/__init__.py` with public exports**

Replace the contents:
```python
"""Embedder layer — pluggable providers, one file per provider."""
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.embedders.openai_compatible import OpenAICompatibleEmbedder
from semanticsd.embedders.openai import OpenAIEmbedder
from semanticsd.embedders.ollama import OllamaEmbedder
from semanticsd.embedders.registry import PROVIDER_REGISTRY, build_embedder

__all__ = [
    "Embedder",
    "EmbedResult",
    "LocalEmbedder",
    "OpenAICompatibleEmbedder",
    "OpenAIEmbedder",
    "OllamaEmbedder",
    "PROVIDER_REGISTRY",
    "build_embedder",
]


_active: Embedder | None = None


def get_active_embedder(force_reload: bool = False) -> Embedder | None:
    """Return the configured embedder for the running daemon.

    Reads `[embedding]` from the daemon's config. Caches the instance.
    Returns None if no provider is selected (preset is empty).
    """
    global _active
    if _active is not None and not force_reload:
        return _active

    from semanticsd import config as cfg_mod
    from semanticsd import keychain
    cfg = cfg_mod.load()
    preset = cfg.embedding.preset
    if not preset:
        return None

    api_key = ""
    if PROVIDER_REGISTRY.get(preset, {}).get("needs_api_key"):
        api_key = keychain.get_provider_key(preset) or ""

    config_dict = {
        "api_key": api_key,
        "base_url": cfg.embedding.base_url,
        "model": cfg.embedding.model,
        "dimensions": cfg.embedding.dimensions,
    }
    _active = build_embedder(preset, config_dict)
    return _active


def reset_active_embedder() -> None:
    """Clear the cached embedder (used by tests + after config changes)."""
    global _active
    _active = None
```

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedders_registry.py -v
```
Expected: 9 passed.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/pytest -q
```
Expected: 56 passed.

- [ ] **Step 7: Commit**

```bash
git add semanticsd/embedders/registry.py semanticsd/embedders/__init__.py tests/test_embedders_registry.py
git commit -m "feat(embedders): provider registry + build_embedder factory"
```

---

## Task 11: `/v1/presets` route (TDD)

**Files:**
- Create: `semanticsd/server/routes/presets.py`
- Modify: `semanticsd/server/app.py`
- Create: `tests/test_presets_route.py`

- [ ] **Step 1: Write failing tests**

`tests/test_presets_route.py`:
```python
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth


def test_presets_unauthenticated_rejected(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.get("/v1/presets")
    assert r.status_code == 401


def test_presets_returns_registry(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.get("/v1/presets", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert "presets" in body
    assert "local" in body["presets"]
    assert "openai" in body["presets"]
    assert "ollama" in body["presets"]
    assert body["presets"]["openai"]["needs_api_key"] is True
    assert body["presets"]["local"]["needs_api_key"] is False
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_presets_route.py -v
```

- [ ] **Step 3: Implement `semanticsd/server/routes/presets.py`**

```python
"""GET /v1/presets — provider registry for frontend dropdowns."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd.server.auth import require_token
from semanticsd.embedders.registry import PROVIDER_REGISTRY

router = APIRouter()


@router.get("/presets", dependencies=[Depends(require_token)])
def presets() -> dict:
    return {"presets": PROVIDER_REGISTRY}
```

- [ ] **Step 4: Wire the router into the app**

Edit `semanticsd/server/app.py` — change the `from semanticsd.server.routes import health` line and the `include_router` calls to include presets:

```python
"""FastAPI app factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from semanticsd import __version__
from semanticsd.server.routes import health, presets


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
    app.include_router(presets.router, prefix="/v1")
    return app
```

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_presets_route.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/server/routes/presets.py semanticsd/server/app.py tests/test_presets_route.py
git commit -m "feat(server): GET /v1/presets endpoint"
```

---

## Task 12: `/v1/embedder/test` route (TDD)

**Files:**
- Create: `semanticsd/server/routes/embedder_test.py`
- Modify: `semanticsd/server/app.py`
- Create: `tests/test_embedder_test_route.py`

- [ ] **Step 1: Write failing tests**

`tests/test_embedder_test_route.py`:
```python
"""POST /v1/embedder/test — round-trip a probe embed against a configured backend."""
import pytest
from types import SimpleNamespace
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth


@pytest.fixture
def fake_openai(monkeypatch):
    """Stub OpenAI client so /v1/embedder/test can succeed without a network call."""
    class FakeEmbeddings:
        def create(self, model, input, **kwargs):
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.5] * 384) for _ in input],
                usage=SimpleNamespace(prompt_tokens=1),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(
        "semanticsd.embedders.openai_compatible.OpenAI",
        lambda **kwargs: FakeClient(**kwargs),
    )


def test_embedder_test_unauthed(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post("/v1/embedder/test", json={"preset": "local"})
    assert r.status_code == 401


def test_embedder_test_custom_preset(monkeypatch, fake_openai):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/embedder/test",
        headers={"X-Auth-Token": "secret"},
        json={
            "preset": "custom",
            "base_url": "http://localhost:1234/v1",
            "api_key": "anything",
            "model": "test-model",
            "dim": 384,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["dim"] == 384
    assert body["latency_ms"] >= 0


def test_embedder_test_missing_required_field(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    # custom requires base_url + model + dim
    r = client.post(
        "/v1/embedder/test",
        headers={"X-Auth-Token": "secret"},
        json={"preset": "custom", "api_key": "x"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "base_url" in body["error"] or "model" in body["error"] or "dim" in body["error"]


def test_embedder_test_unknown_preset(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/embedder/test",
        headers={"X-Auth-Token": "secret"},
        json={"preset": "nonexistent"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_embedder_test_route.py -v
```

- [ ] **Step 3: Implement `semanticsd/server/routes/embedder_test.py`**

```python
"""POST /v1/embedder/test — round-trip a probe embed against a candidate config.

Used by frontends to render a 'Test Connection' button. Never persists state.
"""
from __future__ import annotations
import time
from typing import Any
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from semanticsd.server.auth import require_token
from semanticsd.embedders.registry import build_embedder

router = APIRouter()


class TestRequest(BaseModel):
    preset: str
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    dim: int | None = None
    dimensions: int | None = None


@router.post("/embedder/test", dependencies=[Depends(require_token)])
def embedder_test(req: TestRequest) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_url": req.base_url or "",
        "api_key": req.api_key or "",
        "model": req.model or "",
        "dimensions": req.dimensions or 0,
    }
    if req.dim is not None:
        config["dim"] = req.dim

    try:
        embedder = build_embedder(req.preset, config=config)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    start = time.perf_counter()
    try:
        result = embedder.embed(["ping"], kind="query")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "ok": True,
        "provider_id": embedder.provider_id,
        "model_id": embedder.model_id,
        "dim": len(result.vectors[0]) if result.vectors else embedder.dim,
        "latency_ms": elapsed_ms,
    }
```

- [ ] **Step 4: Wire the router into the app**

Edit `semanticsd/server/app.py` — add `embedder_test` to the imports and an `include_router` call. The full file should now read:

```python
"""FastAPI app factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from semanticsd import __version__
from semanticsd.server.routes import health, presets, embedder_test


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
    app.include_router(presets.router, prefix="/v1")
    app.include_router(embedder_test.router, prefix="/v1")
    return app
```

- [ ] **Step 5: Run tests — expect pass**

```bash
.venv/bin/pytest tests/test_embedder_test_route.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/server/routes/embedder_test.py semanticsd/server/app.py tests/test_embedder_test_route.py
git commit -m "feat(server): POST /v1/embedder/test endpoint"
```

---

## Task 13: `/v1/health` upgrade — exercise configured embedder (TDD)

**Files:**
- Modify: `semanticsd/server/routes/health.py`
- Modify: `tests/test_health.py`

- [ ] **Step 1: Update health tests**

Add to `tests/test_health.py`:

```python
def test_health_reports_embedder_when_unconfigured(monkeypatch):
    """If no preset is configured, embedder section reports 'unconfigured'."""
    monkeypatch.setattr(auth, "_token_cache", "secret")
    from semanticsd.embedders import reset_active_embedder
    reset_active_embedder()
    # Default config has preset="local" — but we want the unconfigured path.
    # Patch get_active_embedder to return None.
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: None)
    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    assert r.json()["embedder"]["ok"] is False
    assert "not configured" in r.json()["embedder"]["message"].lower()


def test_health_reports_embedder_when_configured(monkeypatch):
    """If an embedder is configured, surface its provider/model/dim."""
    monkeypatch.setattr(auth, "_token_cache", "secret")

    class FakeEmb:
        provider_id = "fake"
        model_id = "fake-1"
        dim = 384
        def health_check(self):
            return (True, "fake ok")

    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: FakeEmb())

    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["embedder"]["ok"] is True
    assert body["embedder"]["provider_id"] == "fake"
    assert body["embedder"]["model_id"] == "fake-1"
    assert body["embedder"]["dim"] == 384
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_health.py::test_health_reports_embedder_when_unconfigured tests/test_health.py::test_health_reports_embedder_when_configured -v
```

- [ ] **Step 3: Update `semanticsd/server/routes/health.py`**

Replace the entire file with:

```python
"""Health endpoint."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from semanticsd import __version__
from semanticsd.server.auth import require_token
from semanticsd.db import connection
from semanticsd import paths
from semanticsd import embedders as emb_pkg

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

    embedder_section: dict
    try:
        embedder = emb_pkg.get_active_embedder()
    except Exception as e:
        embedder = None
        embedder_section = {"ok": False, "message": f"embedder build failed: {e}"}
    else:
        if embedder is None:
            embedder_section = {"ok": False, "message": "embedder not configured"}
        else:
            ok, msg = embedder.health_check()
            embedder_section = {
                "ok": ok,
                "message": msg,
                "provider_id": embedder.provider_id,
                "model_id": embedder.model_id,
                "dim": embedder.dim,
            }

    overall_ok = db_ok and embedder_section["ok"]
    return {
        "status": "ok" if overall_ok else "degraded",
        "version": __version__,
        "doc_count": doc_count,
        "vector_store": {"ok": db_ok},
        "embedder": embedder_section,
    }
```

- [ ] **Step 4: Update older health tests that relied on the old "Plan 2" message**

In `tests/test_health.py`, the existing `test_health_authenticated` and `test_smoke_install_then_query` (in `test_e2e_smoke.py`) may have asserted `"not configured (Plan 2)"` literally. Update any such assertions to be flexible:

In `tests/test_health.py::test_health_authenticated` — replace the entire test body so it (a) loosens the assertion and (b) mocks `get_active_embedder` so the test does NOT trigger a real model download:

```python
def test_health_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")

    class FakeEmb:
        provider_id = "fake"
        model_id = "fake-1"
        dim = 384
        def health_check(self):
            return (True, "fake ok")

    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: FakeEmb())

    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "doc_count" in body
    assert "embedder" in body
```

The other health tests (`test_health_unauthenticated_rejected`, `test_openapi_docs_available`, `test_cors_allows_localhost`) do not need this mock because they don't hit the GET /v1/health code path (auth rejects before route, OpenAPI is a different endpoint, CORS is preflight OPTIONS).

In `tests/test_e2e_smoke.py::test_smoke_install_then_query`, the assertion `assert body["status"] == "ok"` will now fail because the active embedder hasn't been mocked and real LocalEmbedder will try to download a model. Change that test to mock the embedder:

```python
def test_smoke_install_then_query(tmp_app_support, monkeypatch):
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    monkeypatch.setattr(auth, "_token_cache", "smoke-token")
    import semanticsd.embedders as emb_pkg

    class FakeEmb:
        provider_id = "fake"
        model_id = "fake-1"
        dim = 384
        def health_check(self):
            return (True, "fake ok")

    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: FakeEmb())

    app = server_app.create_app()
    client = TestClient(app)

    r = client.get("/v1/health", headers={"X-Auth-Token": "smoke-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["doc_count"] == 0
    assert body["version"]

    r = client.get("/v1/health", headers={"X-Auth-Token": "wrong"})
    assert r.status_code == 401
```

- [ ] **Step 5: Run all health-related tests — expect pass**

```bash
.venv/bin/pytest tests/test_health.py tests/test_e2e_smoke.py -v
```
Expected: 6 passed (4 prior in test_health + 2 new) plus 1 e2e smoke = 7.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/pytest -q
```
Expected: 64 passed.

- [ ] **Step 7: Commit**

```bash
git add semanticsd/server/routes/health.py tests/test_health.py tests/test_e2e_smoke.py
git commit -m "feat(server): /v1/health exercises configured embedder"
```

---

## Task 14: CLI — `ssearch --presets` and `ssearch --test-embedder`

**Files:**
- Modify: `semanticsd/cli.py`

- [ ] **Step 1: Add the two flags to the ssearch_root callback**

Edit `semanticsd/cli.py` — replace the `ssearch_root` callback with:

```python
@ssearch_app.callback(invoke_without_command=True)
def ssearch_root(
    ctx: typer.Context,
    status: bool = typer.Option(False, "--status", help="Show daemon status."),
    presets: bool = typer.Option(False, "--presets", help="List available embedder presets."),
    test_embedder: str = typer.Option(
        "", "--test-embedder", metavar="PRESET",
        help="Round-trip test the embedder for the given preset.",
    ),
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
            emb = body.get("embedder", {})
            typer.echo(f"embedder:  {emb.get('message','')}")
            if emb.get("provider_id"):
                typer.echo(f"  provider: {emb['provider_id']}")
                typer.echo(f"  model:    {emb['model_id']}")
                typer.echo(f"  dim:      {emb['dim']}")
        return

    if presets:
        try:
            with _client() as c:
                r = c.get("/v1/presets")
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            for preset_id, info in body["presets"].items():
                key_flag = "(needs API key)" if info.get("needs_api_key") else ""
                url_flag = "(needs base URL)" if info.get("needs_base_url") else ""
                model = info.get("default_model") or "<user-pick>"
                typer.echo(f"  {preset_id:<20} model={model} {key_flag} {url_flag}".rstrip())
        return

    if test_embedder:
        body_req = {"preset": test_embedder}
        try:
            with _client() as c:
                r = c.post("/v1/embedder/test", json=body_req)
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            if body["ok"]:
                typer.echo(f"OK  preset={test_embedder}")
                typer.echo(f"  provider: {body['provider_id']}")
                typer.echo(f"  model:    {body['model_id']}")
                typer.echo(f"  dim:      {body['dim']}")
                typer.echo(f"  latency:  {body['latency_ms']}ms")
            else:
                typer.echo(f"FAIL preset={test_embedder}: {body.get('error','unknown error')}")
                raise typer.Exit(4)
        return

    if ctx.invoked_subcommand is None:
        typer.echo("Usage: ssearch [QUERY] | --status | --presets | --test-embedder PRESET")
        typer.echo("Search subcommands land in Plan 5.")
        raise typer.Exit(0)
```

- [ ] **Step 2: Verify both apps still import**

```bash
.venv/bin/python -c "from semanticsd.cli import semanticsd_app, ssearch_app; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Verify help output**

```bash
.venv/bin/ssearch --help
```
Expected: Includes `--status`, `--presets`, `--test-embedder`.

- [ ] **Step 4: Run full suite — no regressions**

```bash
.venv/bin/pytest -q
```
Expected: 64 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/cli.py
git commit -m "feat(cli): ssearch --presets + --test-embedder"
```

---

## Task 15: End-to-end vec round-trip (TDD with real LocalEmbedder)

**Files:**
- Create: `tests/test_e2e_vec.py`

This test downloads the real `BAAI/bge-small-en-v1.5` model on first run (~30MB, cached afterward at `~/Library/Application Support/semanticsd/models/`). Subsequent runs are fast. It is the integration smoke for the embedder + sqlite-vec pipeline.

- [ ] **Step 1: Write the test**

`tests/test_e2e_vec.py`:
```python
"""End-to-end: real LocalEmbedder + sqlite-vec round-trip.

Downloads BAAI/bge-small-en-v1.5 on first run (~30MB).
Marked slow so a developer can skip via `pytest -m "not slow"`.
"""
import struct
import pytest
from semanticsd.db import connection, migrations
from semanticsd.embedders.local import LocalEmbedder


pytestmark = pytest.mark.slow


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def test_local_embedder_round_trip_through_vec_embeddings(tmp_path, tmp_app_support):
    """Embed real text, store it via sqlite-vec, query nearest, retrieve self."""
    db = tmp_path / "vec.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    embedder = LocalEmbedder()
    docs = [
        "The alpha protocol authenticates two parties via a shared secret.",
        "Pasta carbonara is a Roman dish with egg and pancetta.",
        "The beta release improved indexing throughput via batching.",
    ]
    out = embedder.embed(docs, kind="doc")
    assert len(out.vectors) == 3
    assert all(len(v) == 384 for v in out.vectors)

    for i, v in enumerate(out.vectors, start=1):
        conn.execute(
            "INSERT INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
            (i, _vec_to_blob(v)),
        )

    query_vec = embedder.embed(["how does authentication work"], kind="query").vectors[0]
    rows = conn.execute(
        "SELECT rowid, distance FROM vec_embeddings WHERE embedding MATCH ? "
        "ORDER BY distance LIMIT 3",
        (_vec_to_blob(query_vec),),
    ).fetchall()
    assert len(rows) == 3
    # Nearest neighbour must be doc #1 (the auth one).
    nearest_id = rows[0][0]
    assert nearest_id == 1, f"expected nearest=1 (auth doc), got {nearest_id}"
```

- [ ] **Step 2: Register the `slow` marker**

Edit `pyproject.toml` to add the `slow` marker — append to `[tool.pytest.ini_options]`:
```toml
markers = ["slow: marks tests that download models or hit the network"]
```

(If the section already exists with other entries, append `markers` to it.)

- [ ] **Step 3: Run the test — expect pass (downloads model on first run)**

```bash
.venv/bin/pytest tests/test_e2e_vec.py -v -m slow
```
Expected: 1 passed (may take 30s+ on first run for the model download). On subsequent runs it should be under 10s.

- [ ] **Step 4: Run the full suite excluding slow tests — should still be 64**

```bash
.venv/bin/pytest -q -m "not slow"
```
Expected: 64 passed.

- [ ] **Step 5: Run full suite including slow — should be 65**

```bash
.venv/bin/pytest -q
```
Expected: 65 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_e2e_vec.py pyproject.toml
git commit -m "test(e2e): real LocalEmbedder + sqlite-vec round-trip"
```

---

## Task 16: Manual install verification (controller-driven)

This task is the human-driven smoke of the full Plan-2 stack. No code change. Run from the repo root.

- [ ] **Step 1: Reinstall the venv with the new deps**

```bash
.venv/bin/pip install -r requirements.txt --upgrade
```

- [ ] **Step 2: Re-bootstrap the daemon**

```bash
./scripts/install.sh --uninstall
./scripts/install.sh
```
Expected: clean install, "ready." printed.

- [ ] **Step 3: Verify presets endpoint works**

```bash
.venv/bin/ssearch --presets
```
Expected: lines for `local`, `ollama`, `openai`, `lmstudio`, `vllm`, `openai_compatible`, `custom`.

- [ ] **Step 4: Test the local embedder**

```bash
.venv/bin/ssearch --test-embedder local
```
Expected: `OK preset=local`, provider=local, model=BAAI/bge-small-en-v1.5, dim=384, latency in ms. (First run downloads the model; later runs are fast.)

- [ ] **Step 5: Verify health surfaces the embedder**

```bash
.venv/bin/ssearch --status
```
Expected: status: ok, embedder: local model loaded, provider: local, model: BAAI/bge-small-en-v1.5, dim: 384.

- [ ] **Step 6: Test sandbox harness (interrupted run)**

In one terminal:
```bash
make dev-sandbox
```
Expected: foreground daemon prints uvicorn startup. The sandbox config has `directories=["./sandbox"]` and `SEMANTICSD_HOME=./sandbox/.semanticsd`. (Note: file watcher arrives in Plan 4, so nothing is actually indexed — but the daemon should boot cleanly with the sandbox config.)

In a second terminal:
```bash
SEMANTICSD_HOME=$(pwd)/sandbox/.semanticsd .venv/bin/ssearch --status
```
Expected: status: ok against the sandboxed daemon.

Stop the foreground daemon with Ctrl+C.

- [ ] **Step 7: Confirm no regression in the launchd-managed daemon**

```bash
launchctl list | grep semanticsd
.venv/bin/ssearch --status
```
Expected: launchd agent still running (this is the `~/Library/...` install from Step 2), responds to `--status`.

If any step fails, fix and re-run. Do not consider Plan 2 complete until all seven steps pass.

---

## What's Next (Plan 2.5 + Plan 3 preview)

**Plan 2.5 (small follow-up):** add native SDK files for the providers that need their own quirks: `semanticsd/embedders/voyage.py`, `cohere.py`, `gemini.py`, `vertex.py`, `bedrock.py`. Each is a one-task plan: install SDK, write embedder class with `input_type` / batching / signing handled, register, test with mocked SDK. Order driven by user demand.

**Plan 3:** Document pipeline. Extractors (text/code/markdown/PDF/DOCX/XLSX/PPTX/HTML/EPUB/RTF/email/notebook/archive/images via OCR+CLIP/audio via Whisper/video via ffmpeg+Whisper). Sliding-window chunker + content hasher. Resumable job queue. `POST /v1/index` endpoint. After Plan 3 you can manually `ssearch --index <path>` and watch chunks + embeddings flow into SQLite.

**Plans 4-6:** File watcher + power modes; search engine (3 modes); cost tracking + budgets.
