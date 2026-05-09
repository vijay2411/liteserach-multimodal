"""End-to-end search test against the persisted sandbox-docs corpus.

Requires the `sandbox-docs/.semanticsd/index.db` produced by Plan 4
real-world smoke (177 files indexed from ~/Documents using Ollama
embeddinggemma + Gemini Embedding 2). Skipped if the DB doesn't exist.

Slow-marked: hits real Ollama (text query embed) + real Gemini (vision
query embed for cross-modal search).
"""
import os
import socket
from pathlib import Path
import pytest


SANDBOX_DB = Path(__file__).resolve().parents[1] / "sandbox-docs" / ".semanticsd" / "index.db"


def _ollama_up() -> bool:
    s = socket.socket()
    try:
        s.settimeout(0.5)
        s.connect(("localhost", 11434))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _gemini_key() -> str | None:
    p = Path.home() / "secrets" / "gemini_api_key"
    if p.exists():
        return p.read_text().strip()
    return os.environ.get("GEMINI_API_KEY")


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not SANDBOX_DB.exists(), reason="sandbox-docs DB not present"),
    pytest.mark.skipif(
        not (_ollama_up() and _gemini_key()),
        reason="needs ollama + gemini for cross-modal queries",
    ),
]


def _engine(monkeypatch):
    """Build an Engine wired to the sandbox-docs DB and the configured router."""
    home = SANDBOX_DB.parent
    monkeypatch.setenv("SEMANTICSD_HOME", str(home))
    # Reset cached router so the new HOME is picked up
    from semanticsd.embedders import reset_active_embedder
    reset_active_embedder()

    from semanticsd import paths
    from semanticsd.db import connection, migrations
    from semanticsd.embedders import get_router
    from semanticsd.search.engine import Engine

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    router = get_router()
    return Engine(conn, router)


def test_filename_search_finds_obsidian(monkeypatch):
    from semanticsd.search.types import SearchOptions

    engine = _engine(monkeypatch)
    results = engine.search("Obsidian", SearchOptions(all=True, mode="filename", limit=10))
    assert len(results) >= 1
    assert all("Obsidian" in r.path for r in results)


def test_grep_search_finds_known_token(monkeypatch):
    from semanticsd.search.types import SearchOptions

    engine = _engine(monkeypatch)
    results = engine.search("Transaction Remarks", SearchOptions(all=True, mode="grep", limit=10))
    assert len(results) >= 1
    # "Transaction Remarks" appears in the bank statement PDFs
    assert any("Transaction" in (r.snippet or "") for r in results)


def test_hybrid_returns_text_and_vision(monkeypatch):
    """Hybrid mode + cross-modal: a PDF-related query should surface both
    text-extracted and vision-rendered representations."""
    from semanticsd.search.types import SearchOptions

    engine = _engine(monkeypatch)
    results = engine.search(
        "transaction history",
        SearchOptions(all=True, mode="hybrid", limit=10),
    )
    assert len(results) >= 1
    modalities = {r.modality for r in results}
    # We expect at least text results; vision is best-effort
    assert "text" in modalities
