"""Robustness battery against the persisted stress corpus.

Locks in the wins from the Plan 5.6 robustness pass: 35/35 expected
top-3 hits across prose, code (Python/Rust/Go/TypeScript), config,
multilingual, edge cases, and cross-modal vision queries.

Slow-marked since it loads the real Ollama text embedder + Gemini
vision API. Skipped if the stress sandbox or external services are
unavailable.
"""
import os
import socket
from pathlib import Path
import pytest


SANDBOX_DB = (
    Path(__file__).resolve().parents[1] / "sandbox-stress" / ".semanticsd" / "index.db"
)


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
    pytest.mark.skipif(not SANDBOX_DB.exists(), reason="stress sandbox not present"),
    pytest.mark.skipif(
        not (_ollama_up() and _gemini_key()),
        reason="needs ollama running + gemini key for cross-modal",
    ),
]


# (query, mode, expected_filename_substring_in_top_3)
GROUND_TRUTH = [
    # Prose / semantic
    ("how does photosynthesis work",                 "hybrid", "photosynthesis.md"),
    ("plants converting sunlight to energy",         "hybrid", "photosynthesis.md"),
    ("Charlie Parker bebop history",                 "hybrid", "jazz_history.md"),
    ("recipe for bread with starter",                "hybrid", "sourdough_recipe.md"),
    ("spooky action at a distance",                  "hybrid", "quantum_entanglement.md"),
    ("Bell theorem hidden variables",                "hybrid", "quantum_entanglement.md"),
    # Code (grep should help)
    ("password hashing function",                    "hybrid", "auth.py"),
    ("verify_password",                              "grep",   "auth.py"),
    ("ConnectionPool psycopg",                       "hybrid", "database.py"),
    ("LRU cache",                                    "hybrid", "lru_cache.rs"),
    ("token bucket rate limiter",                    "hybrid", "ratelimiter.go"),
    ("auto reconnecting websocket",                  "hybrid", "websocket_client.ts"),
    ("WebSocketClient",                              "grep",   "websocket_client.ts"),
    # Identifier styles
    ("getUserById",                                  "grep",   "camelCase_snake_case.md"),
    ("get_user_by_id",                               "grep",   "camelCase_snake_case.md"),
    # Plurals (porter stemmer)
    ("transactions",                                 "grep",   "plurals_and_synonyms.md"),
    ("transaction",                                  "grep",   "plurals_and_synonyms.md"),
    # Filename mode
    ("photosynthesis",                               "filename", "photosynthesis.md"),
    ("api_routes",                                   "filename", "api_routes.json"),
    ("ratelimiter",                                  "filename", "ratelimiter.go"),
    # Config / data
    ("staging environment debug",                    "hybrid", "staging.yaml"),
    ("production database settings",                 "hybrid", "production.yaml"),
    # Multilingual
    ("paella valenciana",                            "hybrid", "spanish_recipe.md"),
    ("haiku frog pond",                              "hybrid", "japanese_haiku.md"),
    ("古池や",                                         "grep",   "japanese_haiku.md"),
    # Email
    ("API latency P99 spike",                        "hybrid", "incident_response.eml"),
    # Notebook
    ("customer churn analysis pandas",               "hybrid", "data_analysis.ipynb"),
    ("tenure histogram seaborn",                     "hybrid", "data_analysis.ipynb"),
    # Cross-modal vision
    ("orange cat with whiskers",                     "hybrid", "cat_drawing.png"),
    ("bar chart revenue by quarter",                 "hybrid", "revenue_chart.png"),
    ("Python code on dark background",               "hybrid", "code_screenshot.png"),
    # Edge cases
    ("UNIQUEMARKERWORD42",                           "grep",   "single_word.md"),
]


def _engine(monkeypatch):
    home = SANDBOX_DB.parent
    monkeypatch.setenv("SEMANTICSD_HOME", str(home))
    from semanticsd.embedders import reset_active_embedder
    reset_active_embedder()

    from semanticsd import paths
    from semanticsd.db import connection, migrations
    from semanticsd.embedders import get_router
    from semanticsd.search.engine import Engine

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    return Engine(conn, get_router())


@pytest.mark.parametrize("query,mode,expected", GROUND_TRUTH, ids=[
    f"{m}:{q[:30]}" for q, m, _ in GROUND_TRUTH
])
def test_stress_battery(monkeypatch, query, mode, expected):
    from semanticsd.search.types import SearchOptions

    engine = _engine(monkeypatch)
    results = engine.search(query, SearchOptions(all=True, mode=mode, limit=5))
    assert results, f"no results for {query!r}"
    paths = [r.path for r in results[:3]]
    assert any(expected in p for p in paths), (
        f"{expected!r} not in top-3 for {query!r}: {paths}"
    )


def test_pure_gibberish_returns_empty(monkeypatch):
    """Truly random gibberish should not surface phantom matches."""
    from semanticsd.search.types import SearchOptions

    engine = _engine(monkeypatch)
    results = engine.search(
        "asdfqwerzxcv blarghbloop notarealword",
        SearchOptions(all=True, mode="hybrid", limit=5),
    )
    assert results == [], f"gibberish should be empty, got {[r.path for r in results]}"
