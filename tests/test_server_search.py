"""GET /v1/search smoke tests with mocked router."""
import struct
import pytest
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth
from semanticsd.embedders.base import Embedder, EmbedResult


class _FakeText(Embedder):
    provider_id = "fake_t"
    model_id = "ft"
    dim = 768
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def embed(self, texts, kind):
        return EmbedResult(
            vectors=[[float(len(t))] + [0.0] * 767 for t in texts],
            input_tokens=1,
        )

    def health_check(self):
        return (True, "ok")

    def estimate_tokens(self, texts):
        return 1


class _FakeRouter:
    def __init__(self, text=None, vision=None):
        self.text = text
        self.vision = vision

    def get(self, modality):
        if modality == "text":
            return self.text
        if modality == "vision":
            return self.vision
        return None


def _vec_blob(values):
    return struct.pack(f"{len(values)}f", *values)


def _seed(tmp_app_support):
    from semanticsd.db import connection, migrations
    from semanticsd import paths
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    conn.execute(
        "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (1, '/tmp/cwd/notes.md', 1, 1, 'text', 1)"
    )
    conn.execute("INSERT INTO fts_paths(rowid, path) VALUES (1, '/tmp/cwd/notes.md')")
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, "
        "byte_start, byte_end, modality) "
        "VALUES (1, 1, 0, 'fox jumps over the lazy dog', 'h', 0, 27, 'text')"
    )
    conn.execute("INSERT INTO fts_chunks(rowid, text) VALUES (1, 'fox jumps over the lazy dog')")
    conn.execute(
        "INSERT INTO vec_text_embeddings(rowid, embedding) VALUES (1, ?)",
        (_vec_blob([float(27)] + [0.0] * 767),),
    )


def test_search_unauthed(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.get("/v1/search?q=fox")
    assert r.status_code == 401


def test_search_returns_results(tmp_app_support, monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(
        emb_pkg, "get_router",
        lambda **kw: _FakeRouter(text=_FakeText()),
    )
    _seed(tmp_app_support)

    client = TestClient(server_app.create_app())
    r = client.get(
        "/v1/search",
        params={"q": "fox", "all": "true", "mode": "hybrid"},
        headers={"X-Auth-Token": "secret"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert "took_ms" in body
    assert len(body["results"]) >= 1
    assert body["results"][0]["path"] == "/tmp/cwd/notes.md"


def test_search_unknown_mode_400(tmp_app_support, monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(
        emb_pkg, "get_router", lambda **kw: _FakeRouter(text=_FakeText())
    )
    _seed(tmp_app_support)
    client = TestClient(server_app.create_app())
    r = client.get(
        "/v1/search",
        params={"q": "x", "mode": "bogus"},
        headers={"X-Auth-Token": "secret"},
    )
    assert r.status_code == 400
