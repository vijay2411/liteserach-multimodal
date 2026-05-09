"""POST /v1/reembed endpoint tests."""
from fastapi.testclient import TestClient
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.server import app as server_app
from semanticsd.server import auth
from semanticsd import paths


class _Text(Embedder):
    provider_id = "ollama"; model_id = "embeddinggemma"; dim = 768
    supports_kind = False; cost_per_million_input_tokens_usd = 0.0
    def embed(self, texts, kind):
        return EmbedResult(vectors=[[0.0]*768 for _ in texts], input_tokens=1)
    def health_check(self): return (True, "ok")
    def estimate_tokens(self, texts): return 1


def _setup(tmp_app_support, monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    # Seed two text chunks, neither with embeddings from _Text.
    conn.execute("INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
                 "VALUES('/x', 1, 1, 'text', 1)")
    fid = int(conn.execute("SELECT id FROM files").fetchone()[0])
    for i in range(2):
        conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, content_hash, "
            "byte_start, byte_end, modality) VALUES (?, ?, ?, ?, 0, ?, 'text')",
            (fid, i, f"chunk {i}", f"h{i}", 16),
        )
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_router", lambda **kw: EmbedderRouter(text=_Text()))
    return conn


def test_reembed_queues_jobs(tmp_app_support, monkeypatch):
    conn = _setup(tmp_app_support, monkeypatch)
    client = TestClient(server_app.create_app())
    r = client.post("/v1/reembed",
                    headers={"X-Auth-Token": "secret"},
                    json={"modality": "text"})
    assert r.status_code == 200
    body = r.json()
    assert body["queued"]["text"] == 2
    assert body["total"] == 2

    n = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert n == 2


def test_reembed_unknown_modality_400(tmp_app_support, monkeypatch):
    _setup(tmp_app_support, monkeypatch)
    client = TestClient(server_app.create_app())
    r = client.post("/v1/reembed",
                    headers={"X-Auth-Token": "secret"},
                    json={"modality": "audio"})
    assert r.status_code == 400


def test_reembed_unauthed():
    client = TestClient(server_app.create_app())
    r = client.post("/v1/reembed", json={"modality": "all"})
    assert r.status_code == 401
