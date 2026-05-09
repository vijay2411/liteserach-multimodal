import pytest
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth
from tests._fixtures import make_text


class FakeEmb:
    provider_id = "fake"
    model_id = "fake-1"
    dim = 384
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def embed(self, texts, kind):
        from semanticsd.embedders.base import EmbedResult
        return EmbedResult(vectors=[[0.1] * 384 for _ in texts], input_tokens=1)

    def health_check(self):
        return (True, "fake ok")

    def estimate_tokens(self, texts):
        return 1


@pytest.fixture
def fake_active(monkeypatch):
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: FakeEmb())


def test_index_unauthed(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post("/v1/index", json={"path": "/tmp"})
    assert r.status_code == 401


def test_index_path(tmp_app_support, monkeypatch, tmp_path, fake_active):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    src = tmp_path / "corpus"
    src.mkdir()
    make_text(src, body="Hello world.")

    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={"path": str(src), "drain": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["files_indexed"] == 1
    assert body["jobs_queued"] >= 1
    assert body["drained"] >= 1


def test_index_inline(tmp_app_support, monkeypatch, fake_active):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={"source": "test://1", "content": "inline text"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["files_indexed"] == 1


def test_index_missing_args_returns_400(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={},
    )
    assert r.status_code == 400
