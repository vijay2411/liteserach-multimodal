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
