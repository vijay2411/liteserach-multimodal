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
