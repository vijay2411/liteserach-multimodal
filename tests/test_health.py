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
