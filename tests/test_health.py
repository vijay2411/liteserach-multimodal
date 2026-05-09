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
