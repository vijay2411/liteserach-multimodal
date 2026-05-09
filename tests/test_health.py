from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth


def test_health_unauthenticated_rejected(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 401


class _FakeEmb:
    provider_id = "fake"
    model_id = "fake-1"
    dim = 384

    def health_check(self):
        return (True, "fake ok")


class _FakeRouter:
    def __init__(self, text=None, vision=None):
        self.text = text
        self.vision = vision


def test_health_authenticated(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_router", lambda **kw: _FakeRouter(text=_FakeEmb()))

    app = server_app.create_app()
    client = TestClient(app)
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "doc_count" in body
    assert "embedder" in body
    assert "embedders" in body  # new modality-aware field


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
    """If no text embedder, embedder section reports 'not configured'."""
    monkeypatch.setattr(auth, "_token_cache", "secret")
    from semanticsd.embedders import reset_active_embedder
    reset_active_embedder()
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_router", lambda **kw: _FakeRouter(text=None))
    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["embedder"]["ok"] is False
    assert "not configured" in body["embedder"]["message"].lower()
    # Vision is reported as benign-disabled
    assert body["embedders"]["vision"]["ok"] is True
    assert "disabled" in body["embedders"]["vision"]["message"].lower()


def test_health_reports_embedder_when_configured(monkeypatch):
    """When a text embedder is configured, surface its provider/model/dim."""
    monkeypatch.setattr(auth, "_token_cache", "secret")
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_router", lambda **kw: _FakeRouter(text=_FakeEmb()))

    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["embedder"]["ok"] is True
    assert body["embedder"]["provider_id"] == "fake"
    assert body["embedder"]["model_id"] == "fake-1"
    assert body["embedder"]["dim"] == 384
    # Modality-aware view also present:
    assert body["embedders"]["text"]["provider_id"] == "fake"


def test_health_reports_vision_when_configured(monkeypatch):
    """When both text + vision are configured, both are surfaced."""
    monkeypatch.setattr(auth, "_token_cache", "secret")

    class _FakeVis:
        provider_id = "vfake"
        model_id = "vfake-1"
        dim = 3072

        def health_check(self):
            return (True, "vfake ok")

    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(
        emb_pkg, "get_router",
        lambda **kw: _FakeRouter(text=_FakeEmb(), vision=_FakeVis()),
    )

    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    body = r.json()
    assert body["embedders"]["text"]["provider_id"] == "fake"
    assert body["embedders"]["vision"]["provider_id"] == "vfake"
    assert body["embedders"]["vision"]["dim"] == 3072
