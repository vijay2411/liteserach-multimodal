"""In-process end-to-end test: app + auth + health, no real launchd."""
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth
from semanticsd.db import connection, migrations
from semanticsd import paths


def test_smoke_install_then_query(tmp_app_support, monkeypatch):
    # Apply migrations the way `install` would.
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    monkeypatch.setattr(auth, "_token_cache", "smoke-token")
    app = server_app.create_app()
    client = TestClient(app)

    r = client.get("/v1/health", headers={"X-Auth-Token": "smoke-token"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["doc_count"] == 0
    assert body["version"]

    # Wrong token rejected:
    r = client.get("/v1/health", headers={"X-Auth-Token": "wrong"})
    assert r.status_code == 401
