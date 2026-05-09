"""GET /v1/usage tests."""
import time
from fastapi.testclient import TestClient
from semanticsd.db import connection, migrations
from semanticsd.server import app as server_app
from semanticsd.server import auth
from semanticsd import paths


def _seed(tmp_app_support, rows):
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)
    for r in rows:
        conn.execute(
            "INSERT INTO usage(timestamp, provider_id, model_id, operation, "
            "input_tokens, cost_usd, chunk_count, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (r["ts"], r["provider"], r["model"], r["op"], r["tok"], r["cost"],
             r["chunks"], r["dur"]),
        )


def _client(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    return TestClient(server_app.create_app())


def test_usage_default_window_is_this_month(tmp_app_support, monkeypatch):
    now = int(time.time())
    _seed(tmp_app_support, [
        {"ts": now - 100, "provider": "gemini", "model": "g2", "op": "text_embed",
         "tok": 1000, "cost": 0.001, "chunks": 1, "dur": 50},
        {"ts": now - 100, "provider": "gemini", "model": "g2", "op": "text_embed",
         "tok": 2000, "cost": 0.002, "chunks": 2, "dur": 80},
        {"ts": now - 100, "provider": "ollama", "model": "embeddinggemma", "op": "text_embed",
         "tok": 500, "cost": 0.0, "chunks": 1, "dur": 20},
    ])

    client = _client(monkeypatch)
    r = client.get("/v1/usage", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["calls"] == 3
    assert body["chunks"] == 4
    assert body["input_tokens"] == 3500
    assert abs(body["cost_usd"] - 0.003) < 1e-9
    providers = {r["provider_id"] for r in body["by_provider"]}
    assert providers == {"gemini", "ollama"}


def test_usage_provider_filter(tmp_app_support, monkeypatch):
    now = int(time.time())
    _seed(tmp_app_support, [
        {"ts": now - 100, "provider": "a", "model": "m", "op": "text_embed",
         "tok": 100, "cost": 0.001, "chunks": 1, "dur": 10},
        {"ts": now - 100, "provider": "b", "model": "m", "op": "text_embed",
         "tok": 200, "cost": 0.002, "chunks": 1, "dur": 10},
    ])
    client = _client(monkeypatch)
    r = client.get("/v1/usage", params={"provider": "a"},
                   headers={"X-Auth-Token": "secret"})
    body = r.json()
    assert body["calls"] == 1
    assert body["by_provider"][0]["provider_id"] == "a"


def test_usage_unauthed(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.get("/v1/usage")
    assert r.status_code == 401


def test_usage_bad_date_400(tmp_app_support, monkeypatch):
    client = _client(monkeypatch)
    r = client.get("/v1/usage", params={"since": "not-a-date"},
                   headers={"X-Auth-Token": "secret"})
    assert r.status_code == 400


def test_health_includes_budget_block(tmp_app_support, monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    # Make sure the usage table exists before the health endpoint queries it.
    paths.ensure_dirs()
    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    client = TestClient(server_app.create_app())
    r = client.get("/v1/health", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert "budget" in body
    # No spend yet, no cap configured by default → all zeros, not blocked.
    bud = body["budget"]
    assert "spent_this_month_usd" in bud
    assert "limit_usd" in bud
    assert "blocked" in bud
    assert bud["blocked"] is False
