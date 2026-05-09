"""HTTP /v1/watch and /v1/power endpoints."""
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth


class _FakePC:
    """Just enough of PowerController for the routes' needs."""
    def __init__(self):
        self.mode = "active"
        self.set_mode_calls = []

    def status(self):
        return {
            "mode": self.mode,
            "auto_saver_on_battery": True,
            "power_source": "ac",
            "directories": ["/x"],
            "watcher_running": True,
            "dirty_pending": 0,
            "last_sweep_at": None,
            "saver_interval_s": 3600,
        }

    async def force_sweep(self):
        return {"files_indexed": 7, "elapsed_s": 0.4}

    async def set_mode(self, target, reason="manual"):
        self.set_mode_calls.append((target, reason))
        self.mode = target


def _client(monkeypatch, pc=None):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    app = server_app.create_app(power_controller=pc)
    return TestClient(app)


def test_watch_status_returns_payload(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.get("/v1/watch", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "active"
    assert body["watcher_running"] is True
    assert body["directories"] == ["/x"]


def test_watch_sweep_invokes_force_sweep(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.post("/v1/watch/sweep", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["stats"]["files_indexed"] == 7


def test_watch_503_when_no_pc(monkeypatch):
    client = _client(monkeypatch, pc=None)
    r = client.get("/v1/watch", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 503


def test_power_get(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.get("/v1/power", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "active"
    assert body["power_source"] == "ac"
    assert body["auto_saver_on_battery"] is True


def test_power_set_to_saver(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.post("/v1/power",
                    headers={"X-Auth-Token": "secret"},
                    json={"mode": "saver"})
    assert r.status_code == 200
    assert pc.set_mode_calls == [("saver", "api")]
    assert pc.mode == "saver"


def test_power_set_unknown_mode_400(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.post("/v1/power",
                    headers={"X-Auth-Token": "secret"},
                    json={"mode": "bogus"})
    assert r.status_code == 400


def test_power_unauthed(monkeypatch):
    pc = _FakePC()
    client = _client(monkeypatch, pc)
    r = client.get("/v1/power")
    assert r.status_code == 401
