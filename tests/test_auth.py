import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from semanticsd.server import auth


def make_app(token: str) -> FastAPI:
    app = FastAPI()
    auth._token_cache = token  # for test only

    @app.get("/protected", dependencies=[Depends(auth.require_token)])
    def protected():
        return {"ok": True}

    return app


def test_missing_header_returns_401(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected")
    assert r.status_code == 401


def test_wrong_token_returns_401(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected", headers={"X-Auth-Token": "wrong"})
    assert r.status_code == 401


def test_correct_token_allows(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(make_app("secret"))
    r = client.get("/protected", headers={"X-Auth-Token": "secret"})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
