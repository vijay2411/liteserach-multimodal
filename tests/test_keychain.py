import pytest
import keyring
from keyring.backend import KeyringBackend
from semanticsd import keychain


class InMemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        self._store.pop((service, username), None)


@pytest.fixture(autouse=True)
def in_memory_keychain(monkeypatch):
    backend = InMemoryKeyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    yield backend


def test_set_and_get_token():
    keychain.set_auth_token("abc123")
    assert keychain.get_auth_token() == "abc123"


def test_get_token_missing_returns_none():
    assert keychain.get_auth_token() is None


def test_generate_or_get_creates_when_missing():
    tok = keychain.generate_or_get_auth_token()
    assert len(tok) >= 32
    # Stable on second call:
    assert keychain.generate_or_get_auth_token() == tok


def test_provider_api_key():
    keychain.set_provider_key("openai", "sk-xxx")
    assert keychain.get_provider_key("openai") == "sk-xxx"
    keychain.delete_provider_key("openai")
    assert keychain.get_provider_key("openai") is None
