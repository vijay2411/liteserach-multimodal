"""Keychain wrapper for the SemanticsD auth token and provider API keys."""
from __future__ import annotations
import secrets
import keyring

SERVICE = "semanticsd"
TOKEN_ACCOUNT = "api_token"


def set_auth_token(token: str) -> None:
    keyring.set_password(SERVICE, TOKEN_ACCOUNT, token)


def get_auth_token() -> str | None:
    return keyring.get_password(SERVICE, TOKEN_ACCOUNT)


def generate_or_get_auth_token() -> str:
    """Return existing token, or generate and store a new one."""
    existing = get_auth_token()
    if existing:
        return existing
    new_token = secrets.token_urlsafe(32)
    set_auth_token(new_token)
    return new_token


def delete_auth_token() -> None:
    try:
        keyring.delete_password(SERVICE, TOKEN_ACCOUNT)
    except keyring.errors.PasswordDeleteError:
        pass


def set_provider_key(provider_id: str, api_key: str) -> None:
    keyring.set_password(SERVICE, provider_id, api_key)


def get_provider_key(provider_id: str) -> str | None:
    return keyring.get_password(SERVICE, provider_id)


def delete_provider_key(provider_id: str) -> None:
    try:
        keyring.delete_password(SERVICE, provider_id)
    except keyring.errors.PasswordDeleteError:
        pass
