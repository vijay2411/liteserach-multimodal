"""Bearer-token auth dependency for FastAPI routes."""
from __future__ import annotations
from fastapi import Header, HTTPException, status
from semanticsd import keychain

_token_cache: str | None = None


def get_expected_token() -> str:
    """Lazy-load the token from Keychain on first use; cache thereafter."""
    global _token_cache
    if _token_cache is None:
        _token_cache = keychain.get_auth_token()
        if not _token_cache:
            raise RuntimeError(
                "No SemanticsD auth token in Keychain. Run `semanticsd install` first."
            )
    return _token_cache


def reload_token() -> None:
    """Force re-read from Keychain (used after rotation)."""
    global _token_cache
    _token_cache = None


def require_token(x_auth_token: str | None = Header(default=None)) -> None:
    expected = get_expected_token()
    if not x_auth_token or x_auth_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing X-Auth-Token",
        )
