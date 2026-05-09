"""Snippet extraction — trim a chunk's text to a window around the query terms."""
from __future__ import annotations
import re

MAX_SNIPPET_CHARS = 240
CONTEXT_HALF = 100


def extract_snippet(text: str, query_terms: list[str] | None = None) -> str:
    """Return a snippet of `text` of at most MAX_SNIPPET_CHARS, biased to
    include the first matching query term if provided.

    For vision chunks (text starts with "<image:"), return as-is — the
    descriptor is already the right length.
    """
    if not text:
        return ""
    if text.startswith("<image:"):
        return text
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= MAX_SNIPPET_CHARS:
        return text

    if query_terms:
        for term in query_terms:
            if not term:
                continue
            idx = text.lower().find(term.lower())
            if idx >= 0:
                start = max(0, idx - CONTEXT_HALF)
                end = min(len(text), idx + len(term) + CONTEXT_HALF)
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                return prefix + text[start:end] + suffix
    return text[:MAX_SNIPPET_CHARS] + "..."


def tokenize_query(query: str) -> list[str]:
    """Pull alphanumeric terms out of a free-form query for snippet biasing."""
    return [t for t in re.findall(r"\w+", query) if len(t) >= 2]
