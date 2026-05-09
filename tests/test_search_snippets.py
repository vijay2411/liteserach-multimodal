"""Snippet extraction."""
from semanticsd.search.snippets import extract_snippet, tokenize_query, MAX_SNIPPET_CHARS


def test_short_text_returned_verbatim():
    assert extract_snippet("hello world") == "hello world"


def test_long_text_truncated():
    text = "x" * 500
    out = extract_snippet(text)
    assert len(out) <= MAX_SNIPPET_CHARS + 3  # plus "..."
    assert out.endswith("...")


def test_snippet_centers_on_query_term():
    text = "padding " * 50 + "MATCH_TARGET" + " padding" * 50
    out = extract_snippet(text, query_terms=["MATCH_TARGET"])
    assert "MATCH_TARGET" in out
    assert out.startswith("...") and out.endswith("...")


def test_image_descriptor_returned_verbatim():
    text = "<image: foo.png page=2>"
    assert extract_snippet(text) == text


def test_tokenize_query():
    assert tokenize_query("hello world!") == ["hello", "world"]
    assert tokenize_query("foo-bar baz") == ["foo", "bar", "baz"]
    assert tokenize_query("a b cd") == ["cd"]  # too short filtered
