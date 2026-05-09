"""Token-aware FTS query rewriting + stop-word filter."""
from semanticsd.search.grep import _build_fts_query


def test_or_of_tokens():
    assert _build_fts_query("password hashing function") == "password OR hashing OR function"


def test_stopwords_dropped():
    # "with" / "the" / "and" are filtered; only meaningful tokens remain.
    out = _build_fts_query("the orange cat with whiskers")
    assert "with" not in out.lower().split(" or ")
    assert "the" not in out.lower().split(" or ")
    assert "orange" in out and "cat" in out and "whiskers" in out


def test_two_char_tokens_dropped():
    # "at" / "or" / "is" are 2-char and dropped (FTS5 ignores them anyway).
    out = _build_fts_query("spooky action at a distance")
    assert "at" not in out.lower().split(" or ")
    assert "spooky" in out and "action" in out and "distance" in out


def test_safe_tokens_unquoted_for_stemmer():
    # Bare alphanumeric tokens go unquoted so porter stemmer applies.
    out = _build_fts_query("fox")
    assert out == "fox"


def test_special_chars_quoted():
    # Hyphens, apostrophes etc. force phrase quoting.
    out = _build_fts_query("don't break")
    # "break" is bare, "don't" must be quoted (5+ chars, has apostrophe)
    assert '"' in out


def test_cjk_tokens_pass_through():
    # Python's `\w` includes CJK; FTS5's unicode61 tokenizer handles them.
    # Bare token works — verified end-to-end against real japanese_haiku.md.
    out = _build_fts_query("古池や")
    assert "古池や" in out


def test_empty_input():
    assert _build_fts_query("") == ""
    assert _build_fts_query("   ") == ""


def test_only_stopwords():
    # Nothing meaningful left; falls back to literal phrase.
    out = _build_fts_query("the and with")
    assert out == '"the and with"'
