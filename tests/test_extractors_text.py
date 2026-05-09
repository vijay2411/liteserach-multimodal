from pathlib import Path
from semanticsd.extractors.text import TextExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_text, make_markdown, make_code, make_json, make_yaml, make_csv


def test_extracts_plain_text(tmp_path):
    p = make_text(tmp_path, body="Hello\nWorld")
    out = TextExtractor().extract(p)
    assert out.file_type == "text"
    assert len(out.segments) == 1
    assert out.segments[0].text == "Hello\nWorld"
    assert out.segments[0].byte_start == 0
    assert out.segments[0].byte_end == len("Hello\nWorld".encode())


def test_extracts_markdown(tmp_path):
    p = make_markdown(tmp_path)
    out = TextExtractor().extract(p)
    assert "Title" in out.segments[0].text


def test_extracts_code(tmp_path):
    p = make_code(tmp_path)
    out = TextExtractor().extract(p)
    assert "def greet" in out.segments[0].text


def test_extracts_json(tmp_path):
    p = make_json(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_extracts_yaml(tmp_path):
    p = make_yaml(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_extracts_csv(tmp_path):
    p = make_csv(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_registry_picks_text_for_md(tmp_path):
    p = make_markdown(tmp_path)
    e = registry.get_extractor(p)
    assert isinstance(e, TextExtractor)


def test_registry_picks_text_for_py(tmp_path):
    p = make_code(tmp_path)
    e = registry.get_extractor(p)
    assert isinstance(e, TextExtractor)


def test_handles_invalid_utf8_gracefully(tmp_path):
    p = tmp_path / "bin.txt"
    p.write_bytes(b"\xff\xfe\x00\x00bad")
    out = TextExtractor().extract(p)
    assert isinstance(out.segments[0].text, str)  # decoded with replace
