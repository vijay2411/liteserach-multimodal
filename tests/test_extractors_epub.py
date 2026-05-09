from semanticsd.extractors.epub import EpubExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_epub


def test_extracts_chapters_as_segments(tmp_path):
    p = make_epub(tmp_path)
    out = EpubExtractor().extract(p)
    assert out.file_type == "epub"
    full = "\n".join(s.text for s in out.segments)
    assert "Chapter 1" in full
    assert "Lorem ipsum" in full


def test_metadata_has_title(tmp_path):
    p = make_epub(tmp_path)
    out = EpubExtractor().extract(p)
    assert out.metadata.get("title") == "Tiny Book"


def test_registry_picks_epub(tmp_path):
    p = make_epub(tmp_path)
    assert isinstance(registry.get_extractor(p), EpubExtractor)
