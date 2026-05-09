from semanticsd.extractors.html import HtmlExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_html


def test_strips_tags_keeps_text(tmp_path):
    p = make_html(tmp_path)
    out = HtmlExtractor().extract(p)
    text = out.segments[0].text
    assert "Heading" in text
    assert "Paragraph one" in text
    assert "console.log" not in text
    assert "<h1>" not in text


def test_extractor_metadata_has_title(tmp_path):
    p = make_html(tmp_path)
    out = HtmlExtractor().extract(p)
    assert out.metadata.get("title") == "Page"


def test_registry_picks_html(tmp_path):
    p = make_html(tmp_path)
    assert isinstance(registry.get_extractor(p), HtmlExtractor)
