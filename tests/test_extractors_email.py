from semanticsd.extractors.email_msg import EmailExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_eml


def test_extracts_body_and_subject(tmp_path):
    p = make_eml(tmp_path)
    out = EmailExtractor().extract(p)
    assert out.file_type == "email"
    text = out.segments[0].text
    assert "Hello Bob" in text
    assert out.metadata.get("subject") == "A test message"
    assert out.metadata.get("from") == "alice@example.com"


def test_registry_picks_eml(tmp_path):
    p = make_eml(tmp_path)
    assert isinstance(registry.get_extractor(p), EmailExtractor)
