from semanticsd.extractors.rtf import RtfExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_rtf


def test_strips_rtf_control_codes(tmp_path):
    p = make_rtf(tmp_path)
    out = RtfExtractor().extract(p)
    assert out.file_type == "rtf"
    text = out.segments[0].text
    assert "Hello from RTF" in text
    assert "\\rtf1" not in text


def test_registry_picks_rtf(tmp_path):
    p = make_rtf(tmp_path)
    assert isinstance(registry.get_extractor(p), RtfExtractor)
