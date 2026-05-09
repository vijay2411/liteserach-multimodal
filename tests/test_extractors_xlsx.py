from semanticsd.extractors.xlsx import XlsxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_xlsx


def test_extracts_rows(tmp_path):
    p = make_xlsx(tmp_path)
    out = XlsxExtractor().extract(p)
    assert out.file_type == "xlsx"
    full = "\n".join(s.text for s in out.segments)
    assert "alice" in full
    assert "90" in full


def test_segment_per_sheet(tmp_path):
    p = make_xlsx(tmp_path)
    out = XlsxExtractor().extract(p)
    assert len(out.segments) == 1
    assert out.segments[0].metadata.get("sheet") == "data"


def test_registry_picks_xlsx(tmp_path):
    p = make_xlsx(tmp_path)
    assert isinstance(registry.get_extractor(p), XlsxExtractor)
