from semanticsd.extractors.notebook import NotebookExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_ipynb


def test_extracts_cells_as_segments(tmp_path):
    p = make_ipynb(tmp_path)
    out = NotebookExtractor().extract(p)
    assert out.file_type == "notebook"
    assert len(out.segments) == 2
    assert out.segments[0].metadata.get("cell_type") == "markdown"
    assert out.segments[1].metadata.get("cell_type") == "code"
    assert "Notebook Title" in out.segments[0].text
    assert "x = 42" in out.segments[1].text


def test_registry_picks_ipynb(tmp_path):
    p = make_ipynb(tmp_path)
    assert isinstance(registry.get_extractor(p), NotebookExtractor)
