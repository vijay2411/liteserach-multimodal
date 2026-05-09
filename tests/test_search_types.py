"""SearchResult / SearchOptions models."""
from pathlib import Path
from semanticsd.search.types import SearchResult, SearchOptions


def test_search_options_defaults():
    o = SearchOptions()
    assert o.mode == "hybrid"
    assert o.limit == 20
    assert o.cwd is None
    assert o.all is False
    assert o.vision is True


def test_search_options_with_cwd():
    o = SearchOptions(cwd=Path("/x"), all=False, mode="semantic")
    assert o.cwd == Path("/x")


def test_search_result_required_fields():
    r = SearchResult(path="/a", modality="text", mode="semantic",
                     score=0.5, file_id=1)
    assert r.metadata == {}
    assert r.chunk_id is None
    assert r.snippet is None
