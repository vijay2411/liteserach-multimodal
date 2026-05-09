"""Reciprocal Rank Fusion."""
from semanticsd.search.fusion import reciprocal_rank_fusion
from semanticsd.search.types import SearchResult


def _r(file_id, chunk_id, mode, score=0.5):
    return SearchResult(
        path=f"/p/{file_id}", modality="text", mode=mode,
        score=score, file_id=file_id, chunk_id=chunk_id,
    )


def test_rrf_unique_lists():
    semantic = [_r(1, 11, "semantic"), _r(2, 22, "semantic")]
    grep = [_r(3, 33, "grep"), _r(4, 44, "grep")]
    out = reciprocal_rank_fusion([semantic, grep])
    assert len(out) == 4
    assert all(r.mode == "hybrid" for r in out)
    # The top-1 from each list (rank 0) should outrank the rank-1 entries.
    top_keys = [(r.file_id, r.chunk_id) for r in out[:2]]
    assert (1, 11) in top_keys and (3, 33) in top_keys


def test_rrf_overlapping_boosts_score():
    """A chunk appearing in two lists should outrank chunks appearing in one."""
    semantic = [_r(1, 11, "semantic"), _r(2, 22, "semantic")]
    grep = [_r(1, 11, "grep"), _r(3, 33, "grep")]
    out = reciprocal_rank_fusion([semantic, grep])
    assert out[0].file_id == 1 and out[0].chunk_id == 11
    assert "semantic" in out[0].metadata["contributing_modes"]
    assert "grep" in out[0].metadata["contributing_modes"]


def test_rrf_respects_limit():
    rs = [[_r(i, i * 10, "semantic") for i in range(1, 30)]]
    out = reciprocal_rank_fusion(rs, limit=5)
    assert len(out) == 5
