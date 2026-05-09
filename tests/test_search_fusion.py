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


def _r_typed(file_id, chunk_id, mode, file_type, path="/x/foo"):
    return SearchResult(
        path=path, modality="text", mode=mode,
        score=0.5, file_id=file_id, chunk_id=chunk_id,
        metadata={"file_type": file_type},
    )


def test_rrf_weighted_boosts_grep_for_code():
    """For a .py file, grep contribution should outweigh semantic of equal rank."""
    code_grep = [_r_typed(1, 11, "grep", "text", "/x/foo.py")]
    code_semantic = [_r_typed(2, 22, "semantic", "text", "/x/foo.py")]
    out = reciprocal_rank_fusion(
        [code_grep, code_semantic],
        mode_labels=["grep", "semantic"],
    )
    # Both files appear; the one whose mode matches the code profile wins.
    assert out[0].file_id == 1


def test_rrf_weighted_excludes_grep_for_image():
    """For images, grep weight is 0 — grep results vanish from the fused output."""
    img_grep = [_r_typed(1, 11, "grep", "image", "/x/foo.png")]
    img_filename = [_r_typed(2, 22, "filename", "image", "/x/bar.png")]
    out = reciprocal_rank_fusion(
        [img_grep, img_filename],
        mode_labels=["grep", "filename"],
    )
    paths = [r.path for r in out]
    assert "/x/foo.png" not in paths
    assert "/x/bar.png" in paths
