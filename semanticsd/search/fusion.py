"""Reciprocal Rank Fusion for combining heterogeneous ranking signals."""
from __future__ import annotations
from collections import defaultdict
from semanticsd.search.types import SearchResult


RRF_K = 60


def reciprocal_rank_fusion(
    rank_lists: list[list[SearchResult]],
    k: int = RRF_K,
    limit: int = 20,
) -> list[SearchResult]:
    """Fuse multiple ranked lists into one via RRF.

    score(d) = sum_i 1 / (k + rank_i(d) + 1)

    Documents are deduped by `(file_id, chunk_id)` so the same chunk
    appearing in multiple input lists boosts its score rather than
    appearing twice. The result inherits the highest-scored representation
    (first appearance wins for snippet/metadata) but updates `mode='hybrid'`
    and records contributing modes in metadata.
    """
    scores: dict[tuple[int, int | None], float] = defaultdict(float)
    contributing: dict[tuple[int, int | None], list[str]] = defaultdict(list)
    representative: dict[tuple[int, int | None], SearchResult] = {}

    for ranks in rank_lists:
        for i, r in enumerate(ranks):
            key = (r.file_id, r.chunk_id)
            scores[key] += 1.0 / (k + i + 1)
            contributing[key].append(r.mode)
            if key not in representative:
                representative[key] = r

    fused: list[SearchResult] = []
    for key, score in sorted(scores.items(), key=lambda x: -x[1]):
        rep = representative[key]
        modes = contributing[key]
        fused.append(rep.model_copy(update={
            "score": score,
            "mode": "hybrid",
            "metadata": {**rep.metadata, "contributing_modes": modes},
        }))
    return fused[:limit]
