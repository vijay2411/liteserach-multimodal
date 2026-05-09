"""Reciprocal Rank Fusion with per-file-type weights.

Standard RRF treats every input list and every document equally. Here we
apply two modifiers:

1. **Per-list mode tag**: each input list is labelled with the search mode
   that produced it ("semantic", "grep", "filename", "vision"). The mode
   plus the doc's file_type combine to produce a weight via profiles.py.

2. **Vision is its own mode**: vision-modality semantic results are
   labelled "vision" (not "semantic") so the profile can boost them on
   image files and skip them on code.
"""
from __future__ import annotations
from collections import defaultdict
from semanticsd.search.types import SearchResult
from semanticsd.search.profiles import file_class_for, weight_for


RRF_K = 60


def reciprocal_rank_fusion(
    rank_lists: list[list[SearchResult]],
    k: int = RRF_K,
    limit: int = 20,
    mode_labels: list[str] | None = None,
) -> list[SearchResult]:
    """Fuse multiple ranked lists into one via weighted RRF.

    score(d) = Σ_i  weight(file_class(d), mode_i) / (k + rank_i + 1)

    `mode_labels` parallels `rank_lists`. If absent, falls back to each
    list's first-result `.mode` (or "default" if empty), preserving the
    pre-weighted-RRF behaviour.

    Documents are deduped by `(file_id, chunk_id)`. Contributing modes
    are recorded in metadata for display.
    """
    if mode_labels is None:
        mode_labels = []
        for lst in rank_lists:
            mode_labels.append(lst[0].mode if lst else "default")

    scores: dict[tuple[int, int | None], float] = defaultdict(float)
    contributing: dict[tuple[int, int | None], list[str]] = defaultdict(list)
    representative: dict[tuple[int, int | None], SearchResult] = {}

    for ranks, mode_label in zip(rank_lists, mode_labels):
        for i, r in enumerate(ranks):
            key = (r.file_id, r.chunk_id)
            ftype = r.metadata.get("file_type") if r.metadata else None
            cls = file_class_for(ftype, r.path)
            w = weight_for(cls, mode_label)
            if w == 0.0:
                continue
            scores[key] += w / (k + i + 1)
            contributing[key].append(mode_label)
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
