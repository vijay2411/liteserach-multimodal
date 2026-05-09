"""Search engine — orchestrates per-mode searches, applies CWD filter, fuses."""
from __future__ import annotations
import logging
import sqlite3
from collections import OrderedDict
from pathlib import Path
from semanticsd.embedders.router import EmbedderRouter
from semanticsd.search.types import SearchResult, SearchOptions
from semanticsd.search.filename import search_filename
from semanticsd.search.grep import search_grep
from semanticsd.search.semantic import search_semantic_text, search_semantic_vision
from semanticsd.search.fusion import reciprocal_rank_fusion
from semanticsd.search.snippets import extract_snippet, tokenize_query

log = logging.getLogger(__name__)

OVERFETCH = 3  # over-fetch this multiple before applying CWD filter


class _LRU:
    """Tiny LRU keyed by hashable. Used for embedded-query caching."""

    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._store: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


class Engine:
    def __init__(self, conn: sqlite3.Connection, router: EmbedderRouter):
        self.conn = conn
        self.router = router
        self._cache_text = _LRU()
        self._cache_vision = _LRU()

    def search(self, query: str, opts: SearchOptions | None = None) -> list[SearchResult]:
        opts = opts or SearchOptions()
        query = (query or "").strip()
        if not query:
            return []

        fetch_limit = opts.limit * OVERFETCH

        rank_lists: list[list[SearchResult]] = []
        mode_labels: list[str] = []

        if opts.mode in ("filename", "hybrid"):
            try:
                hits = search_filename(self.conn, query, limit=fetch_limit)
                rank_lists.append(hits)
                mode_labels.append("filename")
            except Exception as e:
                log.warning("filename search failed: %s", e)

        if opts.mode in ("grep", "hybrid"):
            try:
                hits = search_grep(self.conn, query, limit=fetch_limit)
                rank_lists.append(hits)
                mode_labels.append("grep")
            except Exception as e:
                log.warning("grep search failed: %s", e)

        if opts.mode in ("semantic", "hybrid"):
            text_hits = self._semantic_text(query, fetch_limit)
            if text_hits:
                rank_lists.append(text_hits)
                mode_labels.append("semantic")
            if opts.vision and self.router.vision is not None:
                vision_hits = self._semantic_vision(query, fetch_limit)
                if vision_hits:
                    rank_lists.append(vision_hits)
                    mode_labels.append("vision")

        if not rank_lists:
            return []

        if opts.mode == "hybrid":
            results = reciprocal_rank_fusion(rank_lists, limit=fetch_limit, mode_labels=mode_labels)
        else:
            results = rank_lists[0]

        results = self._cwd_filter(results, opts)[: opts.limit]
        results = self._enrich_snippets(results, query)
        return results

    def _semantic_text(self, query: str, limit: int) -> list[SearchResult]:
        text_em = self.router.text
        if text_em is None:
            return []
        cache_key = (query, text_em.provider_id, text_em.model_id)
        vec = self._cache_text.get(cache_key)
        if vec is None:
            try:
                res = text_em.embed([query], kind="query")
                vec = res.vectors[0] if res.vectors else []
            except Exception as e:
                log.warning("text embed for query failed: %s", e)
                return []
            self._cache_text.put(cache_key, vec)
        return search_semantic_text(self.conn, vec, limit=limit)

    def _semantic_vision(self, query: str, limit: int) -> list[SearchResult]:
        vision_em = self.router.vision
        if vision_em is None:
            return []
        cache_key = (query, vision_em.provider_id, vision_em.model_id)
        vec = self._cache_vision.get(cache_key)
        if vec is None:
            vec = self._embed_query_for_vision(query, vision_em)
            if not vec:
                return []
            self._cache_vision.put(cache_key, vec)
        return search_semantic_vision(self.conn, vec, dim=vision_em.dim, limit=limit)

    def _embed_query_for_vision(self, query: str, vision_em) -> list[float]:
        """Cross-modal embedders accept text input but the API differs.

        Gemini Embedding 2: text via text part, image via inline_data — already
        works at the GeminiTextEmbedder level. To embed a TEXT query into the
        vision space, we re-use the Gemini text endpoint with the same model
        (the resulting vector lives in the unified text+image space).

        Qwen3-VL local: SentenceTransformer.encode accepts strings directly and
        produces vectors in the unified text+image space.
        """
        from semanticsd.embedders.gemini_vision import GeminiVisionEmbedder
        from semanticsd.embedders.qwen3_vl import LocalQwen3VisionEmbedder

        if isinstance(vision_em, GeminiVisionEmbedder):
            from semanticsd.embedders.gemini import GeminiTextEmbedder
            sibling = GeminiTextEmbedder(api_key=vision_em.api_key, model=vision_em.model_id)
            try:
                res = sibling.embed([query], kind="query")
                return res.vectors[0] if res.vectors else []
            except Exception as e:
                log.warning("gemini cross-modal text embed failed: %s", e)
                return []
        if isinstance(vision_em, LocalQwen3VisionEmbedder):
            try:
                model = vision_em._ensure_model()
                vec = model.encode([query], normalize_embeddings=True)
                return [float(x) for x in vec[0]]
            except Exception as e:
                log.warning("qwen3-vl cross-modal text embed failed: %s", e)
                return []
        log.info("no cross-modal text path for %s; skipping vision search", type(vision_em).__name__)
        return []

    def _cwd_filter(self, results: list[SearchResult], opts: SearchOptions) -> list[SearchResult]:
        if opts.all:
            return results
        cwd = opts.cwd or Path.cwd()
        prefix = str(cwd.resolve())
        if not prefix.endswith("/"):
            prefix = prefix + "/"
        return [r for r in results if r.path == prefix.rstrip("/") or r.path.startswith(prefix)]

    def _enrich_snippets(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        terms = tokenize_query(query)
        out: list[SearchResult] = []
        for r in results:
            if r.snippet:
                trimmed = extract_snippet(r.snippet, query_terms=terms)
                out.append(r.model_copy(update={"snippet": trimmed}))
            else:
                out.append(r)
        return out
