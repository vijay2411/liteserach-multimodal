"""Indexer orchestrator: walks/extracts/chunks/hashes, persists files+chunks,
queues jobs. Does NOT call the embedder."""
from __future__ import annotations
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any
from semanticsd.extractors import registry as ext_registry
from semanticsd.pipeline.chunker import SlidingWindowChunker, Chunk
from semanticsd.pipeline.hasher import sha256_hex, sha256_bytes
from semanticsd.pipeline.ignore import IgnoreMatcher
from semanticsd.pipeline.walker import walk_indexable

log = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        conn: sqlite3.Connection,
        max_file_size_mb: int = 50,
        ignore_patterns: list[str] | None = None,
        chunker: SlidingWindowChunker | None = None,
    ):
        self.conn = conn
        self.max_file_size_mb = max_file_size_mb
        self.matcher = IgnoreMatcher(patterns=ignore_patterns, include_defaults=True)
        self.chunker = chunker or SlidingWindowChunker()

    def index_path(self, path: Path) -> dict[str, int]:
        path = path.resolve()
        files_indexed = 0
        files_skipped_unsupported = 0
        files_skipped_unchanged = 0
        chunks_created = 0
        jobs_queued = 0

        if path.is_file():
            paths = [path]
        else:
            paths = list(walk_indexable(path, self.matcher, self.max_file_size_mb))

        for f in paths:
            extractor = ext_registry.get_extractor(f)
            if extractor is None:
                files_skipped_unsupported += 1
                continue
            stats = self._index_one_file(f, extractor)
            if stats is None:
                files_skipped_unchanged += 1
                continue
            files_indexed += 1
            chunks_created += stats["chunks"]
            jobs_queued += stats["jobs"]

        return {
            "files_indexed": files_indexed,
            "files_skipped_unsupported": files_skipped_unsupported,
            "files_skipped_unchanged": files_skipped_unchanged,
            "chunks_created": chunks_created,
            "jobs_queued": jobs_queued,
        }

    def index_inline(self, source: str, content: str, metadata: dict[str, Any] | None = None) -> dict[str, int]:
        now = int(time.time())
        size = len(content.encode("utf-8"))
        existing = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (source,)
        ).fetchone()
        if existing:
            file_id = int(existing[0])
            self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            self.conn.execute(
                "UPDATE files SET modified_at = ?, size = ?, indexed_at = ? WHERE id = ?",
                (now, size, now, file_id),
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
                "VALUES (?, ?, ?, 'inline', ?)",
                (source, now, size, now),
            )
            file_id = int(cur.lastrowid)
        chunks_created, jobs_queued = self._chunk_segment_into_jobs(file_id, content, base_offset=0)
        return {
            "files_indexed": 1,
            "files_skipped_unsupported": 0,
            "files_skipped_unchanged": 0,
            "chunks_created": chunks_created,
            "jobs_queued": jobs_queued,
        }

    def _index_one_file(self, path: Path, extractor) -> dict | None:
        stat = path.stat()
        existing = self.conn.execute(
            "SELECT id, modified_at, size FROM files WHERE path = ?",
            (str(path),),
        ).fetchone()
        now = int(time.time())
        if existing is not None and int(existing[1]) == int(stat.st_mtime) and int(existing[2]) == stat.st_size:
            return None  # unchanged

        try:
            doc = extractor.extract(path)
        except Exception as e:
            log.warning("extractor failed for %s: %s", path, e)
            return None

        if existing is not None:
            file_id = int(existing[0])
            # Clear FTS rows for the chunks we're about to replace.
            old_chunk_ids = [
                int(r[0]) for r in self.conn.execute(
                    "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
                )
            ]
            if old_chunk_ids:
                ph = ",".join("?" for _ in old_chunk_ids)
                self.conn.execute(
                    f"DELETE FROM fts_chunks WHERE rowid IN ({ph})", old_chunk_ids
                )
            self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
            self.conn.execute(
                "UPDATE files SET modified_at = ?, size = ?, file_type = ?, indexed_at = ? WHERE id = ?",
                (int(stat.st_mtime), stat.st_size, doc.file_type, now, file_id),
            )
        else:
            cur = self.conn.execute(
                "INSERT INTO files(path, modified_at, size, file_type, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(path), int(stat.st_mtime), stat.st_size, doc.file_type, now),
            )
            file_id = int(cur.lastrowid)

        # Re-populate FTS path entry on every (re)index.
        self.conn.execute("DELETE FROM fts_paths WHERE rowid = ?", (file_id,))
        self.conn.execute(
            "INSERT INTO fts_paths(rowid, path) VALUES (?, ?)",
            (file_id, str(path)),
        )

        chunks_created = 0
        jobs_queued = 0
        for seg in doc.segments:
            if seg.modality == "vision":
                cc, jq = self._add_vision_chunk(file_id, seg)
            else:
                cc, jq = self._chunk_segment_into_jobs(
                    file_id=file_id, text=seg.text, base_offset=seg.byte_start,
                )
            chunks_created += cc
            jobs_queued += jq
        return {"chunks": chunks_created, "jobs": jobs_queued}

    def _add_vision_chunk(self, file_id: int, seg) -> tuple[int, int]:
        """Insert one chunk + one job for a vision segment (no sub-chunking)."""
        if seg.image_data is None:
            return 0, 0
        chash = sha256_bytes(seg.image_data)
        row = self.conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM chunks WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        next_idx = int(row[0]) + 1
        cur = self.conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, content_hash, byte_start, byte_end, modality, image_blob) "
            "VALUES (?, ?, ?, ?, ?, ?, 'vision', ?)",
            (file_id, next_idx, seg.text, chash, seg.byte_start, seg.byte_end, seg.image_data),
        )
        chunk_id = int(cur.lastrowid)
        now = int(time.time())
        self.conn.execute(
            "INSERT INTO jobs(chunk_id, status, attempts, created_at, updated_at) "
            "VALUES (?, 'pending', 0, ?, ?)",
            (chunk_id, now, now),
        )
        return 1, 1

    def _chunk_segment_into_jobs(self, file_id: int, text: str, base_offset: int) -> tuple[int, int]:
        chunks: list[Chunk] = self.chunker.chunk(text, base_byte_offset=base_offset)
        if not chunks:
            return 0, 0
        row = self.conn.execute(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM chunks WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        next_idx = int(row[0]) + 1

        chunks_created = 0
        jobs_queued = 0
        now = int(time.time())
        for ch in chunks:
            chash = sha256_hex(ch.text)
            cur = self.conn.execute(
                "INSERT INTO chunks(file_id, chunk_index, text, content_hash, byte_start, byte_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (file_id, next_idx, ch.text, chash, ch.byte_start, ch.byte_end),
            )
            chunk_id = int(cur.lastrowid)
            self.conn.execute(
                "INSERT INTO fts_chunks(rowid, text) VALUES (?, ?)",
                (chunk_id, ch.text),
            )
            self.conn.execute(
                "INSERT INTO jobs(chunk_id, status, attempts, created_at, updated_at) "
                "VALUES (?, 'pending', 0, ?, ?)",
                (chunk_id, now, now),
            )
            chunks_created += 1
            jobs_queued += 1
            next_idx += 1
        return chunks_created, jobs_queued
