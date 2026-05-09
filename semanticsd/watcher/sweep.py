"""Full-walk indexing — used for the initial sweep on daemon startup
and for periodic reindex in saver mode."""
from __future__ import annotations
import logging
import time
from pathlib import Path
from semanticsd.pipeline.indexer import Indexer

log = logging.getLogger(__name__)


def sweep_directories(indexer: Indexer, directories: list[Path]) -> dict:
    """Index every configured directory once. Returns aggregate stats.

    Cheap on warm caches: the indexer's mtime+size unchanged check
    short-circuits files that haven't changed, so a periodic re-sweep
    walks the tree but doesn't re-extract.
    """
    t0 = time.monotonic()
    totals = {
        "files_indexed": 0,
        "files_skipped_unsupported": 0,
        "files_skipped_unchanged": 0,
        "chunks_created": 0,
        "jobs_queued": 0,
        "directories": 0,
    }
    for d in directories:
        d = Path(d).expanduser()
        if not d.exists():
            log.info("sweep: %s does not exist; skipping", d)
            continue
        try:
            stats = indexer.index_path(d)
        except Exception as e:
            log.warning("sweep failed for %s: %s", d, e)
            continue
        for k in totals:
            if k in stats:
                totals[k] += stats[k]
        totals["directories"] += 1
    totals["elapsed_s"] = round(time.monotonic() - t0, 2)
    log.info("sweep: %s", totals)
    return totals
