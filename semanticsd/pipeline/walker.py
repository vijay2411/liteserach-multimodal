"""Directory walker that yields indexable file paths."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator
from semanticsd.pipeline.ignore import IgnoreMatcher


def walk_indexable(root: Path, matcher: IgnoreMatcher, max_size_mb: int) -> Iterator[Path]:
    """Yield each file under `root` that:
      - is a regular file
      - is not matched by `matcher`
      - is at or below `max_size_mb` bytes
    Walks recursively; ignored directories are pruned.
    """
    root = root.resolve()
    max_bytes = max_size_mb * 1024 * 1024
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        # Prune ignored directories in-place so os.walk skips descending.
        rel_dirs = []
        for name in list(dirnames):
            sub = (d / name).relative_to(root)
            # Check both without and with trailing slash (gitignore dir patterns use "dir/")
            sub_str = str(sub).replace("\\", "/")
            if matcher.is_ignored(sub) or matcher._spec.match_file(sub_str + "/"):
                continue
            rel_dirs.append(name)
        dirnames[:] = rel_dirs

        for name in filenames:
            f = d / name
            rel = f.relative_to(root)
            if matcher.is_ignored(rel):
                continue
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            yield f
