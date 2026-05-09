from pathlib import Path
from semanticsd.pipeline.walker import walk_indexable
from semanticsd.pipeline.ignore import IgnoreMatcher


def test_walks_files_skipping_ignored(tmp_path):
    (tmp_path / "keep.txt").write_text("hi")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("//")

    matcher = IgnoreMatcher.from_defaults()
    files = list(walk_indexable(tmp_path, matcher, max_size_mb=50))
    rels = sorted(f.relative_to(tmp_path).as_posix() for f in files)
    assert rels == ["keep.txt"]


def test_skips_oversize_files(tmp_path):
    big = tmp_path / "huge.bin"
    big.write_bytes(b"\x00" * (3 * 1024 * 1024))
    small = tmp_path / "small.txt"
    small.write_text("hi")

    matcher = IgnoreMatcher.from_defaults()
    files = list(walk_indexable(tmp_path, matcher, max_size_mb=1))
    rels = sorted(f.relative_to(tmp_path).as_posix() for f in files)
    assert rels == ["small.txt"]


def test_walks_recursively(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.txt").write_text("hi")
    matcher = IgnoreMatcher.from_defaults()
    files = list(walk_indexable(tmp_path, matcher, max_size_mb=50))
    rels = sorted(f.relative_to(tmp_path).as_posix() for f in files)
    assert rels == ["a/b.txt"]
