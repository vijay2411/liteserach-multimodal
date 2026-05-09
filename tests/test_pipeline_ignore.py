from pathlib import Path
from semanticsd.pipeline.ignore import IgnoreMatcher


def test_default_patterns_block_dotgit_and_node_modules():
    m = IgnoreMatcher.from_defaults()
    assert m.is_ignored(Path(".git/HEAD"))
    assert m.is_ignored(Path("node_modules/foo/bar.js"))
    assert m.is_ignored(Path(".DS_Store"))
    assert not m.is_ignored(Path("README.md"))


def test_custom_patterns_extend_defaults():
    m = IgnoreMatcher(patterns=["*.tmp", "secrets/"])
    assert m.is_ignored(Path("foo.tmp"))
    assert m.is_ignored(Path("secrets/.env"))
    assert not m.is_ignored(Path("foo.txt"))


def test_load_from_file(tmp_path):
    f = tmp_path / ".semanticsdignore"
    f.write_text("*.bak\n# comment\nbuild/\n")
    m = IgnoreMatcher.from_file(f)
    assert m.is_ignored(Path("foo.bak"))
    assert m.is_ignored(Path("build/x.o"))
    assert not m.is_ignored(Path("foo.txt"))
