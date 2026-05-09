"""gitignore-style path filter, backed by pathspec."""
from __future__ import annotations
from pathlib import Path
import pathspec


DEFAULT_PATTERNS = [
    ".git/", ".svn/", ".hg/",
    "node_modules/", "__pycache__/", "*.pyc",
    "build/", "dist/", "target/",
    ".venv/", "venv/",
    ".DS_Store", "*.swp", "*.swo",
    ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
    "*.o", "*.so", "*.dylib", "*.dll",
]


class IgnoreMatcher:
    def __init__(self, patterns: list[str] | None = None, include_defaults: bool = True):
        all_patterns: list[str] = []
        if include_defaults:
            all_patterns.extend(DEFAULT_PATTERNS)
        if patterns:
            all_patterns.extend(patterns)
        self._spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)

    @classmethod
    def from_defaults(cls) -> "IgnoreMatcher":
        return cls(patterns=None, include_defaults=True)

    @classmethod
    def from_file(cls, path: Path) -> "IgnoreMatcher":
        if not path.exists():
            return cls.from_defaults()
        lines = [
            ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        return cls(patterns=lines, include_defaults=True)

    def is_ignored(self, path: Path) -> bool:
        s = str(path).replace("\\", "/")
        return self._spec.match_file(s)
