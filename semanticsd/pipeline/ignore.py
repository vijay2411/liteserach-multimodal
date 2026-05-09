"""gitignore-style path filter, backed by pathspec."""
from __future__ import annotations
from pathlib import Path
import pathspec


DEFAULT_PATTERNS = [
    # Version control / OS metadata
    ".git/", ".svn/", ".hg/",
    ".DS_Store", "Thumbs.db",
    # Editor leftovers
    "*.swp", "*.swo", "*~",
    # Generic build / dependency / cache directories
    "node_modules/", "__pycache__/", "*.pyc",
    "build/", "dist/", "target/",
    ".venv/", "venv/", "env/", ".env/",
    ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".tox/",
    ".turbo/", ".next/", ".nuxt/", ".cache/", ".parcel-cache/",
    "coverage/", ".coverage", ".semanticsd/",
    # Compiled / binary
    "*.o", "*.obj", "*.so", "*.dylib", "*.dll", "*.exe",
    "*.class",
    # Minified / bundled JS+CSS — these are giant tokens-per-byte and pollute
    # FTS with random identifier matches. Source maps are pure noise too.
    "*.min.js", "*.min.css",
    "*.bundle.js", "*.bundle.css",
    "*.map",
    "**/_astro/**",
    "**/.astro/**",
    # Vendor + plugin code that ships as bundled JS — index-bypass reduces noise
    "**/.obsidian/plugins/**",
    "**/.playwright-cli/**",
    "**/.husky/**",
    # Lock files (machine-generated, low search value)
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "uv.lock",
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
