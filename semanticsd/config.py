"""TOML config loader with defaults."""
from __future__ import annotations
import sys
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, field_validator
from semanticsd import paths

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class WatchConfig(BaseModel):
    directories: list[str] = []
    ignore_patterns: list[str] = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
    max_file_size_mb: int = 50


class TextEmbeddingConfig(BaseModel):
    preset: str = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str = ""
    dimensions: int = 0
    batch_size: int = 128


class VisionEmbeddingConfig(BaseModel):
    preset: str = ""
    model: str = ""
    base_url: str = ""
    dimensions: int = 0
    batch_size: int = 16


class EmbeddingConfig(BaseModel):
    text: TextEmbeddingConfig = Field(default_factory=TextEmbeddingConfig)
    vision: VisionEmbeddingConfig | None = None
    # Legacy passthrough fields (not used; tolerated for older configs)
    backend: str | None = None
    preset: str | None = None
    model: str | None = None
    base_url: str | None = None
    dimensions: int | None = None
    batch_size: int | None = None


class SearchConfig(BaseModel):
    default_mode: str = "semantic"
    max_results: int = 20

    @field_validator("default_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("semantic", "filename", "grep"):
            raise ValueError(f"default_mode must be semantic|filename|grep, got {v}")
        return v


class ChunkingConfig(BaseModel):
    strategy: str = "sliding"
    window_tokens: int = 512
    overlap_tokens: int = 64


class DaemonConfig(BaseModel):
    http_host: str = "127.0.0.1"
    http_port: int = 47600
    log_level: str = "info"


class PowerConfig(BaseModel):
    mode: str = "active"
    saver_reindex_interval: str = "1h"
    saver_pause_watcher: bool = True
    auto_saver_on_battery: bool = True


class IndexingConfig(BaseModel):
    max_attempts: int = 5
    worker_concurrency: int = 2


class Config(BaseModel):
    watch: WatchConfig = Field(default_factory=WatchConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)


def load(path: Path | None = None) -> Config:
    """Load config, falling back to defaults if file missing or section absent.

    Migrates legacy flat [embedding] sections to [embedding.text] in memory.
    """
    p = path or paths.config_path()
    if not p.exists():
        return Config()
    raw = tomllib.loads(p.read_text())
    emb = raw.get("embedding", {})
    if emb and "text" not in emb and "vision" not in emb and (
        "preset" in emb or "model" in emb or "backend" in emb
    ):
        text_keys = ("preset", "model", "base_url", "dimensions", "batch_size")
        raw["embedding"] = {
            "text": {k: v for k, v in emb.items() if k in text_keys},
        }
    try:
        return Config(**raw)
    except ValidationError as e:
        raise ValueError(str(e)) from e


DEFAULT_TOML = """\
[watch]
directories = []
ignore_patterns = [".git", "node_modules", ".DS_Store", "target", "build", "*.o", ".semanticsd"]
max_file_size_mb = 50

[embedding.text]
preset = "ollama"
model = "embeddinggemma"
base_url = "http://localhost:11434/v1"
batch_size = 128

# [embedding.vision]
# preset = "gemini"
# model = "gemini-embedding-2"
# batch_size = 8

[search]
default_mode = "semantic"
max_results = 20

[chunking]
strategy = "sliding"
window_tokens = 512
overlap_tokens = 64

[daemon]
http_host = "127.0.0.1"
http_port = 47600
log_level = "info"

[power]
mode = "active"
saver_reindex_interval = "1h"
saver_pause_watcher = true
auto_saver_on_battery = true

[indexing]
max_attempts = 5
worker_concurrency = 2
"""


def write_default() -> Path:
    """Write a default config file if none exists. Returns the path."""
    paths.ensure_dirs()
    p = paths.config_path()
    if not p.exists():
        p.write_text(DEFAULT_TOML)
    return p
