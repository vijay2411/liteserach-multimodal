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
    directories: list[str] = ["~/"]
    ignore_patterns: list[str] = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
    max_file_size_mb: int = 50


class EmbeddingConfig(BaseModel):
    backend: str = "local"
    preset: str = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    base_url: str = ""
    dimensions: int = 0
    batch_size: int = 128


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
    """Load config, falling back to defaults if file missing or section absent."""
    p = path or paths.config_path()
    if not p.exists():
        return Config()
    raw = tomllib.loads(p.read_text())
    try:
        return Config(**raw)
    except ValidationError as e:
        raise ValueError(str(e)) from e


DEFAULT_TOML = """\
[watch]
directories = ["~/"]
ignore_patterns = [".git", "node_modules", ".DS_Store", "target", "build", "*.o"]
max_file_size_mb = 50

[embedding]
backend = "local"
preset = "local"
model = "BAAI/bge-small-en-v1.5"
batch_size = 128

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
