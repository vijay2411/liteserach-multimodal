# SemanticsD — Plan 3: Document Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full document-processing pipeline that turns files on disk into chunks + embeddings in SQLite. Per-file-type extractors, sliding-window chunker, content-hash dedup, async job-queue worker, `POST /v1/index` endpoint, `ssearch --index` CLI. After this plan: `ssearch --index ./sandbox` walks the folder, extracts text from every supported file type, embeds, and persists vectors via `sqlite-vec`.

**Architecture:** Extractor layer mirrors the embedder layer — base ABC + one file per file class in `semanticsd/extractors/`, with a registry mapping file extensions to extractor classes. Pipeline modules in `semanticsd/pipeline/` handle chunking, hashing, ignore patterns, walking, the indexer orchestrator, and the async worker. The job queue (already in the SQLite schema from Plan 1) is the source of truth for "what still needs embedding"; the worker is the only thing that calls the embedder.

**Tech Stack:** Python 3.11+, FastAPI, sqlite-vec, sentence-transformers (default embedder), beautifulsoup4 (HTML), pypdf (PDF), python-docx (DOCX), openpyxl (XLSX), python-pptx (PPTX), ebooklib (EPUB), striprtf (RTF), pytesseract (image OCR — graceful degrade), faster-whisper (audio transcription — graceful degrade), pathspec (.semanticsdignore parser), pytest.

**Spec reference:** `/Users/vedantvijay/dev/side_projects/SemanticSearch/docs/superpowers/specs/2026-05-09-semanticsd-design.md`
**Plan 1:** `docs/superpowers/plans/2026-05-09-foundation.md` (shipped — daemon + schema + auth + CLI)
**Plan 2:** `docs/superpowers/plans/2026-05-09-embedders.md` (shipped — embedder layer + sqlite-vec + sandbox)

---

## File Structure for This Plan

```
SemanticSearch/
├── semanticsd/
│   ├── extractors/                    # NEW: pluggable file-type extractors
│   │   ├── __init__.py
│   │   ├── base.py                    # Extractor ABC + ExtractedDoc model
│   │   ├── registry.py                # EXTENSION_TO_CLASS + get_extractor()
│   │   ├── text.py                    # text/markdown/code/json/yaml/toml/csv
│   │   ├── html.py                    # beautifulsoup4
│   │   ├── pdf.py                     # pypdf
│   │   ├── docx.py                    # python-docx
│   │   ├── xlsx.py                    # openpyxl
│   │   ├── pptx.py                    # python-pptx
│   │   ├── epub.py                    # ebooklib
│   │   ├── rtf.py                     # striprtf
│   │   ├── email_msg.py               # builtin email/mailbox
│   │   ├── notebook.py                # builtin json (.ipynb)
│   │   ├── image.py                   # pytesseract OCR, graceful degrade
│   │   └── audio.py                   # faster-whisper, graceful degrade
│   ├── pipeline/                      # NEW: orchestration modules
│   │   ├── __init__.py
│   │   ├── chunker.py                 # SlidingWindowChunker
│   │   ├── hasher.py                  # sha256 + dedup lookup
│   │   ├── ignore.py                  # .semanticsdignore via pathspec
│   │   ├── walker.py                  # walk dir + ignore + size-limit
│   │   ├── indexer.py                 # orchestrator: extract → chunk → hash → upsert → queue
│   │   └── worker.py                  # async job-queue worker
│   ├── server/routes/
│   │   └── index.py                   # NEW: POST /v1/index
│   ├── server/app.py                  # MODIFY: include index router + lifespan worker startup
│   └── cli.py                         # MODIFY: add --index <path>
├── tests/
│   ├── _fixtures.py                   # NEW: programmatic fixture builders (pdf, docx, etc.)
│   ├── test_extractors_text.py        # NEW
│   ├── test_extractors_html.py        # NEW
│   ├── test_extractors_pdf.py         # NEW
│   ├── test_extractors_docx.py        # NEW
│   ├── test_extractors_xlsx.py        # NEW
│   ├── test_extractors_pptx.py        # NEW
│   ├── test_extractors_epub.py        # NEW
│   ├── test_extractors_rtf.py         # NEW
│   ├── test_extractors_email.py       # NEW
│   ├── test_extractors_notebook.py    # NEW
│   ├── test_extractors_image.py       # NEW (graceful-degrade tests)
│   ├── test_extractors_audio.py       # NEW (slow, gated)
│   ├── test_extractors_registry.py    # NEW
│   ├── test_pipeline_chunker.py       # NEW
│   ├── test_pipeline_hasher.py        # NEW
│   ├── test_pipeline_ignore.py        # NEW
│   ├── test_pipeline_walker.py        # NEW
│   ├── test_pipeline_indexer.py       # NEW
│   ├── test_pipeline_worker.py        # NEW
│   ├── test_index_route.py            # NEW
│   └── test_e2e_index.py              # NEW (end-to-end against ./sandbox)
└── requirements.txt                   # MODIFY: add Plan 3 deps
```

---

## Pre-Task: Bootstrap dependencies (controller-run, before Task 1)

```bash
.venv/bin/pip install \
    pypdf \
    python-docx \
    openpyxl \
    python-pptx \
    beautifulsoup4 \
    ebooklib \
    striprtf \
    pytesseract \
    faster-whisper \
    pathspec \
    pillow
```

Optional system deps (for OCR + audio): `brew install tesseract ffmpeg`. The pipeline gracefully degrades if either is missing — image/audio files are simply skipped with a warning.

---

## Task 1: Update requirements.txt + add Plan 3 deps

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Append Plan 3 deps**

```
# Plan 3 additions
pypdf>=4.0
reportlab>=4.0          # only used by tests/_fixtures.py to build test PDFs
python-docx>=1.1
openpyxl>=3.1
python-pptx>=0.6
beautifulsoup4>=4.12
ebooklib>=0.18
striprtf>=0.0.27
pytesseract>=0.3.10
faster-whisper>=1.0
pathspec>=0.12
pillow>=10.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): Plan 3 — extractor + pipeline libs"
```

---

## Task 2: Test fixtures helper (programmatic file generators)

**Files:**
- Create: `tests/_fixtures.py`

This module produces minimal binary files on demand so tests don't have to commit binaries. Used by every extractor test.

- [ ] **Step 1: Create `tests/_fixtures.py`**

```python
"""Programmatic fixture builders for extractor tests.

Each helper returns a Path to a tmp file containing a minimal valid file of
that type. The caller (a test using the `tmp_path` pytest fixture) supplies
the parent dir, so files are auto-cleaned.
"""
from __future__ import annotations
from pathlib import Path


def make_text(tmp_path: Path, name: str = "doc.txt", body: str = "Hello world.\nLine two.") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def make_markdown(tmp_path: Path, name: str = "doc.md") -> Path:
    return make_text(tmp_path, name=name, body="# Title\n\nA paragraph.\n\n- bullet\n")


def make_code(tmp_path: Path, name: str = "hello.py") -> Path:
    return make_text(tmp_path, name=name, body='def greet():\n    return "hi"\n')


def make_json(tmp_path: Path, name: str = "data.json") -> Path:
    return make_text(tmp_path, name=name, body='{"name": "alice", "tags": ["a", "b"]}')


def make_yaml(tmp_path: Path, name: str = "data.yaml") -> Path:
    return make_text(tmp_path, name=name, body="name: alice\ntags:\n  - a\n  - b\n")


def make_csv(tmp_path: Path, name: str = "data.csv") -> Path:
    return make_text(tmp_path, name=name, body="name,tag\nalice,a\nbob,b\n")


def make_html(tmp_path: Path, name: str = "page.html") -> Path:
    body = (
        "<!DOCTYPE html><html><head><title>Page</title></head>"
        "<body><h1>Heading</h1><p>Paragraph one.</p>"
        "<script>console.log('skip')</script></body></html>"
    )
    return make_text(tmp_path, name=name, body=body)


def make_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    """Two-page PDF with simple text, generated via reportlab."""
    from reportlab.pdfgen import canvas
    p = tmp_path / name
    c = canvas.Canvas(str(p))
    c.drawString(100, 700, "Hello from page one.")
    c.showPage()
    c.drawString(100, 700, "Page two has different text.")
    c.showPage()
    c.save()
    return p


def make_docx(tmp_path: Path, name: str = "doc.docx") -> Path:
    from docx import Document
    p = tmp_path / name
    doc = Document()
    doc.add_heading("Title", level=1)
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph with different content.")
    doc.save(str(p))
    return p


def make_xlsx(tmp_path: Path, name: str = "book.xlsx") -> Path:
    from openpyxl import Workbook
    p = tmp_path / name
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(["name", "score"])
    ws.append(["alice", 90])
    ws.append(["bob", 85])
    wb.save(str(p))
    return p


def make_pptx(tmp_path: Path, name: str = "deck.pptx") -> Path:
    from pptx import Presentation
    p = tmp_path / name
    prs = Presentation()
    slide_layout = prs.slide_layouts[5]  # Title only
    s = prs.slides.add_slide(slide_layout)
    s.shapes.title.text = "Slide One"
    s2 = prs.slides.add_slide(slide_layout)
    s2.shapes.title.text = "Slide Two has different content"
    prs.save(str(p))
    return p


def make_epub(tmp_path: Path, name: str = "book.epub") -> Path:
    from ebooklib import epub
    p = tmp_path / name
    book = epub.EpubBook()
    book.set_identifier("id-1")
    book.set_title("Tiny Book")
    book.set_language("en")
    c1 = epub.EpubHtml(title="Chapter 1", file_name="c1.xhtml", lang="en")
    c1.content = "<h1>Chapter 1</h1><p>Lorem ipsum content.</p>"
    book.add_item(c1)
    book.toc = (epub.Link("c1.xhtml", "Chapter 1", "c1"),)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1]
    epub.write_epub(str(p), book)
    return p


def make_rtf(tmp_path: Path, name: str = "doc.rtf") -> Path:
    body = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Helvetica;}}\f0\fs24 Hello from RTF.\par Second line.\par}"
    return make_text(tmp_path, name=name, body=body)


def make_eml(tmp_path: Path, name: str = "msg.eml") -> Path:
    body = (
        "From: alice@example.com\r\n"
        "To: bob@example.com\r\n"
        "Subject: A test message\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "Hello Bob, this is the body of the message.\r\n"
    )
    return make_text(tmp_path, name=name, body=body)


def make_ipynb(tmp_path: Path, name: str = "notebook.ipynb") -> Path:
    import json
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Notebook Title\n", "Some markdown text."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": ["x = 42\n", "print(x)"]},
        ],
    }
    p = tmp_path / name
    p.write_text(json.dumps(nb))
    return p


def make_image_with_text(tmp_path: Path, name: str = "img.png", text: str = "hello world") -> Path:
    """A PNG with rendered text — used for OCR tests.

    Uses a TrueType font when available (better Tesseract accuracy) and falls
    back to Pillow's default bitmap font otherwise. The image is intentionally
    large to give Tesseract enough resolution.
    """
    from PIL import Image, ImageDraw, ImageFont
    p = tmp_path / name
    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    font = None
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 60)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text((20, 60), text, fill="black", font=font)
    img.save(str(p))
    return p


def make_wav_silence(tmp_path: Path, name: str = "audio.wav", duration_s: float = 0.5) -> Path:
    """A tiny silent WAV. faster-whisper transcribes it to '' or near-empty."""
    import wave
    p = tmp_path / name
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        n_frames = int(16000 * duration_s)
        w.writeframes(b"\x00\x00" * n_frames)
    return p
```

- [ ] **Step 2: Commit**

```bash
git add tests/_fixtures.py
git commit -m "test(fixtures): programmatic builders for binary file types"
```

---

## Task 3: Extractor base + `ExtractedDoc` model + registry skeleton (TDD)

**Files:**
- Create: `semanticsd/extractors/__init__.py` (stub)
- Create: `semanticsd/extractors/base.py`
- Create: `semanticsd/extractors/registry.py` (stub: empty `EXTENSION_TO_CLASS = {}` + lookup function that returns None)
- Create: `tests/test_extractors_registry.py` (just the base + registry-shape tests)

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_registry.py`:
```python
"""Extractor base + registry shape."""
import pytest
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors import registry


def test_extracted_doc_required_fields():
    seg = ExtractedSegment(text="hello", byte_start=0, byte_end=5)
    doc = ExtractedDoc(path="/tmp/foo.txt", file_type="text", segments=[seg])
    assert doc.path == "/tmp/foo.txt"
    assert doc.file_type == "text"
    assert doc.segments[0].text == "hello"


def test_extractor_is_abstract():
    try:
        Extractor()  # type: ignore[abstract]
    except TypeError:
        return
    raise AssertionError("Extractor should be abstract")


def test_concrete_extractor():
    class Stub(Extractor):
        file_type = "stub"
        extensions = (".stub",)
        def extract(self, path: Path) -> ExtractedDoc:
            return ExtractedDoc(path=str(path), file_type="stub",
                                segments=[ExtractedSegment(text="x", byte_start=0, byte_end=1)])
    s = Stub()
    out = s.extract(Path("/tmp/x.stub"))
    assert out.file_type == "stub"


def test_registry_returns_none_for_unknown(tmp_path):
    p = tmp_path / "unknown.xyz"
    p.write_text("")
    assert registry.get_extractor(p) is None
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_registry.py -v
```

- [ ] **Step 3: Create the package init (stub)**

`semanticsd/extractors/__init__.py`:
```python
"""Extractor layer — pluggable file-type extractors."""
```

- [ ] **Step 4: Implement `semanticsd/extractors/base.py`**

```python
"""Abstract Extractor base + extracted-document model."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel


class ExtractedSegment(BaseModel):
    """One contiguous chunk of extracted text (e.g. a paragraph, page, or slide).

    `byte_start` and `byte_end` are byte offsets within the *extracted text*
    (not within the original binary). They let us point search results back to
    a region of the doc for highlighting.
    """
    text: str
    byte_start: int
    byte_end: int
    metadata: dict = {}  # e.g. {"page": 2}, {"slide": 3}


class ExtractedDoc(BaseModel):
    path: str
    file_type: str
    segments: list[ExtractedSegment]
    metadata: dict = {}  # e.g. {"title": "...", "author": "..."}


class Extractor(ABC):
    """One file class per concrete subclass.

    Subclasses set:
      file_type: str         — short identifier for the `files.file_type` column
      extensions: tuple[str] — file extensions handled, lowercase, with dot, e.g. (".pdf",)
    """

    file_type: ClassVar[str] = ""
    extensions: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def extract(self, path: Path) -> ExtractedDoc: ...
```

- [ ] **Step 5: Implement `semanticsd/extractors/registry.py`**

```python
"""Extension -> Extractor mapping. Filled in as each concrete extractor lands."""
from __future__ import annotations
from pathlib import Path
from typing import Type
from semanticsd.extractors.base import Extractor


EXTENSION_TO_CLASS: dict[str, Type[Extractor]] = {}


def register(extractor_cls: Type[Extractor]) -> Type[Extractor]:
    """Register a class for each of its extensions. Used as a class decorator."""
    for ext in extractor_cls.extensions:
        EXTENSION_TO_CLASS[ext.lower()] = extractor_cls
    return extractor_cls


def get_extractor(path: Path) -> Extractor | None:
    """Return an Extractor instance for the file at `path`, or None if unsupported."""
    ext = path.suffix.lower()
    cls = EXTENSION_TO_CLASS.get(ext)
    return cls() if cls else None
```

- [ ] **Step 6: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_registry.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Run full suite — no regressions**

```bash
.venv/bin/pytest -q -m "not slow"
```
Expected: 84 passed.

- [ ] **Step 8: Commit**

```bash
git add semanticsd/extractors/ tests/test_extractors_registry.py
git commit -m "feat(extractors): base ABC + ExtractedDoc + registry skeleton"
```

---

## Task 4: Text/markdown/code/structured-data extractor (TDD)

Single file handles `.txt .md .rst .org .py .js .ts .rs .go .java .c .cpp .h .swift .rb .php .sh .json .yaml .yml .toml .csv .tsv .xml .log` — anything that's plain text on disk. Just reads with `read_text()` and returns the whole file as a single segment.

**Files:**
- Create: `semanticsd/extractors/text.py`
- Create: `tests/test_extractors_text.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_text.py`:
```python
from pathlib import Path
from semanticsd.extractors.text import TextExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_text, make_markdown, make_code, make_json, make_yaml, make_csv


def test_extracts_plain_text(tmp_path):
    p = make_text(tmp_path, body="Hello\nWorld")
    out = TextExtractor().extract(p)
    assert out.file_type == "text"
    assert len(out.segments) == 1
    assert out.segments[0].text == "Hello\nWorld"
    assert out.segments[0].byte_start == 0
    assert out.segments[0].byte_end == len("Hello\nWorld".encode())


def test_extracts_markdown(tmp_path):
    p = make_markdown(tmp_path)
    out = TextExtractor().extract(p)
    assert "Title" in out.segments[0].text


def test_extracts_code(tmp_path):
    p = make_code(tmp_path)
    out = TextExtractor().extract(p)
    assert "def greet" in out.segments[0].text


def test_extracts_json(tmp_path):
    p = make_json(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_extracts_yaml(tmp_path):
    p = make_yaml(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_extracts_csv(tmp_path):
    p = make_csv(tmp_path)
    out = TextExtractor().extract(p)
    assert "alice" in out.segments[0].text


def test_registry_picks_text_for_md(tmp_path):
    p = make_markdown(tmp_path)
    e = registry.get_extractor(p)
    assert isinstance(e, TextExtractor)


def test_registry_picks_text_for_py(tmp_path):
    p = make_code(tmp_path)
    e = registry.get_extractor(p)
    assert isinstance(e, TextExtractor)


def test_handles_invalid_utf8_gracefully(tmp_path):
    p = tmp_path / "bin.txt"
    p.write_bytes(b"\xff\xfe\x00\x00bad")
    out = TextExtractor().extract(p)
    assert isinstance(out.segments[0].text, str)  # decoded with replace
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_text.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/text.py`**

```python
"""Plain-text-on-disk extractor: text, markdown, code, structured config files."""
from __future__ import annotations
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


_TEXT_EXTS = (
    # Text and prose
    ".txt", ".md", ".rst", ".org", ".log",
    # Code (most common Mac dev languages — extend as needed)
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java", ".kt",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".swift", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".fish", ".lua", ".pl", ".scala", ".clj",
    # Structured data / config
    ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".xml", ".html",
    ".sql", ".env", ".conf", ".ini", ".gitignore", ".dockerfile",
    # NOTE: ".html" is also covered by HtmlExtractor (richer extraction);
    # registry order in registry.py will let html.py override this for .html.
)


@register
class TextExtractor(Extractor):
    file_type = "text"
    extensions = _TEXT_EXTS

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[
                ExtractedSegment(
                    text=text,
                    byte_start=0,
                    byte_end=len(text.encode("utf-8")),
                )
            ],
        )
```

NOTE: The `@register` decorator runs at import time, registering this extractor for all listed extensions. We'll handle the `.html` precedence concern in Task 5 (the `HtmlExtractor` will register itself for `.html .htm` too; whichever module is imported later wins). To make the order deterministic, the package `__init__.py` will be updated in Task 16 to import all extractor modules in a specific order.

- [ ] **Step 4: Update `semanticsd/extractors/__init__.py` to import this module so the registration runs**

```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401  (registers TextExtractor)
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_text.py -v
```
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/text.py semanticsd/extractors/__init__.py tests/test_extractors_text.py
git commit -m "feat(extractors): text/code/markdown/structured-data (TextExtractor)"
```

---

## Task 5: HTML extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/html.py`
- Modify: `semanticsd/extractors/__init__.py` (import order: html after text so it overrides for .html)
- Create: `tests/test_extractors_html.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_html.py`:
```python
from semanticsd.extractors.html import HtmlExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_html


def test_strips_tags_keeps_text(tmp_path):
    p = make_html(tmp_path)
    out = HtmlExtractor().extract(p)
    text = out.segments[0].text
    assert "Heading" in text
    assert "Paragraph one" in text
    # script content must be stripped
    assert "console.log" not in text
    assert "<h1>" not in text


def test_extractor_metadata_has_title(tmp_path):
    p = make_html(tmp_path)
    out = HtmlExtractor().extract(p)
    assert out.metadata.get("title") == "Page"


def test_registry_picks_html(tmp_path):
    p = make_html(tmp_path)
    assert isinstance(registry.get_extractor(p), HtmlExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_html.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/html.py`**

```python
"""HTML extractor — strips tags, keeps visible text + title."""
from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class HtmlExtractor(Extractor):
    file_type = "html"
    extensions = (".html", ".htm", ".xhtml")

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_bytes()
        soup = BeautifulSoup(raw, "html.parser")
        # Remove non-content tags.
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = (soup.title.string.strip() if soup.title and soup.title.string else None)
        text = soup.get_text(separator="\n", strip=True)
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(text=text, byte_start=0, byte_end=len(text.encode("utf-8")))],
            metadata={"title": title} if title else {},
        )
```

- [ ] **Step 4: Update `semanticsd/extractors/__init__.py` to import html AFTER text (so .html mapping is overridden)**

```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_html.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/html.py semanticsd/extractors/__init__.py tests/test_extractors_html.py
git commit -m "feat(extractors): HTML (beautifulsoup4)"
```

---

## Task 6: PDF extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/pdf.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_pdf.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_pdf.py`:
```python
from semanticsd.extractors.pdf import PdfExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_pdf


def test_extracts_pages_as_segments(tmp_path):
    p = make_pdf(tmp_path)
    out = PdfExtractor().extract(p)
    assert out.file_type == "pdf"
    # Two pages -> two segments.
    assert len(out.segments) == 2
    # Each segment carries its page number in metadata.
    assert out.segments[0].metadata == {"page": 1}
    assert out.segments[1].metadata == {"page": 2}
    assert "Hello" in out.segments[0].text
    assert "Page two" in out.segments[1].text


def test_registry_picks_pdf(tmp_path):
    p = make_pdf(tmp_path)
    assert isinstance(registry.get_extractor(p), PdfExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_pdf.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/pdf.py`**

```python
"""PDF extractor — one segment per page, via pypdf."""
from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class PdfExtractor(Extractor):
    file_type = "pdf"
    extensions = (".pdf",)

    def extract(self, path: Path) -> ExtractedDoc:
        reader = PdfReader(str(path))
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"page": i},
            ))
            cursor += seg_bytes + 1  # +1 for an implicit newline between pages
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
```

- [ ] **Step 4: Update `__init__.py` to import pdf**

`semanticsd/extractors/__init__.py`:
```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
from semanticsd.extractors import pdf  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_pdf.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/pdf.py semanticsd/extractors/__init__.py tests/test_extractors_pdf.py
git commit -m "feat(extractors): PDF (pypdf, page-aware)"
```

---

## Task 7: DOCX extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/docx.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_docx.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_docx.py`:
```python
from semanticsd.extractors.docx import DocxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_docx


def test_extracts_paragraphs(tmp_path):
    p = make_docx(tmp_path)
    out = DocxExtractor().extract(p)
    assert out.file_type == "docx"
    full = "\n".join(s.text for s in out.segments)
    assert "Title" in full
    assert "First paragraph" in full
    assert "Second paragraph" in full


def test_registry_picks_docx(tmp_path):
    p = make_docx(tmp_path)
    assert isinstance(registry.get_extractor(p), DocxExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_docx.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/docx.py`**

```python
"""DOCX extractor — one segment per paragraph, via python-docx."""
from __future__ import annotations
from pathlib import Path
from docx import Document
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class DocxExtractor(Extractor):
    file_type = "docx"
    extensions = (".docx",)

    def extract(self, path: Path) -> ExtractedDoc:
        doc = Document(str(path))
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"paragraph": i},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
```

- [ ] **Step 4: Update `__init__.py`**

```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
from semanticsd.extractors import pdf  # noqa: F401
from semanticsd.extractors import docx  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_docx.py -v
```

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/docx.py semanticsd/extractors/__init__.py tests/test_extractors_docx.py
git commit -m "feat(extractors): DOCX (python-docx, paragraph-aware)"
```

---

## Task 8: XLSX extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/xlsx.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_xlsx.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_xlsx.py`:
```python
from semanticsd.extractors.xlsx import XlsxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_xlsx


def test_extracts_rows(tmp_path):
    p = make_xlsx(tmp_path)
    out = XlsxExtractor().extract(p)
    assert out.file_type == "xlsx"
    full = "\n".join(s.text for s in out.segments)
    assert "alice" in full
    assert "90" in full


def test_segment_per_sheet(tmp_path):
    p = make_xlsx(tmp_path)
    out = XlsxExtractor().extract(p)
    # One segment per sheet — the fixture has one sheet.
    assert len(out.segments) == 1
    assert out.segments[0].metadata.get("sheet") == "data"


def test_registry_picks_xlsx(tmp_path):
    p = make_xlsx(tmp_path)
    assert isinstance(registry.get_extractor(p), XlsxExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_xlsx.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/xlsx.py`**

```python
"""XLSX extractor — one segment per sheet, rows as TSV-ish text, via openpyxl."""
from __future__ import annotations
from pathlib import Path
from openpyxl import load_workbook
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class XlsxExtractor(Extractor):
    file_type = "xlsx"
    extensions = (".xlsx", ".xlsm")

    def extract(self, path: Path) -> ExtractedDoc:
        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        segments: list[ExtractedSegment] = []
        cursor = 0
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows: list[str] = []
            for row in ws.iter_rows(values_only=True):
                cells = ["" if c is None else str(c) for c in row]
                rows.append("\t".join(cells))
            text = "\n".join(rows).strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"sheet": sheet_name},
            ))
            cursor += seg_bytes + 1
        wb.close()
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
```

- [ ] **Step 4: Update `__init__.py`**

```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
from semanticsd.extractors import pdf  # noqa: F401
from semanticsd.extractors import docx  # noqa: F401
from semanticsd.extractors import xlsx  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_xlsx.py -v
```

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/xlsx.py semanticsd/extractors/__init__.py tests/test_extractors_xlsx.py
git commit -m "feat(extractors): XLSX (openpyxl, sheet-aware)"
```

---

## Task 9: PPTX extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/pptx.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_pptx.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_pptx.py`:
```python
from semanticsd.extractors.pptx import PptxExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_pptx


def test_extracts_slides_as_segments(tmp_path):
    p = make_pptx(tmp_path)
    out = PptxExtractor().extract(p)
    assert out.file_type == "pptx"
    assert len(out.segments) == 2
    assert out.segments[0].metadata == {"slide": 1}
    assert out.segments[1].metadata == {"slide": 2}
    assert "Slide One" in out.segments[0].text
    assert "Slide Two" in out.segments[1].text


def test_registry_picks_pptx(tmp_path):
    p = make_pptx(tmp_path)
    assert isinstance(registry.get_extractor(p), PptxExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_pptx.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/pptx.py`**

```python
"""PPTX extractor — one segment per slide, via python-pptx."""
from __future__ import annotations
from pathlib import Path
from pptx import Presentation
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class PptxExtractor(Extractor):
    file_type = "pptx"
    extensions = (".pptx",)

    def extract(self, path: Path) -> ExtractedDoc:
        prs = Presentation(str(path))
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, slide in enumerate(prs.slides, start=1):
            parts: list[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text.strip())
            text = "\n".join(p for p in parts if p)
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"slide": i},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
```

- [ ] **Step 4: Update `__init__.py`**

```python
"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
from semanticsd.extractors import pdf  # noqa: F401
from semanticsd.extractors import docx  # noqa: F401
from semanticsd.extractors import xlsx  # noqa: F401
from semanticsd.extractors import pptx  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_pptx.py -v
```

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/pptx.py semanticsd/extractors/__init__.py tests/test_extractors_pptx.py
git commit -m "feat(extractors): PPTX (python-pptx, slide-aware)"
```

---

## Task 10: EPUB extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/epub.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_epub.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_epub.py`:
```python
from semanticsd.extractors.epub import EpubExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_epub


def test_extracts_chapters_as_segments(tmp_path):
    p = make_epub(tmp_path)
    out = EpubExtractor().extract(p)
    assert out.file_type == "epub"
    full = "\n".join(s.text for s in out.segments)
    assert "Chapter 1" in full
    assert "Lorem ipsum" in full


def test_metadata_has_title(tmp_path):
    p = make_epub(tmp_path)
    out = EpubExtractor().extract(p)
    assert out.metadata.get("title") == "Tiny Book"


def test_registry_picks_epub(tmp_path):
    p = make_epub(tmp_path)
    assert isinstance(registry.get_extractor(p), EpubExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_epub.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/epub.py`**

```python
"""EPUB extractor — one segment per chapter (HTML stripped), via ebooklib."""
from __future__ import annotations
from pathlib import Path
from bs4 import BeautifulSoup
from ebooklib import epub, ITEM_DOCUMENT
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class EpubExtractor(Extractor):
    file_type = "epub"
    extensions = (".epub",)

    def extract(self, path: Path) -> ExtractedDoc:
        book = epub.read_epub(str(path))
        title = book.get_metadata("DC", "title")
        title_str = title[0][0] if title else None

        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, item in enumerate(book.get_items_of_type(ITEM_DOCUMENT)):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"chapter": i + 1, "name": item.get_name()},
            ))
            cursor += seg_bytes + 1

        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
            metadata={"title": title_str} if title_str else {},
        )
```

- [ ] **Step 4: Update `__init__.py`**

```python
from semanticsd.extractors import text  # noqa: F401
from semanticsd.extractors import html  # noqa: F401
from semanticsd.extractors import pdf  # noqa: F401
from semanticsd.extractors import docx  # noqa: F401
from semanticsd.extractors import xlsx  # noqa: F401
from semanticsd.extractors import pptx  # noqa: F401
from semanticsd.extractors import epub  # noqa: F401
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_extractors_epub.py -v
```

- [ ] **Step 6: Commit**

```bash
git add semanticsd/extractors/epub.py semanticsd/extractors/__init__.py tests/test_extractors_epub.py
git commit -m "feat(extractors): EPUB (ebooklib, chapter-aware)"
```

---

## Task 11: RTF extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/rtf.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_rtf.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_rtf.py`:
```python
from semanticsd.extractors.rtf import RtfExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_rtf


def test_strips_rtf_control_codes(tmp_path):
    p = make_rtf(tmp_path)
    out = RtfExtractor().extract(p)
    assert out.file_type == "rtf"
    text = out.segments[0].text
    assert "Hello from RTF" in text
    assert "\\rtf1" not in text  # control words stripped


def test_registry_picks_rtf(tmp_path):
    p = make_rtf(tmp_path)
    assert isinstance(registry.get_extractor(p), RtfExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_rtf.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/rtf.py`**

```python
"""RTF extractor — strips control codes, via striprtf."""
from __future__ import annotations
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class RtfExtractor(Extractor):
    file_type = "rtf"
    extensions = (".rtf",)

    def extract(self, path: Path) -> ExtractedDoc:
        raw = path.read_text(errors="replace")
        text = rtf_to_text(raw).strip()
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[
                ExtractedSegment(
                    text=text,
                    byte_start=0,
                    byte_end=len(text.encode("utf-8")),
                )
            ],
        )
```

- [ ] **Step 4: Update `__init__.py`** — append `from semanticsd.extractors import rtf  # noqa: F401`

- [ ] **Step 5: Run + commit**

```bash
.venv/bin/pytest tests/test_extractors_rtf.py -v
git add semanticsd/extractors/rtf.py semanticsd/extractors/__init__.py tests/test_extractors_rtf.py
git commit -m "feat(extractors): RTF (striprtf)"
```

---

## Task 12: Email (.eml) extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/email_msg.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_email.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_email.py`:
```python
from semanticsd.extractors.email_msg import EmailExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_eml


def test_extracts_body_and_subject(tmp_path):
    p = make_eml(tmp_path)
    out = EmailExtractor().extract(p)
    assert out.file_type == "email"
    text = out.segments[0].text
    assert "Hello Bob" in text
    assert out.metadata.get("subject") == "A test message"
    assert out.metadata.get("from") == "alice@example.com"


def test_registry_picks_eml(tmp_path):
    p = make_eml(tmp_path)
    assert isinstance(registry.get_extractor(p), EmailExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_email.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/email_msg.py`**

```python
"""Email (.eml) extractor — headers + plain-text body, via stdlib email."""
from __future__ import annotations
from pathlib import Path
from email import message_from_bytes
from email.policy import default as default_policy
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class EmailExtractor(Extractor):
    file_type = "email"
    extensions = (".eml",)

    def extract(self, path: Path) -> ExtractedDoc:
        msg = message_from_bytes(path.read_bytes(), policy=default_policy)
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body_text = ""
        if body_part is not None:
            body_text = body_part.get_content()
            # If we got HTML, strip tags.
            if body_part.get_content_type() == "text/html":
                from bs4 import BeautifulSoup
                body_text = BeautifulSoup(body_text, "html.parser").get_text(separator="\n", strip=True)
        text = body_text.strip()
        meta = {
            "subject": msg.get("Subject", ""),
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "date": msg.get("Date", ""),
        }
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(text=text, byte_start=0, byte_end=len(text.encode("utf-8")))],
            metadata={k: v for k, v in meta.items() if v},
        )
```

- [ ] **Step 4: Update `__init__.py`** — append `from semanticsd.extractors import email_msg  # noqa: F401`

- [ ] **Step 5: Run + commit**

```bash
.venv/bin/pytest tests/test_extractors_email.py -v
git add semanticsd/extractors/email_msg.py semanticsd/extractors/__init__.py tests/test_extractors_email.py
git commit -m "feat(extractors): email (.eml, stdlib email)"
```

---

## Task 13: Notebook (.ipynb) extractor (TDD)

**Files:**
- Create: `semanticsd/extractors/notebook.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_notebook.py`

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_notebook.py`:
```python
from semanticsd.extractors.notebook import NotebookExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_ipynb


def test_extracts_cells_as_segments(tmp_path):
    p = make_ipynb(tmp_path)
    out = NotebookExtractor().extract(p)
    assert out.file_type == "notebook"
    assert len(out.segments) == 2  # markdown + code cell
    assert out.segments[0].metadata.get("cell_type") == "markdown"
    assert out.segments[1].metadata.get("cell_type") == "code"
    assert "Notebook Title" in out.segments[0].text
    assert "x = 42" in out.segments[1].text


def test_registry_picks_ipynb(tmp_path):
    p = make_ipynb(tmp_path)
    assert isinstance(registry.get_extractor(p), NotebookExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_notebook.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/notebook.py`**

```python
"""Jupyter notebook (.ipynb) extractor — one segment per code/markdown cell."""
from __future__ import annotations
import json
from pathlib import Path
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register


@register
class NotebookExtractor(Extractor):
    file_type = "notebook"
    extensions = (".ipynb",)

    def extract(self, path: Path) -> ExtractedDoc:
        nb = json.loads(path.read_text())
        segments: list[ExtractedSegment] = []
        cursor = 0
        for i, cell in enumerate(nb.get("cells", [])):
            ctype = cell.get("cell_type", "")
            if ctype not in ("code", "markdown"):
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            text = source.strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"cell_index": i, "cell_type": ctype},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=segments,
        )
```

- [ ] **Step 4: Update `__init__.py`** — append `from semanticsd.extractors import notebook  # noqa: F401`

- [ ] **Step 5: Run + commit**

```bash
.venv/bin/pytest tests/test_extractors_notebook.py -v
git add semanticsd/extractors/notebook.py semanticsd/extractors/__init__.py tests/test_extractors_notebook.py
git commit -m "feat(extractors): Jupyter notebook (.ipynb)"
```

---

## Task 14: Image extractor — Tesseract OCR with graceful degrade (TDD)

**Files:**
- Create: `semanticsd/extractors/image.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_image.py`

The extractor catches `pytesseract.TesseractNotFoundError` and any other OCR runtime errors, returning a doc with **empty segments** and a note in `metadata["ocr_error"]`. Tests cover both the success case (PIL-rendered text) and the degraded case (Tesseract missing).

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_image.py`:
```python
import pytest
from semanticsd.extractors.image import ImageExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_image_with_text


def _has_tesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
        # pytesseract.get_tesseract_version() raises if binary missing.
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


HAS_TESS = _has_tesseract()


@pytest.mark.skipif(not HAS_TESS, reason="tesseract binary not installed")
def test_ocr_real(tmp_path):
    p = make_image_with_text(tmp_path, text="hello")
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    full = " ".join(s.text for s in out.segments)
    assert "hello" in full.lower()


def test_graceful_degrade_when_tesseract_missing(tmp_path, monkeypatch):
    p = make_image_with_text(tmp_path, text="anything")
    import pytesseract
    def boom(*a, **kw):
        raise pytesseract.TesseractNotFoundError()
    monkeypatch.setattr(pytesseract, "image_to_string", boom)
    out = ImageExtractor().extract(p)
    assert out.file_type == "image"
    assert out.segments == []
    assert "ocr_error" in out.metadata


def test_registry_picks_image(tmp_path):
    p = make_image_with_text(tmp_path)
    assert isinstance(registry.get_extractor(p), ImageExtractor)
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_image.py -v
```

- [ ] **Step 3: Implement `semanticsd/extractors/image.py`**

```python
"""Image extractor — Tesseract OCR via pytesseract.

Graceful degrade: if Tesseract binary is missing, returns a doc with no
segments but an `ocr_error` in metadata. The pipeline treats that as a no-op
(file is recorded but no chunks are queued for embedding).
"""
from __future__ import annotations
import logging
from pathlib import Path
import pytesseract
from PIL import Image
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)


@register
class ImageExtractor(Extractor):
    file_type = "image"
    extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic")

    def extract(self, path: Path) -> ExtractedDoc:
        try:
            img = Image.open(str(path))
            text = pytesseract.image_to_string(img).strip()
        except pytesseract.TesseractNotFoundError as e:
            log.warning("Tesseract not installed; skipping OCR for %s", path)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "tesseract_not_found", "detail": str(e)},
            )
        except Exception as e:
            log.warning("OCR failed for %s: %s", path, e)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "ocr_failure", "detail": str(e)},
            )

        if not text:
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"ocr_error": "empty_result"},
            )
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=[ExtractedSegment(
                text=text,
                byte_start=0,
                byte_end=len(text.encode("utf-8")),
            )],
        )
```

- [ ] **Step 4: Update `__init__.py`** — append `from semanticsd.extractors import image  # noqa: F401`

- [ ] **Step 5: Run + commit**

```bash
.venv/bin/pytest tests/test_extractors_image.py -v
git add semanticsd/extractors/image.py semanticsd/extractors/__init__.py tests/test_extractors_image.py
git commit -m "feat(extractors): image OCR via pytesseract, graceful degrade"
```

---

## Task 15: Audio extractor — faster-whisper with graceful degrade (TDD, slow-marked)

**Files:**
- Create: `semanticsd/extractors/audio.py`
- Modify: `semanticsd/extractors/__init__.py`
- Create: `tests/test_extractors_audio.py`

The extractor catches `RuntimeError` (ffmpeg missing) and `OSError` and degrades. The "real" test is slow-marked because it loads a Whisper model.

- [ ] **Step 1: Write failing tests**

`tests/test_extractors_audio.py`:
```python
import pytest
from semanticsd.extractors.audio import AudioExtractor
from semanticsd.extractors import registry
from tests._fixtures import make_wav_silence


def test_graceful_degrade_when_whisper_fails(tmp_path, monkeypatch):
    p = make_wav_silence(tmp_path)

    class BoomModel:
        def transcribe(self, *a, **kw):
            raise RuntimeError("ffmpeg not found")

    monkeypatch.setattr(
        "semanticsd.extractors.audio.WhisperModel",
        lambda *a, **kw: BoomModel(),
    )
    out = AudioExtractor().extract(p)
    assert out.file_type == "audio"
    assert out.segments == []
    assert "transcribe_error" in out.metadata


def test_registry_picks_wav(tmp_path):
    p = make_wav_silence(tmp_path)
    assert isinstance(registry.get_extractor(p), AudioExtractor)


@pytest.mark.slow
def test_real_transcription(tmp_path):
    """Tiny silent wav — Whisper should return empty or near-empty segments."""
    p = make_wav_silence(tmp_path, duration_s=0.5)
    out = AudioExtractor(model_size="tiny").extract(p)
    assert out.file_type == "audio"
    # Silent input should produce no segments OR all-empty text.
    assert all(not s.text.strip() for s in out.segments) or out.segments == []
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_extractors_audio.py -v -m "not slow"
```

- [ ] **Step 3: Implement `semanticsd/extractors/audio.py`**

```python
"""Audio extractor — faster-whisper transcription. Graceful degrade if model
fails to load or ffmpeg is missing."""
from __future__ import annotations
import logging
from pathlib import Path
from faster_whisper import WhisperModel
from semanticsd.extractors.base import Extractor, ExtractedDoc, ExtractedSegment
from semanticsd.extractors.registry import register

log = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"


@register
class AudioExtractor(Extractor):
    file_type = "audio"
    extensions = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".mp4a")

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        self.model_size = model_size
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            # device="cpu", compute_type="int8" keeps the load light.
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def extract(self, path: Path) -> ExtractedDoc:
        try:
            model = self._ensure_model()
            segments_iter, _info = model.transcribe(str(path), beam_size=1)
            wsegs = list(segments_iter)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            log.warning("Whisper transcription failed for %s: %s", path, e)
            return ExtractedDoc(
                path=str(path),
                file_type=self.file_type,
                segments=[],
                metadata={"transcribe_error": str(e)},
            )

        out_segments: list[ExtractedSegment] = []
        cursor = 0
        for s in wsegs:
            text = (s.text or "").strip()
            if not text:
                continue
            seg_bytes = len(text.encode("utf-8"))
            out_segments.append(ExtractedSegment(
                text=text,
                byte_start=cursor,
                byte_end=cursor + seg_bytes,
                metadata={"start_s": s.start, "end_s": s.end},
            ))
            cursor += seg_bytes + 1
        return ExtractedDoc(
            path=str(path),
            file_type=self.file_type,
            segments=out_segments,
        )
```

- [ ] **Step 4: Update `__init__.py`** — append `from semanticsd.extractors import audio  # noqa: F401`

- [ ] **Step 5: Run + commit**

```bash
.venv/bin/pytest tests/test_extractors_audio.py -v -m "not slow"
git add semanticsd/extractors/audio.py semanticsd/extractors/__init__.py tests/test_extractors_audio.py
git commit -m "feat(extractors): audio transcription via faster-whisper, graceful degrade"
```

---

## Task 16: Hasher + dedup query (TDD)

**Files:**
- Create: `semanticsd/pipeline/__init__.py` (empty)
- Create: `semanticsd/pipeline/hasher.py`
- Create: `tests/test_pipeline_hasher.py`

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline_hasher.py`:
```python
from semanticsd.pipeline.hasher import sha256_hex, normalize_for_hash, find_existing_embedding
from semanticsd.db import connection, migrations


def test_sha256_hex_is_deterministic():
    assert sha256_hex("hello") == sha256_hex("hello")


def test_sha256_hex_changes_with_content():
    assert sha256_hex("hello") != sha256_hex("world")


def test_normalize_collapses_whitespace_and_lowercases():
    a = normalize_for_hash("  Hello\nWorld  \t")
    b = normalize_for_hash("hello world")
    assert a == b


def test_find_existing_embedding_returns_none_when_absent(tmp_path):
    db = tmp_path / "h.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    res = find_existing_embedding(conn, content_hash="abc", provider_id="local",
                                  model_id="m", dim=384)
    assert res is None


def test_find_existing_embedding_returns_chunk_id_when_present(tmp_path):
    """Insert a fake chunk + embedding_meta + vec_embeddings, then look it up."""
    import struct
    db = tmp_path / "h.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    conn.execute(
        "INSERT INTO files(id, path, modified_at, size, file_type, indexed_at) "
        "VALUES (1, '/x', 0, 0, 'text', 0)"
    )
    conn.execute(
        "INSERT INTO chunks(id, file_id, chunk_index, text, content_hash, byte_start, byte_end) "
        "VALUES (1, 1, 0, 'hello', 'abc', 0, 5)"
    )
    blob = struct.pack("384f", *([0.1] * 384))
    conn.execute("INSERT INTO vec_embeddings(rowid, embedding) VALUES (1, ?)", (blob,))
    conn.execute(
        "INSERT INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
        "VALUES (1, 'local', 'm', 384, 'abc')"
    )
    res = find_existing_embedding(conn, content_hash="abc", provider_id="local",
                                  model_id="m", dim=384)
    assert res == 1
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_hasher.py -v
```

- [ ] **Step 3: Implement `semanticsd/pipeline/hasher.py`**

```python
"""Content hashing + dedup lookup against (content_hash, provider, model, dim)."""
from __future__ import annotations
import hashlib
import re
import sqlite3


_WS_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Lowercase + collapse whitespace so trivial variants dedup."""
    return _WS_RE.sub(" ", text.strip().lower())


def sha256_hex(text: str) -> str:
    """SHA-256 of normalized text, hex-encoded. Stable across runs."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def find_existing_embedding(
    conn: sqlite3.Connection,
    content_hash: str,
    provider_id: str,
    model_id: str,
    dim: int,
) -> int | None:
    """Return the chunk_id of an existing embedding for this triplet+hash, or None.

    Used to skip re-embedding identical content (cost protection).
    """
    row = conn.execute(
        "SELECT chunk_id FROM embedding_meta "
        "WHERE content_hash = ? AND provider_id = ? AND model_id = ? AND dim = ? "
        "LIMIT 1",
        (content_hash, provider_id, model_id, dim),
    ).fetchone()
    return int(row[0]) if row else None
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_hasher.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/pipeline/__init__.py semanticsd/pipeline/hasher.py tests/test_pipeline_hasher.py
git commit -m "feat(pipeline): hasher + dedup lookup"
```

---

## Task 17: Sliding-window chunker (TDD)

**Files:**
- Create: `semanticsd/pipeline/chunker.py`
- Create: `tests/test_pipeline_chunker.py`

Token counting uses character length / 4 as a coarse approximation (matches the embedder estimate_tokens heuristic). We chunk by approximate tokens, not exact tiktoken counts, to keep the chunker provider-agnostic.

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline_chunker.py`:
```python
from semanticsd.pipeline.chunker import SlidingWindowChunker, Chunk


def test_short_text_returns_one_chunk():
    c = SlidingWindowChunker(window_tokens=512, overlap_tokens=64)
    chunks = c.chunk("Hello world.", base_byte_offset=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].byte_start == 0
    assert chunks[0].byte_end == len("Hello world.".encode("utf-8"))


def test_long_text_splits_into_overlapping_windows():
    body = ("word " * 10000).strip()  # ~50000 chars => ~12500 tokens
    c = SlidingWindowChunker(window_tokens=512, overlap_tokens=64)
    chunks = c.chunk(body, base_byte_offset=0)
    assert len(chunks) > 1
    # Each chunk respects approx token budget (chars/4 <= window_tokens).
    for ch in chunks:
        assert len(ch.text) // 4 <= 512 + 64  # tolerance
    # Successive chunks overlap a little.
    assert chunks[0].byte_end > chunks[1].byte_start


def test_byte_offsets_continuous():
    body = "First part. Second part. Third part."
    c = SlidingWindowChunker(window_tokens=2, overlap_tokens=0)  # tiny windows
    chunks = c.chunk(body, base_byte_offset=100)
    assert chunks[0].byte_start == 100
    # Byte offsets are within the original text frame, shifted by base.
    for ch in chunks:
        assert ch.byte_start >= 100
        assert ch.byte_end > ch.byte_start


def test_empty_text_returns_no_chunks():
    c = SlidingWindowChunker()
    assert c.chunk("", base_byte_offset=0) == []
    assert c.chunk("   \n\t", base_byte_offset=0) == []
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_chunker.py -v
```

- [ ] **Step 3: Implement `semanticsd/pipeline/chunker.py`**

```python
"""Sliding-window chunker. Token count approximated by chars/4 to stay
provider-agnostic — exact tiktoken-based counting can be plugged in later."""
from __future__ import annotations
from dataclasses import dataclass


CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    byte_start: int   # absolute, includes base_byte_offset
    byte_end: int


class SlidingWindowChunker:
    def __init__(self, window_tokens: int = 512, overlap_tokens: int = 64):
        self.window_chars = window_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    def chunk(self, text: str, base_byte_offset: int) -> list[Chunk]:
        text = text or ""
        if not text.strip():
            return []
        n = len(text)
        step = max(1, self.window_chars - self.overlap_chars)
        chunks: list[Chunk] = []
        start = 0
        while start < n:
            end = min(n, start + self.window_chars)
            piece = text[start:end]
            # Convert char offsets to byte offsets within `text` (UTF-8 safe).
            byte_start = base_byte_offset + len(text[:start].encode("utf-8"))
            byte_end = base_byte_offset + len(text[:end].encode("utf-8"))
            chunks.append(Chunk(text=piece, byte_start=byte_start, byte_end=byte_end))
            if end == n:
                break
            start += step
        return chunks
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_chunker.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/pipeline/chunker.py tests/test_pipeline_chunker.py
git commit -m "feat(pipeline): sliding-window chunker"
```

---

## Task 18: `.semanticsdignore` + walker with size-limit (TDD)

**Files:**
- Create: `semanticsd/pipeline/ignore.py`
- Create: `semanticsd/pipeline/walker.py`
- Create: `tests/test_pipeline_ignore.py`
- Create: `tests/test_pipeline_walker.py`

- [ ] **Step 1: Write failing tests for ignore**

`tests/test_pipeline_ignore.py`:
```python
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
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_ignore.py -v
```

- [ ] **Step 3: Implement `semanticsd/pipeline/ignore.py`**

```python
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
        # pathspec wants POSIX-style forward slashes regardless of OS.
        s = str(path).replace("\\", "/")
        return self._spec.match_file(s)
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_ignore.py -v
```

- [ ] **Step 5: Write failing tests for walker**

`tests/test_pipeline_walker.py`:
```python
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
    big.write_bytes(b"\x00" * (3 * 1024 * 1024))  # 3 MB
    small = tmp_path / "small.txt"
    small.write_text("hi")

    matcher = IgnoreMatcher.from_defaults()
    files = list(walk_indexable(tmp_path, matcher, max_size_mb=1))  # 1 MB cap
    rels = sorted(f.relative_to(tmp_path).as_posix() for f in files)
    assert rels == ["small.txt"]


def test_walks_recursively(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b.txt").write_text("hi")
    matcher = IgnoreMatcher.from_defaults()
    files = list(walk_indexable(tmp_path, matcher, max_size_mb=50))
    rels = sorted(f.relative_to(tmp_path).as_posix() for f in files)
    assert rels == ["a/b.txt"]
```

- [ ] **Step 6: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_walker.py -v
```

- [ ] **Step 7: Implement `semanticsd/pipeline/walker.py`**

```python
"""Directory walker that yields indexable file paths."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterator
from semanticsd.pipeline.ignore import IgnoreMatcher


def walk_indexable(root: Path, matcher: IgnoreMatcher, max_size_mb: int) -> Iterator[Path]:
    """Yield each file under `root` that:
      - exists and is a regular file
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
            if matcher.is_ignored(sub) or matcher.is_ignored(sub.with_name(name + "/")):
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
```

- [ ] **Step 8: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_walker.py -v
```

- [ ] **Step 9: Commit**

```bash
git add semanticsd/pipeline/ignore.py semanticsd/pipeline/walker.py \
        tests/test_pipeline_ignore.py tests/test_pipeline_walker.py
git commit -m "feat(pipeline): .semanticsdignore + size-limited walker"
```

---

## Task 19: Indexer orchestrator (TDD)

**Files:**
- Create: `semanticsd/pipeline/indexer.py`
- Create: `tests/test_pipeline_indexer.py`

The indexer takes a path or inline content, walks/extracts, chunks, hashes, upserts the `files` and `chunks` rows, and queues `jobs` rows. It does **not** call the embedder — that's the worker's job in Task 20.

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline_indexer.py`:
```python
from pathlib import Path
import time
from semanticsd.db import connection, migrations
from semanticsd.pipeline.indexer import Indexer
from tests._fixtures import make_text, make_markdown


def _fresh_db(tmp_path: Path):
    db = tmp_path / "x.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    return conn


def test_index_path_creates_files_and_chunks_and_jobs(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    make_text(src, body="Hello world. Second sentence.")

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(src)
    assert stats["files_indexed"] == 1
    assert stats["chunks_created"] >= 1
    assert stats["jobs_queued"] == stats["chunks_created"]

    rows = conn.execute("SELECT count(*) FROM files").fetchone()
    assert rows[0] == 1
    rows = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert rows[0] >= 1
    rows = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()
    assert rows[0] >= 1


def test_index_path_skips_unsupported(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "binary.dat").write_bytes(b"\x00\x01" * 10)
    make_markdown(src)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(src)
    assert stats["files_indexed"] == 1
    assert stats["files_skipped_unsupported"] == 1


def test_re_index_same_content_does_not_duplicate(tmp_path):
    conn = _fresh_db(tmp_path)
    src = tmp_path / "corpus"
    src.mkdir()
    f = make_text(src, body="Stable content.")

    idx = Indexer(conn=conn, max_file_size_mb=50)
    s1 = idx.index_path(src)
    s2 = idx.index_path(src)

    # Second pass: file is already in `files`, no new chunks queued
    # (mtime + size match), so jobs_queued == 0.
    files_count = conn.execute("SELECT count(*) FROM files").fetchone()[0]
    assert files_count == 1
    assert s2["jobs_queued"] == 0


def test_index_inline_creates_synthetic_path(tmp_path):
    conn = _fresh_db(tmp_path)
    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_inline(source="conversation://1", content="hello inline", metadata={"role": "user"})
    assert stats["chunks_created"] >= 1
    assert stats["jobs_queued"] >= 1
    row = conn.execute("SELECT path, file_type FROM files").fetchone()
    assert row[0] == "conversation://1"
    assert row[1] == "inline"
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_indexer.py -v
```

- [ ] **Step 3: Implement `semanticsd/pipeline/indexer.py`**

```python
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
from semanticsd.pipeline.hasher import sha256_hex
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
        """Walk `path` (file or dir), extract+chunk+queue everything indexable."""
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
        """Index inline content under a synthetic 'path' (e.g. `conversation://1`)."""
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

    # ----- internals -----

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

        # Upsert files row.
        if existing is not None:
            file_id = int(existing[0])
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

        chunks_created = 0
        jobs_queued = 0
        for seg in doc.segments:
            cc, jq = self._chunk_segment_into_jobs(
                file_id=file_id, text=seg.text, base_offset=seg.byte_start,
            )
            chunks_created += cc
            jobs_queued += jq
        return {"chunks": chunks_created, "jobs": jobs_queued}

    def _chunk_segment_into_jobs(self, file_id: int, text: str, base_offset: int) -> tuple[int, int]:
        chunks: list[Chunk] = self.chunker.chunk(text, base_byte_offset=base_offset)
        if not chunks:
            return 0, 0
        # Compute the next chunk_index for this file.
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
                "INSERT INTO jobs(chunk_id, status, attempts, created_at, updated_at) "
                "VALUES (?, 'pending', 0, ?, ?)",
                (chunk_id, now, now),
            )
            chunks_created += 1
            jobs_queued += 1
            next_idx += 1
        return chunks_created, jobs_queued
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_indexer.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/pipeline/indexer.py tests/test_pipeline_indexer.py
git commit -m "feat(pipeline): indexer orchestrator (extract→chunk→hash→queue)"
```

---

## Task 20: Job-queue worker (TDD)

**Files:**
- Create: `semanticsd/pipeline/worker.py`
- Create: `tests/test_pipeline_worker.py`

The worker pulls pending jobs, runs the embedder, persists vectors. Provides `drain_once()` (sync, one batch), `reset_stale()` (in_flight → pending on startup), and `run_forever()` (async loop calling drain + sleep).

- [ ] **Step 1: Write failing tests**

`tests/test_pipeline_worker.py`:
```python
import struct
from semanticsd.db import connection, migrations
from semanticsd.embedders.base import Embedder, EmbedResult
from semanticsd.pipeline.worker import Worker
from semanticsd.pipeline.indexer import Indexer
from tests._fixtures import make_text


class FakeEmbedder(Embedder):
    provider_id = "fake"
    model_id = "fake-1"
    dim = 384
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def __init__(self):
        self.calls = 0

    def embed(self, texts, kind):
        self.calls += 1
        return EmbedResult(
            vectors=[[float(i) / 1000.0] * 384 for i in range(len(texts))],
            input_tokens=sum(len(t) // 4 for t in texts),
        )

    def health_check(self):
        return (True, "ok")

    def estimate_tokens(self, texts):
        return sum(len(t) // 4 for t in texts)


def _index_one(tmp_path):
    db = tmp_path / "w.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    src = tmp_path / "src"
    src.mkdir()
    make_text(src, body="Some content to embed.")
    Indexer(conn=conn, max_file_size_mb=50).index_path(src)
    return conn


def test_drain_once_processes_pending_jobs(tmp_path):
    conn = _index_one(tmp_path)
    emb = FakeEmbedder()
    w = Worker(conn=conn, embedder=emb, batch_size=10)
    n = w.drain_once()
    assert n >= 1
    # All jobs done.
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
    # Embeddings exist.
    emb_count = conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0]
    assert emb_count >= 1


def test_dedup_skips_redundant_embed_calls(tmp_path):
    """Re-indexing the same content under a different file shouldn't re-embed."""
    conn = _index_one(tmp_path)
    emb = FakeEmbedder()
    w = Worker(conn=conn, embedder=emb, batch_size=10)
    w.drain_once()
    calls_after_first = emb.calls

    # Re-index by inserting another file with identical text.
    src2 = tmp_path / "src2"
    src2.mkdir()
    make_text(src2, name="dupe.txt", body="Some content to embed.")
    Indexer(conn=conn, max_file_size_mb=50).index_path(src2)
    w.drain_once()

    # No new embed call — content_hash dedup hit.
    assert emb.calls == calls_after_first


def test_reset_stale_resets_in_flight(tmp_path):
    conn = _index_one(tmp_path)
    conn.execute("UPDATE jobs SET status='in_flight'")
    Worker(conn=conn, embedder=FakeEmbedder()).reset_stale()
    in_flight = conn.execute("SELECT count(*) FROM jobs WHERE status='in_flight'").fetchone()[0]
    assert in_flight == 0
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending >= 1


def test_drain_once_returns_zero_when_no_jobs(tmp_path):
    db = tmp_path / "empty.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)
    w = Worker(conn=conn, embedder=FakeEmbedder())
    assert w.drain_once() == 0
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_pipeline_worker.py -v
```

- [ ] **Step 3: Implement `semanticsd/pipeline/worker.py`**

```python
"""Job-queue worker: drain pending jobs by calling the embedder, persist
vectors via sqlite-vec. Idempotent and resumable.

drain_once(): one batch, returns number of jobs marked 'done'.
reset_stale(): on startup — flip in_flight back to pending so crashed work resumes.
run_forever(): async loop that calls drain_once + sleeps.
"""
from __future__ import annotations
import asyncio
import logging
import sqlite3
import struct
import time
from semanticsd.embedders.base import Embedder
from semanticsd.pipeline.hasher import find_existing_embedding

log = logging.getLogger(__name__)


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class Worker:
    def __init__(self, conn: sqlite3.Connection, embedder: Embedder, batch_size: int = 128, max_attempts: int = 5):
        self.conn = conn
        self.embedder = embedder
        self.batch_size = batch_size
        self.max_attempts = max_attempts

    def reset_stale(self) -> None:
        self.conn.execute("UPDATE jobs SET status='pending' WHERE status='in_flight'")

    def drain_once(self) -> int:
        rows = self.conn.execute(
            "SELECT j.id, j.chunk_id, c.text, c.content_hash "
            "FROM jobs j JOIN chunks c ON c.id = j.chunk_id "
            "WHERE j.status = 'pending' "
            "ORDER BY j.id LIMIT ?",
            (self.batch_size,),
        ).fetchall()
        if not rows:
            return 0

        job_ids = [int(r[0]) for r in rows]
        placeholders = ",".join("?" for _ in job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='in_flight', updated_at=? WHERE id IN ({placeholders})",
            [int(time.time()), *job_ids],
        )

        # Dedup: split into already-embedded vs need-embed.
        to_embed: list[tuple[int, int, str, str]] = []  # (job_id, chunk_id, text, content_hash)
        cached: list[tuple[int, int, int]] = []         # (job_id, target_chunk_id, source_chunk_id)
        for jid, cid, text, chash in rows:
            existing_chunk_id = find_existing_embedding(
                self.conn,
                content_hash=chash,
                provider_id=self.embedder.provider_id,
                model_id=self.embedder.model_id,
                dim=self.embedder.dim,
            )
            if existing_chunk_id is not None and int(existing_chunk_id) != int(cid):
                cached.append((int(jid), int(cid), int(existing_chunk_id)))
            else:
                to_embed.append((int(jid), int(cid), text, chash))

        # Embed the not-cached ones.
        if to_embed:
            try:
                texts = [t[2] for t in to_embed]
                result = self.embedder.embed(texts, kind="doc")
            except Exception as e:
                log.warning("embedder failed: %s", e)
                self._mark_failed(job_ids, str(e))
                return 0
            for (jid, cid, _t, chash), vec in zip(to_embed, result.vectors):
                self.conn.execute(
                    "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                    (cid, _vec_to_blob(list(vec))),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (cid, self.embedder.provider_id, self.embedder.model_id, self.embedder.dim, chash),
                )

        # For cached hits, copy the existing vector blob to the new chunk_id.
        for jid, target_cid, source_cid in cached:
            row = self.conn.execute(
                "SELECT embedding FROM vec_embeddings WHERE rowid = ?", (source_cid,)
            ).fetchone()
            if row is None:
                continue
            self.conn.execute(
                "INSERT OR REPLACE INTO vec_embeddings(rowid, embedding) VALUES (?, ?)",
                (target_cid, row[0]),
            )
            row2 = self.conn.execute(
                "SELECT content_hash FROM embedding_meta WHERE chunk_id = ?", (source_cid,)
            ).fetchone()
            chash = row2[0] if row2 else ""
            self.conn.execute(
                "INSERT OR REPLACE INTO embedding_meta(chunk_id, provider_id, model_id, dim, content_hash) "
                "VALUES (?, ?, ?, ?, ?)",
                (target_cid, self.embedder.provider_id, self.embedder.model_id, self.embedder.dim, chash),
            )

        # Mark all jobs done.
        self.conn.execute(
            f"UPDATE jobs SET status='done', updated_at=? WHERE id IN ({placeholders})",
            [int(time.time()), *job_ids],
        )
        return len(job_ids)

    def _mark_failed(self, job_ids: list[int], error: str) -> None:
        placeholders = ",".join("?" for _ in job_ids)
        self.conn.execute(
            f"UPDATE jobs SET status='pending', attempts = attempts + 1, "
            f"last_error = ?, updated_at = ? WHERE id IN ({placeholders})",
            [error, int(time.time()), *job_ids],
        )
        # Demote to 'failed' if past max_attempts.
        self.conn.execute(
            "UPDATE jobs SET status='failed' WHERE attempts >= ?",
            (self.max_attempts,),
        )

    async def run_forever(self, poll_interval_s: float = 2.0) -> None:
        self.reset_stale()
        while True:
            try:
                processed = self.drain_once()
            except Exception as e:
                log.error("worker drain crashed: %s", e)
                processed = 0
            if processed == 0:
                await asyncio.sleep(poll_interval_s)
```

- [ ] **Step 4: Run, expect pass**

```bash
.venv/bin/pytest tests/test_pipeline_worker.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add semanticsd/pipeline/worker.py tests/test_pipeline_worker.py
git commit -m "feat(pipeline): job-queue worker with content-hash dedup"
```

---

## Task 21: `POST /v1/index` route (TDD)

**Files:**
- Create: `semanticsd/server/routes/index.py`
- Modify: `semanticsd/server/app.py`
- Create: `tests/test_index_route.py`

The route accepts either `{path}` or `{source, content, metadata?}`. It builds an Indexer against the daemon's DB connection, runs `index_path` or `index_inline`, and returns counts. **Does not** drain the worker — that's a continuous background task (Plan 4 will start it via lifespan; for now the user runs `ssearch --index <path>` and a separate command/process drains).

To keep the test simple, we expose a `drain` query parameter that tells the route to call `worker.drain_once()` once after queuing. Default is `drain=false`.

- [ ] **Step 1: Write failing tests**

`tests/test_index_route.py`:
```python
import pytest
from fastapi.testclient import TestClient
from semanticsd.server import app as server_app
from semanticsd.server import auth
from tests._fixtures import make_text


class FakeEmb:
    provider_id = "fake"
    model_id = "fake-1"
    dim = 384
    supports_kind = False
    cost_per_million_input_tokens_usd = 0.0

    def embed(self, texts, kind):
        from semanticsd.embedders.base import EmbedResult
        return EmbedResult(vectors=[[0.1] * 384 for _ in texts], input_tokens=1)

    def health_check(self):
        return (True, "fake ok")

    def estimate_tokens(self, texts):
        return 1


@pytest.fixture
def fake_active(monkeypatch):
    import semanticsd.embedders as emb_pkg
    monkeypatch.setattr(emb_pkg, "get_active_embedder", lambda **kw: FakeEmb())


def test_index_unauthed(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post("/v1/index", json={"path": "/tmp"})
    assert r.status_code == 401


def test_index_path(tmp_app_support, monkeypatch, tmp_path, fake_active):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    src = tmp_path / "corpus"
    src.mkdir()
    make_text(src, body="Hello world.")

    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={"path": str(src), "drain": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["files_indexed"] == 1
    assert body["jobs_queued"] >= 1
    assert body["drained"] >= 1


def test_index_inline(tmp_app_support, monkeypatch, fake_active):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={"source": "test://1", "content": "inline text"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["files_indexed"] == 1


def test_index_missing_args_returns_400(monkeypatch):
    monkeypatch.setattr(auth, "_token_cache", "secret")
    client = TestClient(server_app.create_app())
    r = client.post(
        "/v1/index",
        headers={"X-Auth-Token": "secret"},
        json={},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run, expect fail**

```bash
.venv/bin/pytest tests/test_index_route.py -v
```

- [ ] **Step 3: Implement `semanticsd/server/routes/index.py`**

```python
"""POST /v1/index — manual indexing trigger."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from semanticsd.server.auth import require_token
from semanticsd.db import connection, migrations
from semanticsd import paths
from semanticsd import embedders as emb_pkg
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker

router = APIRouter()


class IndexRequest(BaseModel):
    path: str | None = None
    source: str | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None
    drain: bool = False  # if True, worker.drain_once() runs once after queuing


@router.post("/index", dependencies=[Depends(require_token)])
def index(req: IndexRequest) -> dict[str, Any]:
    if not req.path and not (req.source and req.content is not None):
        raise HTTPException(status_code=400, detail="provide either 'path' or both 'source' and 'content'")

    conn = connection.get_connection(paths.db_path())
    migrations.apply(conn)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    if req.path:
        stats = idx.index_path(Path(req.path))
    else:
        stats = idx.index_inline(source=req.source or "", content=req.content or "", metadata=req.metadata)

    drained = 0
    if req.drain:
        emb = emb_pkg.get_active_embedder()
        if emb is None:
            raise HTTPException(status_code=503, detail="no embedder configured")
        worker = Worker(conn=conn, embedder=emb)
        drained = worker.drain_once()

    return {**stats, "drained": drained}
```

- [ ] **Step 4: Wire router into `semanticsd/server/app.py`**

```python
"""FastAPI app factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from semanticsd import __version__
from semanticsd.server.routes import health, presets, embedder_test, index as index_route


def create_app() -> FastAPI:
    app = FastAPI(
        title="SemanticsD",
        version=__version__,
        description="Local semantic search daemon for macOS.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router, prefix="/v1")
    app.include_router(presets.router, prefix="/v1")
    app.include_router(embedder_test.router, prefix="/v1")
    app.include_router(index_route.router, prefix="/v1")
    return app
```

- [ ] **Step 5: Run, expect pass**

```bash
.venv/bin/pytest tests/test_index_route.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add semanticsd/server/routes/index.py semanticsd/server/app.py tests/test_index_route.py
git commit -m "feat(server): POST /v1/index endpoint"
```

---

## Task 22: `ssearch --index <path>` CLI

**Files:**
- Modify: `semanticsd/cli.py`

- [ ] **Step 1: Add the `--index` flag to `ssearch_root`**

In `semanticsd/cli.py`, add a new option to the `ssearch_root` callback. Insert this option **between** `test_embedder` and `json_output`:

```python
    index_path: str = typer.Option(
        "", "--index", metavar="PATH",
        help="Index a file or directory; runs the worker once when done.",
    ),
```

Then add this branch **between** the `test_embedder` branch and the final `if ctx.invoked_subcommand is None:` fallthrough:

```python
    if index_path:
        try:
            with _client() as c:
                r = c.post(
                    "/v1/index",
                    json={"path": index_path, "drain": True},
                    timeout=600.0,
                )
                r.raise_for_status()
                body = r.json()
        except httpx.HTTPError as e:
            typer.echo(f"ERROR: cannot reach daemon: {e}", err=True)
            raise typer.Exit(3)
        if json_output:
            typer.echo(json.dumps(body, indent=2))
        else:
            typer.echo(f"indexed:   {body.get('files_indexed', 0)} files")
            typer.echo(f"chunks:    {body.get('chunks_created', 0)}")
            typer.echo(f"queued:    {body.get('jobs_queued', 0)} jobs")
            typer.echo(f"drained:   {body.get('drained', 0)} jobs in this run")
            if body.get("files_skipped_unsupported"):
                typer.echo(f"skipped:   {body['files_skipped_unsupported']} unsupported files")
            if body.get("files_skipped_unchanged"):
                typer.echo(f"unchanged: {body['files_skipped_unchanged']} files")
        return
```

The `_client()` helper in cli.py uses a 10s default timeout; for `--index` we override to 600s because indexing + embedding can be slow.

- [ ] **Step 2: Verify both CLI apps still import**

```bash
.venv/bin/python -c "from semanticsd.cli import semanticsd_app, ssearch_app; print('ok')"
.venv/bin/ssearch --help | grep index
```
Expected: `ok` and a line for `--index`.

- [ ] **Step 3: Run full suite — no regressions**

```bash
.venv/bin/pytest -q -m "not slow"
```

- [ ] **Step 4: Commit**

```bash
git add semanticsd/cli.py
git commit -m "feat(cli): ssearch --index <path>"
```

---

## Task 23: End-to-end indexing test against ./sandbox (TDD, slow-marked)

**Files:**
- Create: `tests/test_e2e_index.py`

This test points the indexer at the existing `sandbox/` directory (created in Plan 2 Task 2), runs the real LocalEmbedder, drains the worker, and asserts that vectors land in `vec_embeddings`.

- [ ] **Step 1: Write the test**

`tests/test_e2e_index.py`:
```python
"""End-to-end: real LocalEmbedder + indexer + worker against ./sandbox/.

Slow-marked because it loads the real bge-small-en-v1.5 model and embeds
multiple sandbox files.
"""
import pytest
from pathlib import Path
from semanticsd.db import connection, migrations
from semanticsd.embedders.local import LocalEmbedder
from semanticsd.pipeline.indexer import Indexer
from semanticsd.pipeline.worker import Worker


pytestmark = pytest.mark.slow


def test_index_sandbox_end_to_end(tmp_path):
    """Run the full pipeline against the sandbox seed corpus."""
    sandbox = Path(__file__).resolve().parents[1] / "sandbox"
    assert sandbox.exists(), "sandbox/ should exist (Plan 2 Task 2 fixture)"

    db = tmp_path / "e2e.db"
    conn = connection.get_connection(db)
    migrations.apply(conn)

    idx = Indexer(conn=conn, max_file_size_mb=50)
    stats = idx.index_path(sandbox)
    assert stats["files_indexed"] >= 4  # README + alpha + beta + hello.py + design.txt
    assert stats["chunks_created"] >= stats["files_indexed"]
    assert stats["jobs_queued"] == stats["chunks_created"]

    embedder = LocalEmbedder()
    worker = Worker(conn=conn, embedder=embedder, batch_size=128)

    # Drain until empty.
    total_drained = 0
    while True:
        n = worker.drain_once()
        total_drained += n
        if n == 0:
            break
    assert total_drained == stats["jobs_queued"]

    # All embeddings present.
    emb_count = conn.execute("SELECT count(*) FROM vec_embeddings").fetchone()[0]
    assert emb_count == stats["chunks_created"]
    pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    assert pending == 0
```

- [ ] **Step 2: Run with `-m slow`**

```bash
.venv/bin/pytest tests/test_e2e_index.py -v -m slow
```
Expected: 1 passed (5-30 seconds depending on model cache).

- [ ] **Step 3: Run full suite without slow — no regressions**

```bash
.venv/bin/pytest -q -m "not slow"
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_index.py
git commit -m "test(e2e): full pipeline against sandbox corpus"
```

---

## Task 24: Manual install verification (controller-driven)

This task is the human-driven smoke of the full Plan-3 stack. No code change.

- [ ] **Step 1: Reinstall venv with new deps + bounce daemon**

```bash
.venv/bin/pip install -r requirements.txt --upgrade
./scripts/install.sh --uninstall
./scripts/install.sh
```

- [ ] **Step 2: Index the sandbox via the CLI**

```bash
.venv/bin/ssearch --index ./sandbox
```
Expected: output like
```
indexed:   ~4 files
chunks:    ~5
queued:    ~5 jobs
drained:   ~5 jobs in this run
skipped:   1 unsupported files     # README is .md (supported); .DS_Store is filtered
```

- [ ] **Step 3: Verify the database**

```bash
sqlite3 ~/Library/Application\ Support/semanticsd/index.db <<EOF
SELECT count(*) AS files FROM files;
SELECT count(*) AS chunks FROM chunks;
SELECT count(*) AS embeddings FROM vec_embeddings;
SELECT count(*) AS done FROM jobs WHERE status='done';
SELECT count(*) AS pending FROM jobs WHERE status='pending';
EOF
```
Expected: files >= 4, chunks >= 4, embeddings == chunks, done == chunks, pending == 0.

- [ ] **Step 4: Re-run the index command — should be idempotent**

```bash
.venv/bin/ssearch --index ./sandbox
```
Expected: `unchanged: ~4 files`, no new jobs.

- [ ] **Step 5: Verify status reflects new doc count**

```bash
.venv/bin/ssearch --status
```
Expected: `doc_count: 4` (or however many files landed).

- [ ] **Step 6: Add a new file to sandbox, re-index — only the new file is processed**

```bash
echo "A new note about gamma protocols." > sandbox/notes/gamma.md
.venv/bin/ssearch --index ./sandbox
```
Expected: `indexed: 1 files`, `unchanged: 4 files`. Database file count grows by 1.

- [ ] **Step 7: Cleanup the test file**

```bash
rm sandbox/notes/gamma.md
```

If any step fails, fix and re-run. Plan 3 is complete only when all seven steps pass.

---

## What's Next (Plan 3.5 + Plan 4 preview)

**Plan 3.5 (small follow-ups):** archive (.zip walker), video (ffmpeg → audio → Whisper), CLIP image-vision embedding (separate vector path for images), additional embedder providers (voyage, cohere, gemini, vertex, bedrock).

**Plan 4:** FSEvents file watcher + power modes. The watcher detects file changes in real time and queues indexing automatically. The worker becomes a long-running async task started by the FastAPI lifespan. Power-saver mode pauses the watcher and runs periodic diff-reindex on a schedule.

**Plans 5–6:** search engine (3 modes: semantic / filename / grep) + cost tracking + budgets.
