"""Load PDF, DOCX, TXT and Markdown files into LangChain `Document` objects.

Each returned Document covers one (page, section) span of the source file and
carries metadata used for citations: file name, page number (PDF only) and the
nearest section heading.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Numbered headings such as "1.", "2.3", "Section 4", "IV." followed by a title.
_NUMBERED_HEADING = re.compile(r"^((\d+(\.\d+)*\.?)|(section|article|part)\s+\d+[:.]?|[IVX]+\.)\s+\S", re.IGNORECASE)
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S")


class DocumentLoadError(Exception):
    """Raised when a file cannot be read or contains no usable text."""


def is_heading(line: str) -> bool:
    """Heuristically decide whether a single line of text is a section heading."""
    text = line.strip()
    if not text or len(text) > 90:
        return False
    if _MARKDOWN_HEADING.match(text):
        return True
    if text[-1] in ".,;" or text.count(" ") > 12:
        return False
    if _NUMBERED_HEADING.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 3 and text.upper() == text and len(text.split()) <= 10:
        return True
    return _is_title_case(text)


_MINOR_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "&", "-"}


def _is_title_case(text: str) -> bool:
    """Short Title Case lines ("Before Day One") are usually headings in extracted PDF/TXT text."""
    words = text.split()
    if not 2 <= len(words) <= 6 or text[-1] in "?!":
        return False
    significant = [w for w in words if w.lower() not in _MINOR_WORDS]
    return bool(significant) and all(w[0].isupper() or w[0].isdigit() for w in significant) and words[0][0].isupper()


def merge_small_sections(sections: List[Tuple[str, str]], min_chars: int = 60) -> List[Tuple[str, str]]:
    """Fold tiny sections (e.g. a lone document title) into the following section."""
    merged: List[Tuple[str, str]] = []
    carry = ""
    for heading, body in sections:
        body = f"{carry}\n{body}".strip() if carry else body
        if len(body) < min_chars:
            carry = body
        else:
            carry = ""
            merged.append((heading, body))
    if carry:
        if merged:
            merged[-1] = (merged[-1][0], f"{merged[-1][1]}\n{carry}")
        else:
            merged.append((sections[-1][0], carry))
    return merged


def clean_heading(line: str) -> str:
    return line.strip().lstrip("#").strip().rstrip(":")


def split_into_sections(text: str, current_heading: str = "") -> List[Tuple[str, str]]:
    """Split text into (heading, body) pairs. Text before the first heading keeps `current_heading`."""
    sections: List[Tuple[str, str]] = []
    heading = current_heading
    buffer: List[str] = []
    for line in text.splitlines():
        if is_heading(line):
            if any(b.strip() for b in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = clean_heading(line)
            buffer = [line.strip()]
        else:
            buffer.append(line)
    if any(b.strip() for b in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _normalise(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_docs(sections: List[Tuple[str, str]], base_meta: dict, page: Optional[int] = None) -> List[Document]:
    docs = []
    for heading, body in merge_small_sections(sections):
        body = _normalise(body)
        if not body:
            continue
        meta = dict(base_meta)
        meta["section"] = heading or ""  # empty = no heading; never invent one
        if page is not None:
            meta["page"] = page
        docs.append(Document(page_content=body, metadata=meta))
    return docs


def _load_pdf(path: Path, base_meta: dict) -> List[Document]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # pragma: no cover - depends on file
                raise DocumentLoadError("PDF is password-protected") from exc
        docs: List[Document] = []
        heading = ""
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            sections = split_into_sections(text, heading)
            if sections:
                heading = sections[-1][0]
            docs.extend(_make_docs(sections, base_meta, page=page_number))
        base_meta["total_pages"] = len(reader.pages)
        for d in docs:
            d.metadata["total_pages"] = len(reader.pages)
        return docs
    except PdfReadError as exc:
        raise DocumentLoadError(f"Could not read PDF: {exc}") from exc


def _load_docx(path: Path, base_meta: dict) -> List[Document]:
    import docx
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = docx.Document(str(path))
    except (PackageNotFoundError, KeyError, ValueError) as exc:
        raise DocumentLoadError(f"Could not read DOCX: {exc}") from exc

    from docx.table import Table

    sections: List[Tuple[str, str]] = []
    heading = ""
    buffer: List[str] = []

    def flush():
        if any(b.strip() for b in buffer):
            sections.append((heading, "\n".join(buffer)))

    # Walk paragraphs and tables in document order so tables keep their section heading.
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in block.rows]
            buffer.extend(r for r in rows if r.strip(" |"))
            continue
        text = block.text.strip()
        if not text:
            continue
        style = (block.style.name if block.style is not None else "") or ""
        if style.lower().startswith(("heading", "title")) or is_heading(text):
            flush()
            heading = clean_heading(text)
            buffer = [text]
        else:
            prefix = "- " if "list" in style.lower() else ""
            buffer.append(prefix + text)
    flush()
    return _make_docs(sections, base_meta)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 always decodes
        raise DocumentLoadError("Could not decode text file")
    return text


def _load_txt(path: Path, base_meta: dict) -> List[Document]:
    return _make_docs(split_into_sections(_read_text(path)), base_meta)


def split_markdown_sections(text: str) -> List[Tuple[str, str]]:
    """Split Markdown on `#` headings only, ignoring lines inside fenced code blocks.

    Sub-headings are reported with their parent ("4. Refunds > Partial refunds") so a
    citation shows where in the document the excerpt sits. The H1 is treated as the
    document title and is not repeated in every section path.
    """
    sections: List[Tuple[str, str]] = []
    stack: List[Tuple[int, str]] = []
    heading = ""
    buffer: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        match = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if not match:
            buffer.append(line)
            continue
        if any(b.strip() for b in buffer):
            sections.append((heading, "\n".join(buffer).strip()))
        level, title = len(match.group(1)), match.group(2).strip()
        stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
        path = [t for lvl, t in stack if lvl > 1] or [title]
        heading = " > ".join(path)
        buffer = [line.strip()]
    if any(b.strip() for b in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


def _load_md(path: Path, base_meta: dict) -> List[Document]:
    return _make_docs(split_markdown_sections(_read_text(path)), base_meta)


def load_document(path: Path, extra_metadata: Optional[dict] = None) -> List[Document]:
    """Extract text from a supported file. Raises DocumentLoadError on failure or empty content."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.exists():
        raise DocumentLoadError(f"File not found: {path.name}")

    base_meta = {"source": path.name, "file_type": ext.lstrip(".")}
    base_meta.update(extra_metadata or {})

    loader = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt, ".md": _load_md}[ext]
    try:
        docs = loader(path, base_meta)
    except DocumentLoadError:
        raise
    except Exception as exc:
        # Log the type only - never the document contents.
        logger.warning("Failed to parse %s (%s)", path.name, type(exc).__name__)
        raise DocumentLoadError(f"Could not extract text from {path.name}: {type(exc).__name__}") from exc

    if not docs or sum(len(d.page_content) for d in docs) < 20:
        raise DocumentLoadError(
            f"No extractable text found in {path.name}. Scanned/image-only files need OCR before upload."
        )
    return docs
