"""Unified text extraction + chunking for the knowledge base.

Supports the document types contractors typically keep: PDF (pdfplumber),
DOCX (python-docx, optional), and plain text / Markdown. Extraction works from
raw bytes (Streamlit uploads) or a path.

Security note: we never log document text — only file names, sizes, and the
number of characters/chunks produced.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

from dragonpulse.config.logging_config import get_logger

logger = get_logger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json"}


class UnsupportedDocument(ValueError):
    """Raised when a document type cannot be extracted."""


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract plain text from uploaded ``data`` based on ``filename`` suffix.

    Raises
    ------
    UnsupportedDocument
        If the file type is not supported or extraction yields nothing.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf_bytes(data)
    elif suffix == ".docx":
        text = _extract_docx_bytes(data)
    elif suffix in _TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="replace")
    else:
        raise UnsupportedDocument(
            f"Unsupported file type '{suffix}'. Supported: PDF, DOCX, TXT, MD."
        )

    text = (text or "").strip()
    if not text:
        raise UnsupportedDocument(f"No extractable text found in '{filename}'.")
    logger.info("Extracted %d chars from %s", len(text), filename)
    return text


def extract_text_from_path(path: Path) -> str:
    """Extract text from a file on disk."""
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)


def _extract_pdf_bytes(data: bytes) -> str:
    import pdfplumber

    parts: List[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts)


def _extract_docx_bytes(data: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise UnsupportedDocument(
            "DOCX support requires 'python-docx' (pip install python-docx)."
        ) from exc

    document = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    # Include table cell text too — proposals often use tables.
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))
    return "\n".join(paragraphs)


def chunk_text(
    text: str,
    *,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """Split ``text`` into overlapping chunks, preferring paragraph boundaries.

    The splitter accumulates whole paragraphs until adding the next would exceed
    ``chunk_chars``, then starts a new chunk that carries the last ``overlap``
    characters of the previous one for context continuity. Oversized single
    paragraphs are hard-split.
    """
    if overlap >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk_chars")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # Hard-split paragraphs that alone exceed the chunk size.
        if len(para) > chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, chunk_chars, overlap))
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)
    return chunks


def _hard_split(text: str, chunk_chars: int, overlap: int) -> List[str]:
    step = chunk_chars - overlap
    return [text[i : i + chunk_chars] for i in range(0, len(text), step)]


def _normalize_whitespace(text: str) -> str:
    # Collapse 3+ newlines to a paragraph break; strip trailing spaces per line.
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


# Re-export for callers that want both in one place.
def extract_and_chunk(
    data: bytes,
    filename: str,
    *,
    chunk_chars: int = 1200,
    overlap: int = 200,
) -> "tuple[str, List[str]]":
    """Extract then chunk in one call. Returns ``(full_text, chunks)``."""
    text = extract_text_from_bytes(data, filename)
    return text, chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)


__all__ = [
    "UnsupportedDocument",
    "extract_text_from_bytes",
    "extract_text_from_path",
    "chunk_text",
    "extract_and_chunk",
]
