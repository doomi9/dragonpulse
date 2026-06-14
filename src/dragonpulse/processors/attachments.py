"""Attachment download and text extraction for opportunity resource links.

SAM.gov resource-link URLs (``.../opportunities/resources/files/<uuid>/download``)
require the API key as a query parameter. Downloads are cached on disk under
``data/attachments/`` keyed by URL hash so we never re-download the same file.

Text extraction supports PDFs (via ``pdfplumber``) and plain-text/CSV files.
Other binary types are downloaded but not previewed.

Security note: extracted proposal/SOW text can be sensitive. We never log the
extracted text itself — only file names, sizes, and page counts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings

logger = get_logger(__name__)

DOWNLOAD_TIMEOUT = 60
_MAX_PREVIEW_CHARS = 20_000


@dataclass
class AttachmentExtract:
    """Result of downloading + extracting an attachment."""

    url: str
    local_path: Path
    filename: str
    content_type: Optional[str]
    size_bytes: int
    text: Optional[str]  # None if not extractable
    page_count: Optional[int] = None
    truncated: bool = False
    error: Optional[str] = None

    @property
    def has_text(self) -> bool:
        return bool(self.text)


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _filename_from_headers(resp: requests.Response, fallback: str) -> str:
    """Extract a filename from Content-Disposition, falling back to a hash."""
    cd = resp.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?="?([^";]+)"?', cd)
    if match:
        name = match.group(1).strip()
        # Strip any RFC 5987 encoding prefix like "UTF-8''".
        if "''" in name:
            name = name.split("''", 1)[1]
        return name
    return fallback


def download_attachment(
    url: str,
    *,
    settings: Optional[Settings] = None,
    force: bool = False,
) -> AttachmentExtract:
    """Download a resource link to the local attachments dir (cached).

    Returns an :class:`AttachmentExtract` with ``text=None``; call
    :func:`extract_text` to populate text, or use :func:`download_and_extract`.
    """
    settings = settings or get_settings()
    settings.attachments_dir.mkdir(parents=True, exist_ok=True)

    h = _url_hash(url)
    # Look for any previously downloaded file with this hash prefix.
    if not force:
        existing = list(settings.attachments_dir.glob(f"{h}__*"))
        if existing:
            path = existing[0]
            return AttachmentExtract(
                url=url,
                local_path=path,
                filename=path.name.split("__", 1)[1],
                content_type=None,
                size_bytes=path.stat().st_size,
                text=None,
            )

    params = {}
    if settings.has_api_key:
        params["api_key"] = settings.active_api_key

    try:
        resp = requests.get(url, params=params, timeout=DOWNLOAD_TIMEOUT, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Attachment download failed for %s: %s", _url_hash(url), exc)
        return AttachmentExtract(
            url=url,
            local_path=Path(),
            filename="(download failed)",
            content_type=None,
            size_bytes=0,
            text=None,
            error=str(exc),
        )

    content_type = resp.headers.get("Content-Type")
    filename = _filename_from_headers(resp, fallback=f"attachment_{h}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    path = settings.attachments_dir / f"{h}__{safe_name}"

    size = 0
    with open(path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
                size += len(chunk)

    logger.info("Downloaded attachment %s (%s, %d bytes)", safe_name, content_type, size)
    return AttachmentExtract(
        url=url,
        local_path=path,
        filename=filename,
        content_type=content_type,
        size_bytes=size,
        text=None,
    )


def extract_text(attachment: AttachmentExtract) -> AttachmentExtract:
    """Populate ``attachment.text`` (and ``page_count``) where possible.

    Mutates and returns the passed-in object for convenience.
    """
    path = attachment.local_path
    if not path or not path.exists():
        attachment.error = attachment.error or "file missing"
        return attachment

    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf" or (attachment.content_type or "").endswith("pdf"):
            attachment.text, attachment.page_count = _extract_pdf(path)
        elif suffix in {".txt", ".csv", ".md", ".json", ".xml", ".htm", ".html"}:
            attachment.text = path.read_text(encoding="utf-8", errors="replace")
        else:
            attachment.text = None  # unsupported for preview
    except Exception as exc:  # noqa: BLE001 - extraction must not crash the UI
        logger.warning("Text extraction failed for %s: %s", path.name, exc)
        attachment.error = f"extraction failed: {exc}"
        return attachment

    if attachment.text and len(attachment.text) > _MAX_PREVIEW_CHARS:
        attachment.text = attachment.text[:_MAX_PREVIEW_CHARS]
        attachment.truncated = True
    return attachment


def _extract_pdf(path: Path) -> "tuple[str, int]":
    """Extract text from a PDF using pdfplumber. Returns (text, page_count)."""
    import pdfplumber

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
            if sum(len(p) for p in parts) > _MAX_PREVIEW_CHARS:
                break
    return "\n\n".join(parts).strip(), page_count


def download_and_extract(
    url: str, *, settings: Optional[Settings] = None, force: bool = False
) -> AttachmentExtract:
    """Convenience: download then extract text in one call."""
    att = download_attachment(url, settings=settings, force=force)
    if att.error:
        return att
    return extract_text(att)
