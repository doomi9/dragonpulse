"""Local, on-disk RAG knowledge base.

A deliberately simple and auditable vector store:

- ``vectors.npy``   — float32 matrix of L2-normalized chunk embeddings.
- ``chunks.json``   — parallel list of chunk metadata + text (same row order).
- ``documents.json``— per-document metadata.
- ``index.json``    — store-level metadata, incl. the embedding signature.

Because the matrix is L2-normalized, cosine similarity is a single dot product,
which is fast and exact for the scale of a contractor's proposal library
(hundreds–thousands of chunks). No cloud, no external vector DB.

If the configured embedding backend changes (different signature), the store
transparently **reindexes** from the stored chunk texts on next load.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.knowledge import Chunk, Document, KBStats, RetrievedChunk
from dragonpulse.processors.embeddings import EmbeddingBackend, get_embedding_backend
from dragonpulse.processors.text_extract import semantic_chunks

logger = get_logger(__name__)

# After one failed LLM summary in a session we stop retrying (keeps ingestion fast
# when the configured model is missing/unavailable) and fall back to a heuristic.
_LLM_SUMMARY_DISABLED = False


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def infer_doc_type(name: str, category: str, text: str) -> str:
    """Best-effort classification of a document's kind from its name/category/text."""
    hay = f"{name} {category}".lower()
    head = (text or "")[:1500].lower()

    def has(*words: str) -> bool:
        return any(w in hay or w in head for w in words)

    if has("resume", "curriculum vitae", " cv "):
        return "Resume"
    if has("statement of work", "performance work statement", "pws", "sow"):
        return "Statement of Work"
    if has("past performance", "cpars", "reference contract"):
        return "Past Performance"
    if has("capability statement", "capabilities statement", "core competencies"):
        return "Capability Statement"
    if has("price", "pricing", "cost proposal", "rate", "labor category"):
        return "Pricing"
    if has("certification", "iso 9001", "cage code", "sam registration"):
        return "Certification"
    if has("proposal", "technical volume", "section l", "section m"):
        return "Proposal"
    # Fall back to the user-assigned category where it maps cleanly.
    mapping = {
        "Past Performance": "Past Performance",
        "Capabilities": "Capability Statement",
        "Technical": "Technical Document",
        "Pricing": "Pricing",
        "Certifications": "Certification",
    }
    return mapping.get(category, "Other")


def _heuristic_summary(text: str, *, max_chars: int = 280) -> str:
    """A cheap summary: the first substantive sentence(s), capped."""
    from dragonpulse.processors.text_extract import _split_sentences

    for para in (text or "").split("\n\n"):
        para = " ".join(para.split())
        if len(para) < 40:
            continue
        sentences = _split_sentences(para) or [para]
        out = ""
        for sentence in sentences:
            candidate = f"{out} {sentence}".strip() if out else sentence
            if len(candidate) > max_chars and out:
                break
            out = candidate
            if len(out) >= max_chars:
                break
        return out[:max_chars].strip()
    return " ".join((text or "").split())[:max_chars].strip()


def generate_doc_summary(text: str, settings: Settings) -> str:
    """1-2 sentence document gist — LLM when available, else a heuristic."""
    global _LLM_SUMMARY_DISABLED
    if not getattr(settings, "kb_summarize", True):
        return ""
    heuristic = _heuristic_summary(text)
    if _LLM_SUMMARY_DISABLED or not settings.llm_active:
        return heuristic
    try:
        from dragonpulse.processors.llm import LLMClient

        client = LLMClient(settings)
        if not client.available:
            return heuristic
        result = client.complete(
            instruction=(
                "In 1-2 sentences, summarize what this company document covers: its "
                "domain, the capabilities/services it describes, and the type of work. "
                "Do not add facts that are not present."
            ),
            context=text[:6000],
            sources=["knowledge base document"],
            max_tokens=120,
            system_prompt=(
                "You write concise, factual one-line summaries of business documents. "
                "Use only the provided text."
            ),
        )
        summary = (result.text or "").strip()
        return summary or heuristic
    except Exception as exc:  # noqa: BLE001 - any failure -> heuristic, disable retries
        _LLM_SUMMARY_DISABLED = True
        logger.info("LLM summary unavailable (%s); using heuristic summaries.", exc)
        return heuristic


def _reconstruct_text(chunks: List[Chunk]) -> str:
    """Best-effort rebuild of a document's text from its (overlapping) chunks.

    Used only for legacy documents indexed before full text was persisted. New
    documents store their full text on disk, so this lossy path is not needed.
    """
    ordered = sorted(chunks, key=lambda c: c.ordinal)
    if not ordered:
        return ""
    out = ordered[0].text
    for chunk in ordered[1:]:
        nxt = chunk.text
        max_ov = min(len(out), len(nxt), 1500)
        overlap = 0
        for k in range(max_ov, 24, -1):
            if out[-k:] == nxt[:k]:
                overlap = k
                break
        out += "\n\n" + nxt[overlap:]
    return out


class KnowledgeBase:
    """A persistent, local vector store over document chunks."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.backend = backend or get_embedding_backend(self.settings)
        self.dir = self.settings.rag_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        self._documents: List[Document] = []
        self._chunks: List[Chunk] = []
        self._vectors: np.ndarray = np.zeros((0, self.backend.dimension), dtype=np.float32)

        self._load()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    @property
    def _vectors_path(self) -> Path:
        return self.dir / "vectors.npy"

    @property
    def _chunks_path(self) -> Path:
        return self.dir / "chunks.json"

    @property
    def _documents_path(self) -> Path:
        return self.dir / "documents.json"

    @property
    def _index_path(self) -> Path:
        return self.dir / "index.json"

    @property
    def _texts_dir(self) -> Path:
        return self.dir / "texts"

    def _text_path(self, doc_id: str) -> Path:
        return self._texts_dir / f"{doc_id}.txt"

    def _write_text(self, doc_id: str, text: str) -> None:
        try:
            self._texts_dir.mkdir(parents=True, exist_ok=True)
            self._text_path(doc_id).write_text(text, "utf-8")
        except OSError:  # best-effort; reconstruction remains as a fallback
            logger.debug("Could not persist full text for %s", doc_id, exc_info=True)

    def _document_text(self, doc: Document) -> str:
        """Full text for a document — from disk, or reconstructed from chunks."""
        path = self._text_path(doc.doc_id)
        if path.exists():
            try:
                return path.read_text("utf-8")
            except OSError:
                pass
        return _reconstruct_text([c for c in self._chunks if c.doc_id == doc.doc_id])

    def _chunk_embed_text(self, chunk: Chunk, doc: Optional[Document]) -> str:
        """Context-enriched text to embed (the cited ``chunk.text`` stays clean).

        Prefixing each chunk with its document name, category/type, summary, and
        section heading meaningfully improves retrieval relevance — a fragment
        about "exciter testing" embeds closer to the document it belongs to.
        """
        header_bits: List[str] = []
        summary = ""
        if doc is not None:
            header_bits.append(doc.name)
            if doc.category and doc.category != "Uncategorized":
                header_bits.append(doc.category)
            if doc.doc_type and doc.doc_type != "Other":
                header_bits.append(doc.doc_type)
            summary = doc.summary or ""
        section = chunk.section or ""
        if section:
            header_bits.append(section)
        parts: List[str] = []
        if header_bits:
            parts.append(" — ".join(header_bits))
        if summary:
            parts.append(summary)
        parts.append(chunk.text)
        return "\n".join(parts)

    def _doc_map(self) -> "dict[str, Document]":
        return {d.doc_id: d for d in self._documents}

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self._chunks_path.exists():
            return
        try:
            self._documents = [
                Document.model_validate(d)
                for d in json.loads(self._documents_path.read_text("utf-8"))
            ]
            self._chunks = [
                Chunk.model_validate(c)
                for c in json.loads(self._chunks_path.read_text("utf-8"))
            ]
            index_meta = json.loads(self._index_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Knowledge base index unreadable (%s); starting empty.", exc)
            self._documents, self._chunks = [], []
            return

        stored_sig = index_meta.get("signature")
        if stored_sig != self.backend.signature():
            logger.info(
                "Embedding backend changed (%s -> %s); reindexing knowledge base.",
                stored_sig,
                self.backend.signature(),
            )
            self._reindex_from_chunks()
            return

        if self._vectors_path.exists():
            self._vectors = np.load(self._vectors_path)
        else:
            self._reindex_from_chunks()

    def _save(self) -> None:
        np.save(self._vectors_path, self._vectors)
        self._chunks_path.write_text(
            json.dumps([c.model_dump() for c in self._chunks], indent=2), "utf-8"
        )
        self._documents_path.write_text(
            json.dumps([d.model_dump() for d in self._documents], indent=2), "utf-8"
        )
        self._index_path.write_text(
            json.dumps(
                {
                    "signature": self.backend.signature(),
                    "dimension": self.backend.dimension,
                    "documents": len(self._documents),
                    "chunks": len(self._chunks),
                },
                indent=2,
            ),
            "utf-8",
        )

    def _reindex_from_chunks(self) -> None:
        """Recompute all vectors from stored chunk texts (e.g. backend change)."""
        if not self._chunks:
            self._vectors = np.zeros((0, self.backend.dimension), dtype=np.float32)
        else:
            doc_map = self._doc_map()
            self._vectors = self.backend.embed(
                [self._chunk_embed_text(c, doc_map.get(c.doc_id)) for c in self._chunks]
            )
        now = _now_iso()
        for doc in self._documents:
            doc.indexed_at = now
        self._save()

    def reindex(self, backend: Optional[EmbeddingBackend] = None) -> str:
        """Re-embed every stored chunk, optionally switching the embedding backend.

        This is how DragonPulse upgrades an existing library from lexical to
        semantic embeddings (or vice-versa) without re-uploading documents: the
        original chunk *text* is retained, so only the vectors are rebuilt.

        Parameters
        ----------
        backend:
            New backend to adopt. If ``None``, the current backend is re-run
            (useful to refresh after a previously-unavailable server comes up).

        Returns
        -------
        str
            The signature of the backend now in use (e.g. ``"ollama:nomic-embed-text"``).
        """
        with self._lock:
            if backend is not None:
                self.backend = backend
            self._reindex_from_chunks()
        logger.info(
            "Reindexed %d chunks across %d documents using %s.",
            len(self._chunks),
            len(self._documents),
            self.backend.signature(),
        )
        return self.backend.signature()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_document(
        self,
        name: str,
        text: str,
        *,
        source_type: str = "upload",
        category: str = "Uncategorized",
        tags: Optional[List[str]] = None,
        skip_if_duplicate: bool = True,
    ) -> Document:
        """Chunk, embed, and persist a document. Returns its :class:`Document`.

        If ``skip_if_duplicate`` and a document with identical content already
        exists, the existing record is returned without re-indexing.
        """
        sha = Document.sha_of(text)
        if skip_if_duplicate:
            for existing in self._documents:
                if existing.content_sha == sha:
                    logger.info("Skipping duplicate document '%s'.", name)
                    return existing

        pieces = semantic_chunks(
            text,
            chunk_chars=self.settings.rag_chunk_chars,
            overlap=self.settings.rag_chunk_overlap,
        )
        if not pieces:
            raise ValueError(f"Document '{name}' produced no chunks.")

        category = category or "Uncategorized"
        doc_type = infer_doc_type(name, category, text)
        summary = generate_doc_summary(text, self.settings)
        doc = Document(
            name=name,
            source_type=source_type,
            category=category,
            doc_type=doc_type,
            summary=summary,
            content_sha=sha,
            char_count=len(text),
            chunk_count=len(pieces),
            tags=tags or [],
        )
        new_chunks = [
            Chunk(
                doc_id=doc.doc_id,
                doc_name=name,
                ordinal=i,
                text=piece.text,
                category=category,
                doc_type=doc_type,
                section=piece.section,
            )
            for i, piece in enumerate(pieces)
        ]
        new_vectors = self.backend.embed(
            [self._chunk_embed_text(c, doc) for c in new_chunks]
        )

        with self._lock:
            self._documents.append(doc)
            self._chunks.extend(new_chunks)
            self._vectors = (
                new_vectors
                if self._vectors.size == 0
                else np.vstack([self._vectors, new_vectors])
            )
            self._write_text(doc.doc_id, text)
            self._save()
        logger.info(
            "Indexed '%s': %d chunks (%d chars, type=%s).",
            name, len(pieces), len(text), doc_type,
        )
        return doc

    def rechunk_all(self, *, regenerate_summary: bool = False) -> dict:
        """Re-chunk and re-embed every document with the current ingestion logic.

        Original documents, categories, tags, and ids are preserved; only the
        chunking, metadata (doc type / summary / section), and vectors are
        rebuilt. Returns a small stats dict for the UI.
        """
        with self._lock:
            new_chunks: List[Chunk] = []
            rebuilt = 0
            for doc in self._documents:
                text = self._document_text(doc)
                if not text.strip():
                    logger.warning("No recoverable text for '%s'; skipping.", doc.name)
                    continue
                self._write_text(doc.doc_id, text)
                doc.doc_type = infer_doc_type(doc.name, doc.category, text)
                if regenerate_summary or not doc.summary:
                    doc.summary = generate_doc_summary(text, self.settings)
                pieces = semantic_chunks(
                    text,
                    chunk_chars=self.settings.rag_chunk_chars,
                    overlap=self.settings.rag_chunk_overlap,
                )
                doc.char_count = len(text)
                doc.chunk_count = len(pieces)
                doc.indexed_at = _now_iso()
                for i, piece in enumerate(pieces):
                    new_chunks.append(
                        Chunk(
                            doc_id=doc.doc_id,
                            doc_name=doc.name,
                            ordinal=i,
                            text=piece.text,
                            category=doc.category,
                            doc_type=doc.doc_type,
                            section=piece.section,
                        )
                    )
                rebuilt += 1
            self._chunks = new_chunks
            doc_map = self._doc_map()
            if new_chunks:
                self._vectors = self.backend.embed(
                    [self._chunk_embed_text(c, doc_map.get(c.doc_id)) for c in new_chunks]
                )
            else:
                self._vectors = np.zeros((0, self.backend.dimension), dtype=np.float32)
            self._save()
        logger.info(
            "Re-chunked %d document(s) into %d chunks using %s.",
            rebuilt, len(new_chunks), self.backend.signature(),
        )
        return {
            "documents": rebuilt,
            "chunks": len(new_chunks),
            "backend": self.backend.signature(),
        }

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        *,
        categories: Optional[List[str]] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Return the top-``k`` chunks most similar to ``query`` (cited).

        When ``categories`` is given, only chunks from documents in those
        categories are considered — used to pull style exemplars from the most
        relevant kinds of documents (e.g. Technical, Past Performance). When
        ``min_score`` is given, chunks below that cosine similarity are dropped
        so low-relevance fragments don't surface.
        """
        k = k or self.settings.rag_top_k
        if not query.strip() or self._vectors.shape[0] == 0:
            return []
        if categories:
            wanted = set(categories)
            allowed = {d.doc_id for d in self._documents if d.category in wanted}
            idxs = [i for i, c in enumerate(self._chunks) if c.doc_id in allowed]
            if not idxs:
                return []
        else:
            idxs = list(range(len(self._chunks)))
        q = self.backend.embed([query])[0]  # already L2-normalized
        sub = self._vectors[idxs]
        # errstate guards against spurious FP warnings from macOS Accelerate BLAS.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = sub @ q  # cosine similarity (vectors are normalized)
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        order = np.argsort(scores)[::-1][:k]
        hits = [
            RetrievedChunk(chunk=self._chunks[idxs[j]], score=float(scores[j]))
            for j in order
        ]
        if min_score is not None:
            hits = [h for h in hits if h.score >= min_score]
        return hits

    def relevance(
        self,
        text: str,
        *,
        categories: Optional[List[str]] = None,
        top_n: int = 3,
    ) -> tuple:
        """Score ``text`` against the (optionally category-filtered) library.

        Returns ``(score, evidence)`` where ``score`` is the mean cosine
        similarity of the ``top_n`` closest chunks and ``evidence`` is that list
        of :class:`RetrievedChunk`. Used to rank external opportunities by how
        well they match the user's own documents.
        """
        if not text.strip() or self._vectors.shape[0] == 0:
            return 0.0, []
        if categories:
            wanted = set(categories)
            allowed = {d.doc_id for d in self._documents if d.category in wanted}
            idxs = [i for i, c in enumerate(self._chunks) if c.doc_id in allowed]
        else:
            idxs = list(range(len(self._chunks)))
        if not idxs:
            return 0.0, []
        q = self.backend.embed([text])[0]
        sub = self._vectors[idxs]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = sub @ q
        scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
        order = np.argsort(scores)[::-1][: max(1, top_n)]
        evidence = [
            RetrievedChunk(chunk=self._chunks[idxs[j]], score=float(scores[j]))
            for j in order
        ]
        mean_score = float(np.mean([e.score for e in evidence])) if evidence else 0.0
        return mean_score, evidence

    def category_texts(self, categories: Optional[List[str]] = None) -> List[str]:
        """Chunk texts for documents in ``categories`` (all if ``None``)."""
        if categories is None:
            return [c.text for c in self._chunks]
        wanted = set(categories)
        allowed = {d.doc_id for d in self._documents if d.category in wanted}
        return [c.text for c in self._chunks if c.doc_id in allowed]

    def category_document_texts(
        self, categories: Optional[List[str]] = None
    ) -> List[str]:
        """One combined text **per document** in ``categories`` (all if ``None``).

        Per-document (not per-chunk) granularity lets query generation reason
        about *document frequency* — terms shared across documents describe the
        firm's general capabilities, while one-off terms are project specifics.
        """
        return [self._document_text(d) for d in self.documents_in_categories(categories)]

    def documents_in_categories(
        self, categories: Optional[List[str]] = None
    ) -> List[Document]:
        """Documents whose category is in ``categories`` (all if ``None``)."""
        if categories is None:
            return list(self._documents)
        wanted = set(categories)
        return [d for d in self._documents if d.category in wanted]

    def update_document(
        self,
        doc_id: str,
        *,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Update a document's organizational metadata (category/tags)."""
        with self._lock:
            doc = next((d for d in self._documents if d.doc_id == doc_id), None)
            if doc is None:
                return False
            category_changed = False
            if category is not None:
                new_cat = category or "Uncategorized"
                category_changed = new_cat != doc.category
                doc.category = new_cat
            if tags is not None:
                doc.tags = tags
            # Category feeds the contextual embedding, so re-embed this doc's
            # chunks when it changes to keep retrieval relevance accurate.
            if category_changed:
                doc.doc_type = infer_doc_type(doc.name, doc.category, self._document_text(doc))
                rows = [i for i, c in enumerate(self._chunks) if c.doc_id == doc_id]
                for i in rows:
                    self._chunks[i].category = doc.category
                    self._chunks[i].doc_type = doc.doc_type
                if rows and self._vectors.size:
                    revec = self.backend.embed(
                        [self._chunk_embed_text(self._chunks[i], doc) for i in rows]
                    )
                    for j, i in enumerate(rows):
                        self._vectors[i] = revec[j]
            self._save()
        return True

    def categories(self) -> List[str]:
        """Sorted, distinct categories currently in use."""
        return sorted({d.category for d in self._documents})

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks/vectors. Returns True if found."""
        with self._lock:
            if not any(d.doc_id == doc_id for d in self._documents):
                return False
            keep_mask = np.array(
                [c.doc_id != doc_id for c in self._chunks], dtype=bool
            )
            self._chunks = [c for c, keep in zip(self._chunks, keep_mask) if keep]
            self._documents = [d for d in self._documents if d.doc_id != doc_id]
            self._vectors = (
                self._vectors[keep_mask]
                if self._vectors.size
                else self._vectors
            )
            self._text_path(doc_id).unlink(missing_ok=True)
            self._save()
        logger.info("Deleted document %s from knowledge base.", doc_id)
        return True

    def clear(self) -> None:
        """Delete the entire knowledge base."""
        with self._lock:
            for doc in self._documents:
                self._text_path(doc.doc_id).unlink(missing_ok=True)
            self._documents, self._chunks = [], []
            self._vectors = np.zeros((0, self.backend.dimension), dtype=np.float32)
            self._save()

    def list_documents(self) -> List[Document]:
        return list(self._documents)

    def stats(self) -> KBStats:
        return KBStats(
            documents=len(self._documents),
            chunks=len(self._chunks),
            backend=self.backend.signature(),
            dimension=self.backend.dimension,
            total_chars=sum(d.char_count for d in self._documents),
        )
