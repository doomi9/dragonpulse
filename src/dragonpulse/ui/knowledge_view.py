"""Knowledge base (RAG) view.

Upload past proposals / performance docs, index them into the local on-disk
vector store, and run grounded semantic search. Every result cites its source
document and chunk, and nothing leaves the machine.
"""

from __future__ import annotations

import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.knowledge import DEFAULT_CATEGORIES
from dragonpulse.processors.embeddings import describe_backend, get_embedding_backend
from dragonpulse.processors.text_extract import (
    ScannedPDFError,
    UnsupportedDocument,
    extract_text_from_bytes,
    format_upload_limit,
    ocr_available,
    ocr_pdf_bytes,
    validate_upload_size,
)
from dragonpulse.ui import state

logger = get_logger(__name__)

_SUPPORTED = ["pdf", "docx", "txt", "md"]
_ALL_CATEGORIES = DEFAULT_CATEGORIES


def render_knowledge() -> None:
    """Render the knowledge base tab."""
    st.subheader("📚 Knowledge base (RAG)")
    kb = state.get_knowledge_base()
    stats = kb.stats()

    st.caption(
        "Index your past proposals and performance docs locally, then retrieve "
        "the most relevant passages — each cited by source — to ground checklists, "
        "outreach, and (next) proposal drafts. Everything stays on this machine."
    )

    _render_backend_status(kb)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", stats.documents)
    c2.metric("Chunks", stats.chunks)
    c3.metric("Embedding", stats.backend)
    c4.metric("Vector dim", stats.dimension)

    _render_uploader(kb)
    st.divider()
    _render_search(kb)
    st.divider()
    _render_library(kb)


def _render_backend_status(kb) -> None:
    """Show the active embedding method and offer to switch/re-index."""
    status = describe_backend(kb.backend, kb.settings)

    if status.is_semantic:
        st.success(f"🧠 {status.headline}. {status.detail}")
    elif status.fell_back:
        st.warning(f"⚠️ {status.headline}. {status.detail}")
    else:
        st.info(f"🔤 {status.headline}. {status.detail}")

    with st.expander("How to enable better semantic search (Ollama)", expanded=False):
        st.markdown(
            "Semantic embeddings rank passages by **meaning**, not just keywords — "
            "much better at matching solicitation language to your past work.\n\n"
            "**One-time setup (fully local, no API keys):**\n"
            "1. Install [Ollama](https://ollama.com).\n"
            "2. Pull the embedding model:\n"
            "   ```bash\n"
            "   ollama pull nomic-embed-text\n"
            "   ```\n"
            "3. In your `.env`, point DragonPulse at the local server:\n"
            "   ```dotenv\n"
            "   DRAGONPULSE_LLM_BASE_URL=http://localhost:11434/v1\n"
            "   ```\n"
            "   (You do **not** need to enable the chat LLM — embeddings work on "
            "their own.)\n"
            "4. Restart the app, or click **Re-index** below.\n\n"
            "When the embedding method changes, your already-uploaded documents are "
            "**automatically re-indexed** from their stored text — no re-uploading."
        )

    if kb.stats().chunks > 0:
        cols = st.columns(2)
        if cols[0].button("🔄 Re-index (embeddings only)", width="stretch",
                          help="Detect the best available backend and rebuild all embeddings."):
            new_backend = get_embedding_backend(kb.settings)
            with st.spinner("Re-embedding every document locally…"):
                sig = kb.reindex(new_backend)
            new_status = describe_backend(kb.backend, kb.settings)
            if new_status.is_semantic:
                st.success(f"Re-indexed {kb.stats().chunks} chunks → {sig} (semantic).")
            else:
                st.info(f"Re-indexed {kb.stats().chunks} chunks → {sig} (lexical).")
            st.session_state.pop(state.KEY_KB_HITS, None)
            st.rerun()
        if cols[1].button(
            "♻️ Re-chunk + re-index (improved)", width="stretch",
            help="Rebuild chunks with smarter, larger semantic chunking and refreshed "
                 "metadata. Keeps your documents and categories.",
        ):
            with st.spinner("Re-chunking and re-embedding every document locally…"):
                info = kb.rechunk_all(regenerate_summary=True)
            st.success(
                f"Re-chunked {info['documents']} document(s) → {info['chunks']} "
                f"chunks using {info['backend']}."
            )
            st.session_state.pop(state.KEY_KB_HITS, None)
            st.rerun()


def _render_upload_result() -> None:
    """Show the outcome of the last index run (survives the post-index rerun)."""
    result = st.session_state.pop(state.KEY_KB_UPLOAD_RESULT, None)
    if not result:
        return
    indexed, skipped, failed = result["indexed"], result["skipped"], result["failed"]
    ocred = result.get("ocred", 0)
    if indexed:
        msg = f"Indexed {indexed} document(s) into '{result['category']}'."
        if ocred:
            msg += f" {ocred} were read via OCR (scanned PDFs)."
        if skipped:
            msg += f" Skipped {skipped} duplicate(s)."
        if failed:
            msg += f" {failed} failed (see below)."
        st.success(msg)
    elif skipped and not failed:
        st.info(f"Skipped {skipped} duplicate(s) — already in the knowledge base.")
    for err in result["errors"]:
        st.error(err)


def _render_uploader(kb) -> None:
    max_mb = kb.settings.kb_max_upload_mb
    limit_label = format_upload_limit(max_mb)
    st.markdown("**Add documents (bulk upload supported)**")
    _render_upload_result()
    ocr_on = kb.settings.kb_ocr_enabled and ocr_available()
    ocr_note = (
        " Scanned/image-only PDFs are read automatically with **OCR**."
        if ocr_on
        else " Scanned/image-only PDFs need a text layer (OCR) before upload."
    )
    st.caption(
        f"PDF, DOCX, TXT, and MD — up to **{limit_label}** per file. "
        "Large PDFs are supported; indexing may take longer for big documents."
        + ocr_note
    )
    files = st.file_uploader(
        "Upload past proposals / performance docs",
        type=_SUPPORTED,
        accept_multiple_files=True,
        help=f"PDF, DOCX, TXT, MD. Up to {limit_label} per file. Drag in several at once — "
        "they're parsed, chunked, and embedded locally.",
    )
    c1, c2 = st.columns(2)
    category = c1.selectbox(
        "Category (folder)",
        _ALL_CATEGORIES,
        index=0,
        help="Group documents so drafts can target the right kind of evidence.",
    )
    tags_raw = c2.text_input(
        "Optional tags (comma-separated)",
        placeholder="e.g. data-center, idiq, army",
    )
    label = f"📥 Index {len(files)} document(s)" if files else "📥 Index uploaded documents"
    if st.button(label, type="primary", width="stretch"):
        if not files:
            st.warning("Choose one or more files first.")
            return
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        ocr_on = kb.settings.kb_ocr_enabled and ocr_available()
        indexed, skipped, failed, ocred = 0, 0, 0, 0
        errors: list[str] = []
        progress = st.progress(0.0, text="Starting…")
        for idx, f in enumerate(files, start=1):
            base = (idx - 1) / len(files)
            progress.progress(base, text=f"Indexing {f.name}…")
            try:
                data = f.getvalue()
                validate_upload_size(len(data), max_mb, f.name)
                try:
                    text = extract_text_from_bytes(data, f.name)
                except ScannedPDFError:
                    if not ocr_on:
                        raise
                    def _on_page(done, total, _f=f, _base=base):
                        progress.progress(
                            _base + (done / total) / len(files),
                            text=f"OCR {_f.name}: page {done}/{total}…",
                        )

                    text = ocr_pdf_bytes(
                        data, dpi=kb.settings.kb_ocr_dpi, page_callback=_on_page
                    )
                    ocred += 1
                before = {d.doc_id for d in kb.list_documents()}
                doc = kb.add_document(
                    f.name, text, source_type="upload", category=category, tags=tags
                )
                if doc.doc_id in before:
                    skipped += 1
                else:
                    indexed += 1
            except UnsupportedDocument as exc:
                failed += 1
                errors.append(f"{f.name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                logger.exception("Indexing failed for %s", f.name)
                errors.append(f"{f.name}: unexpected error: {exc}")
        progress.progress(1.0, text="Done.")
        progress.empty()
        st.session_state[state.KEY_KB_UPLOAD_RESULT] = {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "ocred": ocred,
            "category": category,
            "errors": errors,
        }
        st.rerun()


def _render_search(kb) -> None:
    st.markdown("**Search the knowledge base**")
    query = st.text_input(
        "Query",
        placeholder="e.g. past performance on power line construction for the Army",
        key="kb_query",
    )
    col1, col2 = st.columns([1, 3])
    top_k = col1.slider("Results", 1, 15, value=kb.settings.rag_top_k)
    if col2.button("🔎 Search", width="stretch"):
        if not query.strip():
            st.warning("Enter a query.")
        elif kb.stats().chunks == 0:
            st.info("Knowledge base is empty — index some documents first.")
        else:
            st.session_state[state.KEY_KB_HITS] = kb.search(query, k=top_k)

    hits = st.session_state.get(state.KEY_KB_HITS)
    if not hits:
        return
    st.caption(f"Top {len(hits)} matches (cosine similarity):")
    for hit in hits:
        with st.container(border=True):
            head = st.columns([4, 1])
            head[0].markdown(f"**📄 {hit.citation}**")
            head[1].metric("Score", f"{hit.score:.3f}")
            preview = hit.chunk.text.strip()
            st.text(preview[:800] + ("…" if len(preview) > 800 else ""))


def _render_library(kb) -> None:
    docs = kb.list_documents()
    st.markdown(f"**Indexed library ({len(docs)})**")
    if not docs:
        st.info("No documents indexed yet.")
        return

    used = kb.categories()
    filter_options = ["All categories"] + used
    chosen = st.selectbox("Filter by category", filter_options, index=0)
    visible = docs if chosen == "All categories" else [d for d in docs if d.category == chosen]

    # Group by category for a folder-like view.
    by_cat: dict = {}
    for doc in visible:
        by_cat.setdefault(doc.category, []).append(doc)

    for category in sorted(by_cat):
        st.markdown(f"📁 **{category}** ({len(by_cat[category])})")
        for doc in by_cat[category]:
            _render_doc_row(kb, doc)

    st.divider()
    if st.button("Clear entire knowledge base", help="Deletes all indexed docs"):
        kb.clear()
        st.session_state.pop(state.KEY_KB_HITS, None)
        st.rerun()


def _render_doc_row(kb, doc) -> None:
    with st.container(border=True):
        cols = st.columns([5, 2, 1])
        tag_str = f" · _tags: {', '.join(doc.tags)}_" if doc.tags else ""
        cols[0].markdown(f"**{doc.name}**{tag_str}")
        type_str = f" · type: {doc.doc_type}" if getattr(doc, "doc_type", "") else ""
        cols[0].caption(
            f"{doc.chunk_count} chunks · {doc.char_count:,} chars{type_str} · "
            f"added {doc.added_at} · last indexed {doc.indexed_at} · {doc.source_type}"
        )
        if getattr(doc, "summary", ""):
            cols[0].caption(f"📝 {doc.summary}")
        # Inline category re-assignment.
        opts = _ALL_CATEGORIES + (
            [doc.category] if doc.category not in _ALL_CATEGORIES else []
        )
        new_cat = cols[1].selectbox(
            "Category",
            opts,
            index=opts.index(doc.category) if doc.category in opts else 0,
            key=f"cat_{doc.doc_id}",
            label_visibility="collapsed",
        )
        if new_cat != doc.category:
            kb.update_document(doc.doc_id, category=new_cat)
            st.rerun()
        if cols[2].button("🗑️", key=f"del_{doc.doc_id}", help="Remove from KB"):
            kb.delete_document(doc.doc_id)
            st.session_state.pop(state.KEY_KB_HITS, None)
            st.rerun()
