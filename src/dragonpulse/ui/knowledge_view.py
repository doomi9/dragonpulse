"""Knowledge base (RAG) view.

Upload past proposals / performance docs, index them into the local on-disk
vector store, and run grounded semantic search. Every result cites its source
document and chunk, and nothing leaves the machine.
"""

from __future__ import annotations

import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.processors.text_extract import (
    UnsupportedDocument,
    extract_text_from_bytes,
)
from dragonpulse.ui import state

logger = get_logger(__name__)

_SUPPORTED = ["pdf", "docx", "txt", "md"]


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


def _render_uploader(kb) -> None:
    st.markdown("**Add documents**")
    files = st.file_uploader(
        "Upload past proposals / performance docs",
        type=_SUPPORTED,
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, MD. Files are parsed, chunked, and embedded locally.",
    )
    tags_raw = st.text_input(
        "Optional tags (comma-separated)",
        placeholder="e.g. engineering, past-performance, idiq",
    )
    if st.button("📥 Index uploaded documents", type="primary", width="stretch"):
        if not files:
            st.warning("Choose one or more files first.")
            return
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        indexed, skipped, failed = 0, 0, 0
        with st.spinner("Extracting, chunking, and embedding locally…"):
            for f in files:
                try:
                    text = extract_text_from_bytes(f.getvalue(), f.name)
                    before = {d.doc_id for d in kb.list_documents()}
                    doc = kb.add_document(f.name, text, source_type="upload", tags=tags)
                    if doc.doc_id in before:
                        skipped += 1
                    else:
                        indexed += 1
                except UnsupportedDocument as exc:
                    failed += 1
                    st.error(f"{f.name}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.exception("Indexing failed for %s", f.name)
                    st.error(f"{f.name}: unexpected error: {exc}")
        msg = f"Indexed {indexed} document(s)."
        if skipped:
            msg += f" Skipped {skipped} duplicate(s)."
        if failed:
            msg += f" {failed} failed."
        st.success(msg)
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

    for doc in docs:
        with st.container(border=True):
            cols = st.columns([5, 2, 1])
            tag_str = f" · _tags: {', '.join(doc.tags)}_" if doc.tags else ""
            cols[0].markdown(f"**{doc.name}**{tag_str}")
            cols[0].caption(
                f"{doc.chunk_count} chunks · {doc.char_count:,} chars · "
                f"added {doc.added_at} · {doc.source_type}"
            )
            if cols[2].button("🗑️", key=f"del_{doc.doc_id}", help="Remove from KB"):
                kb.delete_document(doc.doc_id)
                st.session_state.pop(state.KEY_KB_HITS, None)
                st.rerun()

    if st.button("Clear entire knowledge base", help="Deletes all indexed docs"):
        kb.clear()
        st.session_state.pop(state.KEY_KB_HITS, None)
        st.rerun()
