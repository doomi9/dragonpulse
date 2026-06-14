"""Placeholder views for future DragonPulse modules.

These render informative "coming soon" panels so the navigation and overall
structure are in place. Each notes what it will do and which building blocks
already exist in the codebase.
"""

from __future__ import annotations

import streamlit as st


def render_rag() -> None:
    st.subheader("📚 Knowledge base (RAG)")
    st.info(
        "**Planned module.** A local RAG index of your past proposals and "
        "performance to ground future drafts.",
        icon="🛠️",
    )
    st.file_uploader(
        "Upload past proposals / performance docs (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Stub: files are not yet indexed. Local embedding + retrieval is next.",
    )
    st.caption(
        "Implementation plan: chunk → local embeddings (e.g. sentence-transformers) "
        "→ local vector store (FAISS/Chroma on disk) → grounded retrieval with citations."
    )


def render_proposal() -> None:
    st.subheader("📝 Proposal generator")
    st.info(
        "**Planned module.** Generate compliant proposal section drafts grounded "
        "in the solicitation (Sections L/M, SOW) and your knowledge base.",
        icon="🛠️",
    )
    st.markdown(
        "**Will reuse:**\n"
        "- Attachment text extraction (`processors/attachments.py`) for the SOW/PWS.\n"
        "- The grounded LLM wrapper (`processors/llm.py`) with source citations.\n"
        "- The RAG knowledge base (above) for past-performance grounding."
    )
