"""Placeholder views for future DragonPulse modules.

These render informative "coming soon" panels so the navigation and overall
structure are in place. Each notes what it will do and which building blocks
already exist in the codebase.
"""

from __future__ import annotations

import streamlit as st


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
        "- The RAG knowledge base (`processors/knowledge_base.py`) for "
        "past-performance grounding."
    )
