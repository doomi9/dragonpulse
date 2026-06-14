"""Proposal Generator view.

Combines an opportunity's solicitation (attachment text) with the company's RAG
knowledge base to generate grounded, cited proposal sections. Supports
per-section regeneration with chat-style feedback and export to Markdown / DOCX.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.processors.attachments import download_and_extract
from dragonpulse.processors.proposal import (
    SECTION_SPECS,
    ProposalGenerator,
    draft_to_docx_bytes,
)
from dragonpulse.ui import state

logger = get_logger(__name__)

_SPEC_BY_ID = {s.section_id: s for s in SECTION_SPECS}


def render_proposal() -> None:
    """Render the Proposal Generator tab."""
    st.subheader("📝 Proposal Generator")
    st.caption(
        "Generate grounded, source-cited proposal drafts from a solicitation + your "
        "knowledge base. Every section cites the solicitation excerpts and company "
        "documents it was built from. Nothing is invented."
    )

    kb = state.get_knowledge_base()
    _render_llm_status(kb)

    opp = _select_opportunity()
    if opp is None:
        return

    st.divider()
    gen = _ensure_generator(opp, kb)
    _render_solicitation_loader(opp, gen)
    st.divider()
    _render_generate_controls(gen)
    _render_draft()


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #
def _render_llm_status(kb) -> None:
    settings = kb.settings
    from dragonpulse.processors.llm import LLMClient

    llm = LLMClient(settings)
    if llm.available:
        where = "local" if settings.llm_is_local else "cloud"
        st.success(
            f"🧠 Full AI drafting enabled ({where} model: {settings.llm_model}). "
            "Sections are written as grounded prose with citations."
        )
    else:
        st.info(
            "🔤 No drafting LLM enabled — sections render as a **grounded evidence "
            "scaffold** (real solicitation + KB excerpts, nothing invented). For full "
            "prose, enable a local model: set `DRAGONPULSE_LLM_ENABLED=true` and "
            "`DRAGONPULSE_LLM_MODEL=qwen2.5:14b` (or `llama3.2:3b`) in `.env`."
        )
    if kb.stats().documents == 0:
        st.warning(
            "Your knowledge base is empty. Add past proposals/performance in the "
            "**Knowledge Base** tab so drafts can be grounded in your real work.",
            icon="📚",
        )


# --------------------------------------------------------------------------- #
# Opportunity selection
# --------------------------------------------------------------------------- #
def _select_opportunity() -> Optional[Opportunity]:
    st.markdown("**1. Choose an opportunity**")
    opportunities = state.get_opportunities()
    selected = state.get_selected()

    if not opportunities:
        st.info(
            "No opportunities loaded. Run a search in the **Discover** tab first, "
            "then return here (or open one in the **Detail** tab)."
        )
        return None

    options = {f"{o.title or '(untitled)'} · {o.notice_id}": o.notice_id for o in opportunities}
    labels = list(options.keys())
    default_idx = 0
    if selected is not None:
        for i, o in enumerate(opportunities):
            if o.notice_id == selected.notice_id:
                default_idx = i
                break
    chosen_label = st.selectbox("Opportunity", labels, index=default_idx)
    notice_id = options[chosen_label]
    opp = next((o for o in opportunities if o.notice_id == notice_id), None)
    if opp:
        st.caption(
            f"**{opp.title}** · {opp.agency or '—'} · NAICS {opp.naics_code or '—'} · "
            f"{len(opp.resource_links)} attachment(s)"
        )
    return opp


# --------------------------------------------------------------------------- #
# Generator lifecycle
# --------------------------------------------------------------------------- #
def _ensure_generator(opp: Opportunity, kb) -> ProposalGenerator:
    gen = st.session_state.get(state.KEY_PROPOSAL_GEN)
    if gen is None or gen.opp.notice_id != opp.notice_id:
        gen = ProposalGenerator(opp, settings=kb.settings, knowledge_base=kb)
        st.session_state[state.KEY_PROPOSAL_GEN] = gen
        st.session_state[state.KEY_PROPOSAL_ATTACH] = []
        st.session_state.pop(state.KEY_PROPOSAL_DRAFT, None)
    return gen


def _render_solicitation_loader(opp: Opportunity, gen: ProposalGenerator) -> None:
    st.markdown("**2. Load the solicitation**")
    loaded = st.session_state.get(state.KEY_PROPOSAL_ATTACH, [])

    cols = st.columns([1, 2])
    if cols[0].button(
        "📎 Load & extract attachments",
        width="stretch",
        disabled=not opp.resource_links,
        help="Download the opportunity's attachments and index their text.",
    ):
        _load_attachments(opp, gen)

    pasted = st.text_area(
        "…or paste solicitation / SOW text directly",
        height=120,
        placeholder="Paste the statement of work or key requirements here if "
        "attachments are scanned images or unavailable.",
    )
    if cols[1].button("➕ Add pasted text to solicitation", width="stretch"):
        if pasted.strip():
            n = gen.load_solicitation([("Pasted text", pasted)])
            loaded = st.session_state.get(state.KEY_PROPOSAL_ATTACH, [])
            loaded.append(f"Pasted text ({n} chunks)")
            st.session_state[state.KEY_PROPOSAL_ATTACH] = loaded
            st.success(f"Indexed pasted text into {n} chunks.")
        else:
            st.warning("Nothing to add — paste some text first.")

    if loaded:
        st.caption("Solicitation context loaded from: " + "; ".join(loaded))
    else:
        st.caption(
            "No solicitation context yet. Load attachments or paste text. "
            "(You can still generate from KB + metadata, but grounding will be weaker.)"
        )


def _load_attachments(opp: Opportunity, gen: ProposalGenerator) -> None:
    attachments: List[Tuple[str, str]] = []
    names: List[str] = []
    with st.spinner("Downloading and extracting attachments…"):
        for link in opp.resource_links:
            att = download_and_extract(link.url, settings=gen.settings)
            if att.error:
                st.warning(f"Skipped an attachment: {att.error}")
                continue
            if att.text:
                attachments.append((att.filename, att.text))
                names.append(att.filename)
            else:
                st.caption(f"No extractable text: {att.filename}")
    if not attachments:
        st.error("No text could be extracted from the attachments. Try pasting text instead.")
        return
    n = gen.load_solicitation(attachments)
    st.session_state[state.KEY_PROPOSAL_ATTACH] = [f"{nm}" for nm in names] + [f"({n} chunks)"]
    st.success(f"Indexed {len(attachments)} file(s) into {n} solicitation chunks.")


# --------------------------------------------------------------------------- #
# Generation + rendering
# --------------------------------------------------------------------------- #
def _render_generate_controls(gen: ProposalGenerator) -> None:
    st.markdown("**3. Generate the draft**")
    c1, c2 = st.columns([1, 2])
    include_optional = c2.checkbox("Include high-level pricing strategy notes", value=False)
    if c1.button("⚙️ Generate Draft", type="primary", width="stretch"):
        with st.spinner("Generating grounded sections… (local LLM can take a moment)"):
            draft = gen.generate_draft(include_optional=include_optional)
        st.session_state[state.KEY_PROPOSAL_DRAFT] = draft
        st.success("Draft generated. Review, refine per section, and export below.")


def _render_draft() -> None:
    draft = st.session_state.get(state.KEY_PROPOSAL_DRAFT)
    gen = st.session_state.get(state.KEY_PROPOSAL_GEN)
    if draft is None or gen is None:
        return

    st.divider()
    st.markdown(f"### Draft — {draft.opportunity_title}")
    st.caption(
        f"Generated {draft.generated_at} · "
        f"{'model: ' + draft.llm_model if draft.llm_model else 'grounded scaffold (no LLM)'}"
    )

    for section in draft.sections:
        with st.expander(f"📄 {section.title}", expanded=section.section_id == "executive_summary"):
            tag = "🧠 AI prose" if section.used_llm else "🔤 grounded scaffold"
            st.caption(
                f"{tag} · {len(section.solicitation_sources)} solicitation + "
                f"{len(section.kb_sources)} KB source(s)"
            )
            st.markdown(section.content)

            with st.popover("📚 Sources / citations", use_container_width=True):
                if section.solicitation_sources:
                    st.markdown("**From the solicitation:**")
                    for s in section.solicitation_sources:
                        st.caption(f"[{s.label}] {s.snippet}")
                if section.kb_sources:
                    st.markdown("**From your knowledge base:**")
                    for s in section.kb_sources:
                        st.caption(f"[{s.label}] {s.snippet}")
                if not section.sources:
                    st.caption("No grounding sources retrieved for this section.")

            _render_section_feedback(section, gen, draft)

    _render_exports(draft)


def _render_section_feedback(section, gen: ProposalGenerator, draft) -> None:
    fb_key = f"fb_{section.section_id}"
    feedback = st.text_input(
        "Refine this section (chat-style feedback)",
        key=fb_key,
        placeholder='e.g. "Make the technical approach more focused on liquid cooling"',
    )
    cols = st.columns([1, 1, 4])
    if cols[0].button("🔁 Regenerate", key=f"regen_{section.section_id}", width="stretch"):
        spec = _SPEC_BY_ID[section.section_id]
        with st.spinner(f"Regenerating {section.title}…"):
            new_section = gen.generate_section(
                spec, feedback=feedback or None, prior=section
            )
        draft.replace_section(new_section)
        st.session_state[state.KEY_PROPOSAL_DRAFT] = draft
        st.rerun()
    if section.feedback_history:
        cols[2].caption("Revisions: " + " → ".join(section.feedback_history[-3:]))


def _render_exports(draft) -> None:
    st.divider()
    st.markdown("**Export**")
    c1, c2 = st.columns(2)
    md = draft.to_markdown().encode("utf-8")
    c1.download_button(
        "⬇️ Export Markdown",
        data=md,
        file_name=f"proposal_{draft.notice_id}.md",
        mime="text/markdown",
        width="stretch",
    )
    try:
        docx_bytes = draft_to_docx_bytes(draft)
        c2.download_button(
            "⬇️ Export DOCX",
            data=docx_bytes,
            file_name=f"proposal_{draft.notice_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width="stretch",
        )
    except Exception as exc:  # noqa: BLE001
        c2.caption(f"DOCX export unavailable: {exc}")
