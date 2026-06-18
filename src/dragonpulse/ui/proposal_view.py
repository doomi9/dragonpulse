"""Proposal Generator view.

Combines an opportunity's solicitation (attachment text) with the company's RAG
knowledge base to generate grounded, cited proposal sections. Supports
per-section regeneration with chat-style feedback and export to Markdown / DOCX.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.models.proposal import COMPLIANCE_STATUSES
from dragonpulse.processors.attachments import download_and_extract
from dragonpulse.processors.proposal import (
    SECTION_SPECS,
    ProposalGenerator,
    compliance_matrix_to_xlsx_bytes,
    draft_to_docx_bytes,
)
from dragonpulse.processors.text_extract import UnsupportedDocument, extract_text_with_ocr
from dragonpulse.ui import state
from dragonpulse.ui.manual_load import render_manual_loader, render_sam_link_loader

_SOL_UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]

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
    _render_saved_drafts(opp)
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

    opportunities = state.all_opportunities()
    selected = state.get_selected()

    # Lead with the low-friction no-API paths. Open them by default when there's
    # nothing loaded yet: paste a SAM.gov link, or drop the solicitation PDF.
    render_sam_link_loader(key_prefix="proposal", expanded=not opportunities)
    render_manual_loader(
        key_prefix="proposal", allow_upload=True, expanded=not opportunities
    )

    if not opportunities:
        st.info(
            "No opportunities loaded yet. Fastest no-API starts: **🔗 Load from "
            "SAM.gov link** (auto-fills everything from the public page) or **📌 "
            "Manually load** (drop the solicitation PDF). You can also run a search "
            "in **Discover**, pick one from **⭐ Priority Picks**, or open one in "
            "**Detail**."
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
        if opp.loaded_via == "sam_link":
            deadline = opp.response_deadline
            st.caption(
                f"🔗 **{opp.title}** · loaded from SAM.gov link (no API call) · "
                f"{opp.agency or '—'} · NAICS {opp.naics_code or '—'} · "
                f"deadline {deadline.strftime('%Y-%m-%d') if deadline else '—'} · "
                f"{len(opp.resource_links)} attachment(s) · "
                f"[open on SAM.gov ↗]({opp.sam_url})"
            )
        elif opp.manual_entry:
            st.caption(
                f"📌 **{opp.title}** · manually loaded (no API data) · "
                f"NAICS {opp.naics_code or '—'} · [open on SAM.gov ↗]({opp.sam_url})"
            )
        else:
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
        st.session_state.pop(state.KEY_PROPOSAL_LOAD_MSG, None)
        st.session_state.pop(state.KEY_PROPOSAL_LOADED_FROM, None)
    return gen


def _render_solicitation_loader(opp: Opportunity, gen: ProposalGenerator) -> None:
    st.markdown("**2. Solicitation**")

    # Auto-load attachments (API opps) or the uploaded solicitation (manual opps)
    # the first time this opportunity is selected.
    _auto_load_solicitation(opp, gen)

    already_loaded = bool(st.session_state.get(state.KEY_PROPOSAL_ATTACH))
    if opp.manual_entry and not already_loaded:
        st.info(
            "📌 Manually loaded opportunity — **no API calls**. Add the solicitation "
            "by pasting the SOW text or uploading the solicitation PDF(s) below.",
            icon="📌",
        )

    loaded = st.session_state.get(state.KEY_PROPOSAL_ATTACH, [])
    msg = st.session_state.get(state.KEY_PROPOSAL_LOAD_MSG)
    if msg:
        level, text = msg
        getattr(st, level, st.info)(text)

    if loaded:
        st.caption("✅ Solicitation context loaded from: " + "; ".join(loaded))
    else:
        st.caption(
            "No solicitation text yet. Reload the attachments or paste the SOW below. "
            "(You can still generate from KB + metadata, but grounding will be weaker.)"
        )

    cols = st.columns([1, 3])
    if cols[0].button(
        "🔄 Reload attachments",
        width="stretch",
        disabled=not opp.resource_links,
        help="Re-download the opportunity's attachments and re-index their text.",
    ):
        _load_attachments(opp, gen, reset=True)
        st.rerun()

    with st.expander("➕ Add solicitation — paste text or upload PDF(s)", expanded=not loaded):
        st.markdown("**Paste solicitation / SOW text**")
        pasted = st.text_area(
            "Paste solicitation / SOW text",
            height=140,
            placeholder="Paste the statement of work, Section L/M, or key requirements "
            "here if attachments are scanned images or unavailable.",
            label_visibility="collapsed",
        )
        if st.button("Add pasted text to solicitation", width="stretch"):
            if pasted.strip():
                n = gen.load_solicitation([("Pasted text", pasted)])
                current = st.session_state.get(state.KEY_PROPOSAL_ATTACH, [])
                current.append(f"Pasted text ({n} chunks)")
                st.session_state[state.KEY_PROPOSAL_ATTACH] = current
                st.success(f"Indexed pasted text into {n} chunks.")
                st.rerun()
            else:
                st.warning("Nothing to add — paste some text first.")

        st.divider()
        st.markdown("**Or upload the solicitation file(s)** (PDF, DOCX, TXT, MD)")
        st.caption("Processed locally — no SAM.gov API calls. Scanned PDFs are OCR'd.")
        files = st.file_uploader(
            "Upload solicitation file(s)",
            type=_SOL_UPLOAD_TYPES,
            accept_multiple_files=True,
            key=f"sol_upload_{opp.notice_id}",
            label_visibility="collapsed",
        )
        if st.button("Add uploaded file(s) to solicitation", width="stretch"):
            _add_uploaded_solicitation(files, gen)


def _auto_load_solicitation(opp: Opportunity, gen: ProposalGenerator) -> None:
    """Download + extract the opportunity's attachments once, automatically."""
    attempted = st.session_state.setdefault(state.KEY_PROPOSAL_AUTOLOAD, set())
    if opp.notice_id in attempted:
        return
    attempted.add(opp.notice_id)
    if opp.manual_entry:
        # If the user uploaded the solicitation when loading manually, index it
        # now so they land ready to generate — no extra clicks, no API calls.
        stash = st.session_state.get(state.KEY_MANUAL_SOLICITATION, {}).get(opp.notice_id)
        if stash:
            n = gen.load_solicitation(stash)
            names = [name for name, _ in stash]
            st.session_state[state.KEY_PROPOSAL_ATTACH] = names + [f"{n} chunks indexed"]
            st.session_state[state.KEY_PROPOSAL_LOAD_MSG] = (
                "success",
                f"✅ Indexed {len(stash)} uploaded solicitation file(s) into {n} "
                "chunks — zero API calls. Ready to generate below.",
            )
        # Otherwise the manual-entry banner explains how to add the solicitation.
        return
    if not opp.resource_links:
        st.session_state[state.KEY_PROPOSAL_LOAD_MSG] = (
            "info",
            "ℹ️ This opportunity has no downloadable attachments — paste the SOW "
            "text below to ground the draft.",
        )
        return
    with st.spinner(
        f"Auto-loading {len(opp.resource_links)} attachment(s) for this opportunity…"
    ):
        _load_attachments(opp, gen, reset=True)


def _load_attachments(opp: Opportunity, gen: ProposalGenerator, *, reset: bool = False) -> None:
    if reset:
        gen.solicitation = type(gen.solicitation)(gen.backend)
    attachments: List[Tuple[str, str]] = []
    names: List[str] = []
    skipped: List[str] = []
    for link in opp.resource_links:
        att = download_and_extract(link.url, settings=gen.settings)
        if att.error:
            skipped.append(f"{att.filename}: {att.error}")
            continue
        if att.text:
            attachments.append((att.filename, att.text))
            names.append(att.filename)
        else:
            skipped.append(f"{att.filename}: no extractable text (may be scanned)")

    if not attachments:
        st.session_state[state.KEY_PROPOSAL_ATTACH] = []
        st.session_state[state.KEY_PROPOSAL_LOAD_MSG] = (
            "warning",
            "⚠️ Couldn't extract text from the attachments (they may be scanned "
            "images). Paste the SOW text manually below.",
        )
        return

    n = gen.load_solicitation(attachments)
    st.session_state[state.KEY_PROPOSAL_ATTACH] = list(names) + [f"{n} chunks indexed"]
    detail = f"✅ Loaded {len(attachments)} file(s) into {n} searchable chunks."
    if skipped:
        detail += f" Skipped {len(skipped)}: " + "; ".join(skipped[:3])
    st.session_state[state.KEY_PROPOSAL_LOAD_MSG] = ("success", detail)


def _add_uploaded_solicitation(files, gen: ProposalGenerator) -> None:
    """Extract uploaded solicitation file(s) locally and index them (no API call)."""
    if not files:
        st.warning("Choose one or more files first.")
        return
    settings = gen.settings
    extracted: List[Tuple[str, str]] = []
    names: List[str] = []
    errors: List[str] = []
    progress = st.progress(0.0, text="Starting…")
    for idx, f in enumerate(files, start=1):
        base = (idx - 1) / len(files)
        progress.progress(base, text=f"Reading {f.name}…")
        try:
            def _on_page(done, total, _f=f, _base=base):
                progress.progress(
                    _base + (done / total) / len(files),
                    text=f"OCR {_f.name}: page {done}/{total}…",
                )

            text = extract_text_with_ocr(
                f.getvalue(),
                f.name,
                ocr_enabled=settings.kb_ocr_enabled,
                dpi=settings.kb_ocr_dpi,
                page_callback=_on_page,
            )
            extracted.append((f.name, text))
            names.append(f.name)
        except UnsupportedDocument as exc:
            errors.append(f"{f.name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Solicitation upload failed for %s", f.name)
            errors.append(f"{f.name}: unexpected error: {exc}")
    progress.progress(1.0, text="Done.")
    progress.empty()

    for err in errors:
        st.error(err)
    if not extracted:
        return
    n = gen.load_solicitation(extracted)
    current = st.session_state.get(state.KEY_PROPOSAL_ATTACH, [])
    current.extend(names)
    current.append(f"{n} chunks indexed")
    st.session_state[state.KEY_PROPOSAL_ATTACH] = current
    st.success(f"Indexed {len(extracted)} uploaded file(s) into {n} chunks (no API call).")
    st.rerun()


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
        with st.spinner("Building compliance matrix from the solicitation…"):
            try:
                draft.compliance = gen.extract_compliance_matrix(draft)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Compliance matrix generation failed: %s", exc)
        st.session_state[state.KEY_PROPOSAL_DRAFT] = draft
        st.session_state.pop(state.KEY_PROPOSAL_LOADED_FROM, None)
        st.success("Draft generated. Review, refine per section, and export below.")


# --------------------------------------------------------------------------- #
# Saved drafts + version history
# --------------------------------------------------------------------------- #
def _render_saved_drafts(opp: Opportunity) -> None:
    store = state.get_draft_store()
    draft = st.session_state.get(state.KEY_PROPOSAL_DRAFT)
    saved = store.list_drafts(notice_id=opp.notice_id)

    with st.expander(f"💾 Saved drafts for this opportunity ({len(saved)})", expanded=False):
        # Save the current draft.
        if draft is not None:
            loaded_from = st.session_state.get(state.KEY_PROPOSAL_LOADED_FROM)
            c1, c2 = st.columns([3, 1])
            default_name = f"Draft {time.strftime('%Y-%m-%d %H:%M')}"
            name = c1.text_input(
                "Name this draft", value=default_name, key="save_draft_name",
                label_visibility="collapsed",
            )
            if c2.button("💾 Save", width="stretch"):
                store.save(name, draft)
                st.success(f"Saved '{name}'.")
                st.rerun()
            if loaded_from and store.get(loaded_from):
                if st.button("⤴️ Update the loaded saved draft (new version)", width="stretch"):
                    updated = store.update(loaded_from, draft)
                    if updated:
                        st.success(f"Updated '{updated.name}' to v{updated.version}.")
                    st.rerun()
        else:
            st.caption("Generate a draft to save it here.")

        if not saved:
            st.caption("No saved drafts yet for this opportunity.")
            return

        st.divider()
        for sd in saved:
            with st.container(border=True):
                cols = st.columns([4, 1, 1])
                cols[0].markdown(f"**{sd.name}**  ·  v{sd.version}")
                cols[0].caption(
                    f"{sd.section_count} sections · created {sd.created_at} · "
                    f"last modified {sd.modified_at}"
                )
                if cols[1].button("📂 Load", key=f"load_{sd.draft_id}", width="stretch"):
                    st.session_state[state.KEY_PROPOSAL_DRAFT] = sd.draft
                    st.session_state[state.KEY_PROPOSAL_LOADED_FROM] = sd.draft_id
                    st.success(f"Loaded '{sd.name}'.")
                    st.rerun()
                if cols[2].button("🗑️", key=f"deld_{sd.draft_id}", width="stretch",
                                  help="Delete this saved draft"):
                    store.delete(sd.draft_id)
                    if st.session_state.get(state.KEY_PROPOSAL_LOADED_FROM) == sd.draft_id:
                        st.session_state.pop(state.KEY_PROPOSAL_LOADED_FROM, None)
                    st.rerun()


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
        with st.expander(
            f"📄 {section.title}",
            expanded=section.section_id in ("title_page", "tab_1_executive_summary"),
        ):
            tag = "🧠 AI prose" if section.used_llm else "🔤 grounded scaffold"
            kb_facts = [s for s in section.kb_sources if s.origin == "knowledge_base"]
            style_refs = section.style_sources
            caption = (
                f"{tag} · {len(section.solicitation_sources)} solicitation + "
                f"{len(kb_facts)} KB source(s)"
            )
            if style_refs:
                caption += f" · ✍️ {len(style_refs)} style reference(s)"
            st.caption(caption)
            st.markdown(section.content)

            with st.popover("📚 Sources / citations", use_container_width=True):
                if section.solicitation_sources:
                    st.markdown("**From the solicitation:**")
                    for s in section.solicitation_sources:
                        st.caption(f"[{s.label}] {s.snippet}")
                if kb_facts:
                    st.markdown("**From your knowledge base:**")
                    for s in kb_facts:
                        st.caption(f"[{s.label}] {s.snippet}")
                if style_refs:
                    st.markdown("**Writing-style references (matched for tone & structure):**")
                    for s in style_refs:
                        st.caption(f"[{s.label}] {s.snippet}")
                if not section.sources:
                    st.caption("No grounding sources retrieved for this section.")

            _render_section_feedback(section, gen, draft)

    _render_compliance_matrix(gen, draft)
    _render_exports(draft)


# --------------------------------------------------------------------------- #
# Compliance matrix
# --------------------------------------------------------------------------- #
def _render_compliance_matrix(gen: ProposalGenerator, draft) -> None:
    st.divider()
    st.markdown("### ✅ Compliance Matrix")
    matrix = draft.compliance

    cols = st.columns([1, 3])
    if cols[0].button("🔄 Regenerate matrix", width="stretch"):
        with st.spinner("Re-extracting requirements from the solicitation…"):
            try:
                draft.compliance = gen.extract_compliance_matrix(draft)
                st.session_state[state.KEY_PROPOSAL_DRAFT] = draft
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build the matrix: {exc}")
        st.rerun()

    if matrix is None or not matrix.items:
        cols[1].caption(
            "No requirements extracted yet. Load the solicitation attachments (or "
            "paste the SOW/Section L & M text) and regenerate — the matrix focuses on "
            "the most important 'shall'/'must' statements and evaluation factors."
        )
        return

    counts = matrix.status_counts()
    method = "🧠 AI-extracted" if matrix.used_llm else "🔤 rule-extracted (shall/must)"
    cols[1].caption(
        f"{method} · {len(matrix.items)} key requirements · "
        f"{counts['Addressed']} addressed · {counts['Partial']} partial · "
        f"{counts['Not Addressed']} not addressed. "
        "Status is an auto-generated starting point — edit **Status** and **Notes** below."
    )

    df = pd.DataFrame(
        [
            {
                "Requirement / Factor": it.requirement,
                "Category": it.category,
                "Source": it.source_label,
                "Proposal Section": it.section_title or "—",
                "Status": it.status,
                "Notes": it.notes or "",
            }
            for it in matrix.items
        ]
    )
    edited = st.data_editor(
        df,
        width="stretch",
        hide_index=True,
        key=f"compliance_editor_{draft.notice_id}",
        column_config={
            "Requirement / Factor": st.column_config.TextColumn(width="large", disabled=True),
            "Category": st.column_config.TextColumn(width="small", disabled=True),
            "Source": st.column_config.TextColumn(width="medium", disabled=True),
            "Proposal Section": st.column_config.TextColumn(width="medium", disabled=True),
            "Status": st.column_config.SelectboxColumn(
                options=COMPLIANCE_STATUSES, width="small", required=True
            ),
            "Notes": st.column_config.TextColumn(width="large"),
        },
    )

    # Persist user edits (status + notes) back onto the matrix items.
    for i, item in enumerate(matrix.items):
        item.status = str(edited.iloc[i]["Status"])
        item.notes = str(edited.iloc[i]["Notes"] or "")
    st.session_state[state.KEY_PROPOSAL_DRAFT] = draft

    with st.popover("📚 Requirement sources", use_container_width=True):
        for i, item in enumerate(matrix.items, start=1):
            st.caption(f"**{i}. [{item.source_label}]** {item.source_snippet}")

    _render_strengthen_action(gen, draft, matrix)

    mc1, mc2 = st.columns(2)
    mc1.download_button(
        "⬇️ Export matrix (Markdown)",
        data=matrix.to_markdown().encode("utf-8"),
        file_name=f"compliance_matrix_{draft.notice_id}.md",
        mime="text/markdown",
        width="stretch",
    )
    try:
        xlsx = compliance_matrix_to_xlsx_bytes(matrix)
        mc2.download_button(
            "⬇️ Export matrix (Excel)",
            data=xlsx,
            file_name=f"compliance_matrix_{draft.notice_id}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    except Exception as exc:  # noqa: BLE001
        mc2.caption(f"Excel export unavailable: {exc}")


def _render_strengthen_action(gen: ProposalGenerator, draft, matrix) -> None:
    st.markdown("**💪 Strengthen a section to better cover a requirement**")
    msg = st.session_state.pop("strengthen_msg", None)
    if msg:
        st.success(msg)
    labels = [
        f"{i}. [{it.status}] {it.requirement[:80]}"
        f"{'…' if len(it.requirement) > 80 else ''}  →  {it.section_title or '—'}"
        for i, it in enumerate(matrix.items, start=1)
    ]
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Requirement to strengthen",
        list(range(len(matrix.items))),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
    )
    if c2.button("💪 Strengthen this section", width="stretch"):
        item = matrix.items[choice]
        section = draft.get_section(item.section_id) if item.section_id else None
        if section is None:
            st.warning("That requirement isn't mapped to a section yet.")
            return
        with st.spinner(f"Strengthening '{section.title}' to address requirement #{choice + 1}…"):
            new_section = gen.strengthen_section_for_requirement(
                section, item.requirement, source_label=item.source_label
            )
            draft.replace_section(new_section)
            gen.remap_compliance(matrix, draft)
        st.session_state[state.KEY_PROPOSAL_DRAFT] = draft
        st.session_state["strengthen_msg"] = (
            f"Strengthened **{new_section.title}** to better address requirement "
            f"#{choice + 1}: “{item.requirement[:90]}”. Review the updated section above; "
            "its revision history shows the change."
        )
        st.rerun()


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
    st.markdown("**Export full proposal**")
    st.caption("The compliance matrix is appended to both exports below.")
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
