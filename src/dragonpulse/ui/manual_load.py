"""Manually load a SAM.gov opportunity with **zero API calls**.

Two low-friction paths, both fully local (no network):

1. **Upload the solicitation PDF(s)** — the primary path. The user drops the SOW
   they downloaded from SAM.gov; DragonPulse extracts the text (OCR'ing scanned
   pages), builds a local opportunity record, indexes the solicitation, selects
   it, and lands ready to draft — all in one click.
2. **Paste a SAM.gov link or Notice ID** — optional, for linking back to the
   posting or when no file is handy.

Either way the record is registered in the shared pool so the Detail, Proposal
Generator, and Compliance Matrix all treat it like any other opportunity.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.opportunity import Opportunity, parse_opportunity_reference
from dragonpulse.processors.sam_scrape import SamScrapeError, fetch_opportunity_from_link
from dragonpulse.processors.text_extract import UnsupportedDocument, extract_text_with_ocr
from dragonpulse.ui import state

logger = get_logger(__name__)

_UPLOAD_TYPES = ["pdf", "docx", "txt", "md"]


def render_manual_loader(
    *,
    key_prefix: str,
    expanded: bool = False,
    allow_upload: bool = False,
) -> Optional[Opportunity]:
    """Render the manual-load widget. Returns the loaded opportunity, if any.

    Parameters
    ----------
    key_prefix:
        Namespacing prefix for widget keys (the widget can appear on >1 tab).
    expanded:
        Whether the expander starts open.
    allow_upload:
        When True (Proposals tab), show the **PDF-first** flow: the user can load
        an opportunity *and* its solicitation in a single step. The extracted
        solicitation text is stashed so the Proposal Generator picks it up
        automatically. When False (Discover), only the link/ID path is shown.
    """
    title = (
        "📌 Manually load an opportunity (upload PDF) — zero API calls"
        if allow_upload
        else "📌 Manually load an opportunity — no API call"
    )
    with st.expander(title, expanded=expanded):
        if allow_upload:
            return _render_upload_first(key_prefix)
        return _render_reference_only(key_prefix)


# --------------------------------------------------------------------------- #
# Load from a SAM.gov link (public page parsing, no keyed API)
# --------------------------------------------------------------------------- #
def render_sam_link_loader(
    *, key_prefix: str, expanded: bool = False
) -> Optional[Opportunity]:
    """Render the "Load from SAM.gov link" widget. Returns the loaded opp, if any.

    Reads SAM.gov's public page data to auto-fill the opportunity (title, agency,
    NAICS, deadline, place of performance, description, attachments) — using
    **zero of the user's SAM.gov API requests**. The scraped scope/description is
    indexed automatically so the user can draft right away.
    """
    with st.expander("🔗 Load from SAM.gov link (no API call)", expanded=expanded):
        st.caption(
            "Paste a full SAM.gov opportunity URL and DragonPulse reads the **public "
            "page data** to auto-fill the title, agency, NAICS, deadline, scope, and "
            "attachment list — **zero of your SAM.gov API requests** are used."
        )
        url = st.text_input(
            "SAM.gov opportunity URL",
            key=f"{key_prefix}_samlink_url",
            placeholder="https://sam.gov/workspace/contract/opp/<ID>/view",
        )
        if st.button(
            "🔗 Fetch & load from SAM.gov (no API call)",
            key=f"{key_prefix}_samlink_btn",
            type="primary",
            width="stretch",
        ):
            return _load_from_link(url)
    return None


def _load_from_link(url: str) -> Optional[Opportunity]:
    if not (url or "").strip():
        st.warning("Paste a SAM.gov opportunity link first.")
        return None
    try:
        with st.spinner("Reading the public SAM.gov page (no API call)…"):
            scraped = fetch_opportunity_from_link(url)
    except SamScrapeError as exc:
        st.error(
            f"{exc}\n\nTip: you can still **upload the solicitation PDF** above to "
            "load this opportunity without a link."
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never crash the tab
        logger.exception("Unexpected error scraping SAM.gov link")
        st.error(f"Couldn't load that link: {exc}")
        return None

    opp = scraped.opportunity
    state.register_opportunities([opp])
    state.select_opportunity(opp.notice_id)

    # Index the scraped scope/description so the user can draft immediately.
    if scraped.description.strip():
        stash = st.session_state.setdefault(state.KEY_MANUAL_SOLICITATION, {})
        stash[opp.notice_id] = [
            ("SAM.gov description (scope summary)", scraped.description)
        ]
        attempted = st.session_state.get(state.KEY_PROPOSAL_AUTOLOAD)
        if isinstance(attempted, set):
            attempted.discard(opp.notice_id)

    deadline = opp.response_deadline
    bits = [
        f"**{opp.title}**",
        f"Agency: {opp.agency or '—'}",
        f"NAICS: {opp.naics_code or '—'}",
        f"Deadline: {deadline.strftime('%Y-%m-%d') if deadline else '—'}",
        f"Attachments: {len(opp.resource_links)}",
    ]
    st.success(
        "✅ Loaded from SAM.gov — **zero API calls**.\n\n" + " · ".join(bits) + ".\n\n"
        "The scope summary is indexed; add the SOW PDF below for fuller grounding, "
        "then generate the draft."
    )
    if scraped.description.strip():
        with st.expander("📄 Scope / description (from SAM.gov)", expanded=False):
            st.write(scraped.description[:4000])
    st.rerun()
    return opp


# --------------------------------------------------------------------------- #
# PDF-first flow (Proposals tab)
# --------------------------------------------------------------------------- #
def _render_upload_first(key_prefix: str) -> Optional[Opportunity]:
    st.caption(
        "Found it on SAM.gov? **Upload the solicitation / SOW PDF and start "
        "drafting in one step.** Everything is processed locally — **zero SAM.gov "
        "API calls**. Scanned PDFs are OCR'd automatically."
    )

    st.markdown("**1. Upload the solicitation / SOW** (primary — this is all you need)")
    files = st.file_uploader(
        "Upload the solicitation / SOW file(s)",
        type=_UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"{key_prefix}_manual_files",
        label_visibility="collapsed",
        help="PDF, DOCX, TXT, or MD. Scanned PDFs are OCR'd locally.",
    )

    st.markdown("**2. Details** (optional — add any you have)")
    raw = st.text_input(
        "SAM.gov link or Notice ID (optional)",
        key=f"{key_prefix}_manual_ref",
        placeholder="https://sam.gov/opp/<NOTICE_ID>/view  —or—  <NOTICE_ID>",
    )
    c1, c2 = st.columns(2)
    title_in = c1.text_input("Title (optional)", key=f"{key_prefix}_manual_title")
    agency = c2.text_input("Agency (optional)", key=f"{key_prefix}_manual_agency")
    with st.expander("More details (optional)", expanded=False):
        d1, d2 = st.columns(2)
        sol = d1.text_input("Solicitation # (optional)", key=f"{key_prefix}_manual_sol")
        naics = d2.text_input("NAICS (optional)", key=f"{key_prefix}_manual_naics")

    if st.button(
        "🚀 Load & start drafting (no API call)",
        key=f"{key_prefix}_manual_btn",
        type="primary",
        width="stretch",
    ):
        return _load_with_upload(
            files=files,
            raw=raw,
            title_in=title_in,
            agency=agency,
            sol=sol,
            naics=naics,
        )
    return None


def _load_with_upload(
    *,
    files,
    raw: str,
    title_in: str,
    agency: str,
    sol: str,
    naics: str,
) -> Optional[Opportunity]:
    raw = (raw or "").strip()
    if not files and not raw:
        st.warning(
            "Upload the solicitation PDF (recommended) or paste a SAM.gov link / "
            "Notice ID to load an opportunity."
        )
        return None

    # Resolve the Notice ID + link: parse a pasted reference, else mint a local id.
    ui_link: Optional[str] = None
    if raw:
        try:
            notice_id, ui_link = parse_opportunity_reference(raw)
        except ValueError as exc:
            st.error(str(exc))
            return None
    else:
        notice_id = f"MANUAL-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"

    # Extract uploaded solicitation file(s) locally (OCR scanned PDFs).
    extracted, errors = _extract_files(files)
    for err in errors:
        st.error(err)
    if files and not extracted:
        # The user clearly intended to upload but nothing came through.
        st.error("Couldn't read any uploaded file. Try a different file or paste the SOW text.")
        return None

    # A friendly default title from the first file when none was provided.
    title = title_in.strip() or (Path(extracted[0][0]).stem if extracted else None)

    opp = Opportunity.manual(
        notice_id,
        title=title,
        ui_link=ui_link,
        solicitation_number=sol.strip() or None,
        naics_code=naics.strip() or None,
        agency=agency.strip() or None,
    )
    state.register_opportunities([opp])
    state.select_opportunity(opp.notice_id)

    # Stash the extracted solicitation so the generator auto-indexes it on rerun.
    if extracted:
        stash = st.session_state.setdefault(state.KEY_MANUAL_SOLICITATION, {})
        stash[opp.notice_id] = extracted
        # Make sure the generator re-runs its one-time auto-load for this id.
        attempted = st.session_state.get(state.KEY_PROPOSAL_AUTOLOAD)
        if isinstance(attempted, set):
            attempted.discard(opp.notice_id)

    if extracted:
        st.success(
            f"✅ Loaded **{opp.title}** with **{len(extracted)} solicitation file(s)** — "
            "**zero SAM.gov API calls**. Indexing now; you can generate the draft below."
        )
    else:
        st.success(
            f"✅ Loaded Notice ID **{notice_id}** with **zero API calls**. Add the "
            "solicitation (paste text or upload a PDF) below, then generate."
        )
    st.rerun()
    return opp


def _extract_files(files) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Extract text from uploaded files locally (OCR fallback). No API calls."""
    if not files:
        return [], []
    settings = state.settings()
    extracted: List[Tuple[str, str]] = []
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
            if text and text.strip():
                extracted.append((f.name, text))
            else:
                errors.append(f"{f.name}: no extractable text found.")
        except UnsupportedDocument as exc:
            errors.append(f"{f.name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the tab
            logger.exception("Manual solicitation upload failed for %s", f.name)
            errors.append(f"{f.name}: unexpected error: {exc}")
    progress.progress(1.0, text="Done.")
    progress.empty()
    return extracted, errors


# --------------------------------------------------------------------------- #
# Reference-only flow (Discover tab)
# --------------------------------------------------------------------------- #
def _render_reference_only(key_prefix: str) -> Optional[Opportunity]:
    st.caption(
        "At your daily API limit? Paste a SAM.gov link or Notice ID. DragonPulse "
        "builds a local record from what you provide — **zero SAM.gov API calls** — "
        "then you add the solicitation text/PDF in the Proposal Generator."
    )
    raw = st.text_input(
        "SAM.gov link or Notice ID",
        key=f"{key_prefix}_manual_ref",
        placeholder="https://sam.gov/opp/<NOTICE_ID>/view  —or—  <NOTICE_ID>",
    )
    c1, c2, c3 = st.columns(3)
    title = c1.text_input("Title (optional)", key=f"{key_prefix}_manual_title")
    sol = c2.text_input("Solicitation # (optional)", key=f"{key_prefix}_manual_sol")
    naics = c3.text_input("NAICS (optional)", key=f"{key_prefix}_manual_naics")
    agency = st.text_input("Agency (optional)", key=f"{key_prefix}_manual_agency")

    if st.button(
        "📌 Load opportunity (no API call)",
        key=f"{key_prefix}_manual_btn",
        type="primary",
        width="stretch",
    ):
        try:
            notice_id, ui_link = parse_opportunity_reference(raw)
        except ValueError as exc:
            st.error(str(exc))
            return None
        opp = Opportunity.manual(
            notice_id,
            title=title.strip() or None,
            ui_link=ui_link,
            solicitation_number=sol.strip() or None,
            naics_code=naics.strip() or None,
            agency=agency.strip() or None,
        )
        state.register_opportunities([opp])
        state.select_opportunity(opp.notice_id)
        st.success(
            f"✅ Loaded Notice ID **{notice_id}** with **zero API calls**. "
            "It's now selectable in Detail and the Proposal Generator."
        )
        return opp
    return None
