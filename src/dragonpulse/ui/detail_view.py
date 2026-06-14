"""Opportunity detail view: metadata, POCs, checklist, attachments."""

from __future__ import annotations

import streamlit as st

from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import get_settings
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.processors.attachments import download_and_extract
from dragonpulse.processors.checklist import build_checklist
from dragonpulse.processors.outreach import CompanyProfile, generate_outreach_email
from dragonpulse.ui import state

logger = get_logger(__name__)


def render_detail() -> None:
    """Render the detail view for the currently selected opportunity."""
    opp = state.get_selected()
    if opp is None:
        st.info("Select an opportunity from the **Discover** tab to see details here.")
        return

    settings = get_settings()
    use_llm = settings.llm_active

    st.markdown(f"### {opp.title or '(untitled opportunity)'}")
    st.markdown(f"[Open on SAM.gov ↗]({opp.sam_url})")
    if not use_llm:
        st.caption("LLM is off — drafts use grounded templates. Enable it in `.env` to opt in.")

    _render_metadata(opp)
    st.divider()
    _render_contacts(opp, use_llm)
    st.divider()
    _render_checklist(opp, use_llm)
    st.divider()
    _render_attachments(opp)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _render_metadata(opp: Opportunity) -> None:
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    deadline = opp.response_deadline
    days = opp.days_until_deadline()
    c1.metric("Notice type", opp.notice_type or "—")
    c2.metric("NAICS", opp.naics_code or "—")
    c3.metric(
        "Days to deadline",
        days if days is not None else "—",
        delta=("past due" if days is not None and days < 0 else None),
        delta_color="inverse",
    )

    meta = {
        "Notice ID": opp.notice_id,
        "Solicitation #": opp.solicitation_number or "—",
        "Agency": opp.agency or "—",
        "Office": opp.office or "—",
        "Full org path": opp.full_parent_path_name or "—",
        "Set-aside": opp.set_aside_description or opp.set_aside_code or "None",
        "Posted": opp.posted_date_raw or "—",
        "Response deadline": deadline.strftime("%Y-%m-%d %H:%M") if deadline else "—",
        "Archive date": opp.archive_date_raw or "—",
        "Place of performance": (
            opp.place_of_performance.one_line() if opp.place_of_performance else "—"
        ),
        "Active": opp.active or "—",
    }
    st.table({"Field": list(meta.keys()), "Value": list(meta.values())})

    if opp.description_link:
        st.caption(f"Full description (API): {opp.description_link}")


def _render_contacts(opp: Opportunity, use_llm: bool) -> None:
    st.subheader("📇 Who to reach out to")
    if not opp.points_of_contact:
        st.info("No points of contact were listed for this opportunity.")
        return

    profile = _company_profile_editor()

    for idx, poc in enumerate(opp.points_of_contact):
        with st.container(border=True):
            cols = st.columns([3, 2])
            with cols[0]:
                suffix = f" · _{poc.poc_type}_" if poc.poc_type else ""
                st.markdown(f"**{poc.display_name}**" + suffix)
                if poc.title:
                    st.caption(poc.title)
                if poc.email:
                    st.markdown(f"✉️ `{poc.email}`")
                if poc.phone:
                    st.markdown(f"📞 `{poc.phone}`")
            with cols[1]:
                gen_key = f"gen_email_{idx}"
                if st.button("✍️ Generate outreach email", key=f"btn_{gen_key}",
                             width="stretch"):
                    draft = generate_outreach_email(opp, poc, profile, use_llm=use_llm)
                    st.session_state[gen_key] = draft

            draft = st.session_state.get(f"gen_email_{idx}")
            if draft:
                tag = "LLM-generated" if draft.used_llm else "template"
                st.text_input("Subject", value=draft.subject, key=f"subj_{idx}")
                st.text_area("Body", value=draft.body, height=320, key=f"body_{idx}")
                st.caption(f"Draft type: {tag} · Grounded in: " + "; ".join(draft.sources))


def _render_checklist(opp: Opportunity, use_llm: bool) -> None:
    st.subheader("✅ What needs to be done")
    items = build_checklist(opp, use_llm=use_llm)
    if not items:
        st.info("No checklist could be derived for this notice.")
        return

    priority_icon = {"high": "🔴", "normal": "🟡", "low": "⚪"}
    for i, item in enumerate(items):
        with st.container(border=True):
            top = st.columns([6, 2])
            with top[0]:
                st.checkbox(
                    f"{priority_icon.get(item.priority, '🟡')} **{item.action}**",
                    key=f"chk_{opp.notice_id}_{i}",
                )
                st.caption(item.detail)
            with top[1]:
                if item.due:
                    st.caption(f"📅 Due: {item.due}")
                st.caption(f"🔗 {item.source}")

    # Export checklist as markdown.
    md = _checklist_to_md(opp, items)
    st.download_button(
        "⬇️ Export checklist (Markdown)",
        data=md.encode("utf-8"),
        file_name=f"checklist_{opp.notice_id}.md",
        mime="text/markdown",
    )


def _render_attachments(opp: Opportunity) -> None:
    st.subheader("📎 Attachments & resources")
    if not opp.resource_links:
        st.info("No resource links / attachments were listed for this opportunity.")
        return

    settings = get_settings()
    st.caption(
        f"{len(opp.resource_links)} resource link(s). "
        f"Downloads cached in {settings.attachments_dir}."
    )

    for idx, link in enumerate(opp.resource_links):
        with st.container(border=True):
            cols = st.columns([5, 2])
            cols[0].markdown(f"`{link.url}`")
            if cols[1].button("⬇️ Download & preview", key=f"dl_{idx}", width="stretch"):
                with st.spinner("Downloading & extracting…"):
                    att = download_and_extract(link.url, settings=settings)
                st.session_state[f"att_{idx}"] = att

            att = st.session_state.get(f"att_{idx}")
            if att:
                if att.error:
                    st.error(f"Could not process: {att.error}")
                else:
                    st.caption(
                        f"**{att.filename}** · {att.content_type or 'unknown type'} · "
                        f"{att.size_bytes / 1024:.0f} KB"
                        + (f" · {att.page_count} pages" if att.page_count else "")
                    )
                    if att.local_path and att.local_path.exists():
                        with open(att.local_path, "rb") as fh:
                            st.download_button(
                                "Save file",
                                data=fh.read(),
                                file_name=att.filename,
                                key=f"save_{idx}",
                            )
                    if att.has_text:
                        with st.expander("Text preview", expanded=False):
                            preview = att.text or ""
                            st.text(preview[:5000])
                            if att.truncated or len(preview) > 5000:
                                st.caption("… preview truncated.")
                    else:
                        st.caption("No text preview available for this file type.")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _company_profile_editor() -> CompanyProfile:
    """Render an editable company profile (used to personalize outreach)."""
    saved = st.session_state.get(state.KEY_PROFILE) or CompanyProfile()
    with st.expander("Your company profile (used to personalize outreach)", expanded=False):
        c1, c2 = st.columns(2)
        company = c1.text_input("Company name", value=saved.company_name)
        caps = c2.text_input("Core capabilities", value=saved.capabilities)
        name = c1.text_input("Your name", value=saved.sender_name)
        title = c2.text_input("Your title", value=saved.sender_title)
        email = c1.text_input("Your email", value=saved.sender_email)
        phone = c2.text_input("Your phone", value=saved.sender_phone)
    profile = CompanyProfile(
        company_name=company,
        sender_name=name,
        sender_title=title,
        sender_email=email,
        sender_phone=phone,
        capabilities=caps,
    )
    st.session_state[state.KEY_PROFILE] = profile
    return profile


def _checklist_to_md(opp: Opportunity, items) -> str:
    lines = [f"# Action checklist — {opp.title or opp.notice_id}", ""]
    lines.append(f"- Notice ID: {opp.notice_id}")
    lines.append(f"- SAM.gov: {opp.sam_url}")
    lines.append("")
    for item in items:
        due = f" (due {item.due})" if item.due else ""
        lines.append(f"- [ ] **{item.action}**{due}")
        lines.append(f"  - {item.detail}")
        lines.append(f"  - _source: {item.source}_")
    return "\n".join(lines)
