"""Sidebar filters and status panel."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dragonpulse.cache.disk_cache import DiskCache
from dragonpulse.config.settings import get_settings
from dragonpulse.models.common import SET_ASIDE_CHOICES, NoticeType
from dragonpulse.models.filters import OpportunityFilters

# Common NAICS codes shown in the multiselect. The firm's primary codes are
# listed first; additional codes can be configured via DRAGONPULSE_DEFAULT_NAICS
# or typed into the "Additional NAICS" box.
COMMON_NAICS = [
    "237130 — Power & Communication Line / Related Structures Construction",
    "541330 — Engineering Services",
    "541511 — Custom Computer Programming Services",
    "541512 — Computer Systems Design Services",
    "541519 — Other Computer Related Services",
    "541611 — Admin & General Management Consulting",
    "541618 — Other Management Consulting Services",
    "561210 — Facilities Support Services",
    "611430 — Professional & Management Training",
]


def _naics_code(label: str) -> str:
    return label.split("—")[0].strip()


def render_sidebar() -> OpportunityFilters:
    """Render the sidebar and return the assembled :class:`OpportunityFilters`."""
    settings = get_settings()

    st.sidebar.title("🐉 DragonPulse")
    st.sidebar.caption("Local-first SAM.gov opportunity intelligence")

    _render_status(settings)

    st.sidebar.header("Search filters")

    keyword = st.sidebar.text_input(
        "Keyword (title search)",
        value="",
        placeholder="e.g. cybersecurity, staffing, logistics",
        help="Matches against the opportunity title (SAM 'title' param).",
    )

    # Merge any configured default NAICS codes into the option list + selection.
    default_codes = settings.default_naics
    options = list(COMMON_NAICS)
    known_codes = {_naics_code(lbl) for lbl in options}
    for code in default_codes:
        if code not in known_codes:
            options.append(code)  # show bare code if not in the friendly list
    default_selection = [
        lbl for lbl in options if _naics_code(lbl) in default_codes
    ]

    naics_labels = st.sidebar.multiselect(
        "NAICS codes",
        options=options,
        default=default_selection,
        help="Pre-filled from DRAGONPULSE_DEFAULT_NAICS; edit freely.",
    )
    naics_extra = st.sidebar.text_input(
        "Additional NAICS (comma-separated)",
        value="",
        placeholder="541512, 541330",
    )

    set_aside_labels = st.sidebar.multiselect(
        "Set-asides",
        options=list(SET_ASIDE_CHOICES.keys()),
        format_func=lambda c: f"{c} — {SET_ASIDE_CHOICES[c]}",
        default=[],
    )

    notice_labels = st.sidebar.multiselect(
        "Notice types",
        options=[nt for nt in NoticeType],
        format_func=lambda nt: nt.label,
        default=[
            NoticeType.SOLICITATION,
            NoticeType.COMBINED_SYNOPSIS_SOLICITATION,
            NoticeType.SOURCES_SOUGHT,
        ],
    )

    department = st.sidebar.text_input(
        "Department / agency name",
        value="",
        placeholder="e.g. GENERAL SERVICES ADMINISTRATION",
        help="Optional exact department name (SAM 'deptname' param).",
    )

    st.sidebar.subheader("Date range (posted)")
    default_from = date.today() - timedelta(days=30)
    col1, col2 = st.sidebar.columns(2)
    posted_from = col1.date_input("From", value=default_from, max_value=date.today())
    posted_to = col2.date_input("To", value=date.today(), max_value=date.today())

    limit = st.sidebar.slider(
        "Max results",
        min_value=5,
        max_value=100,
        value=25,
        step=5,
        help="Keep small on the basic 10/day key to conserve requests.",
    )

    # Assemble NAICS codes from both inputs.
    naics_codes = [_naics_code(lbl) for lbl in naics_labels]
    naics_codes += [c.strip() for c in naics_extra.split(",") if c.strip()]

    filters = OpportunityFilters(
        keyword=keyword or None,
        naics_codes=naics_codes,
        set_aside_codes=set_aside_labels,
        notice_type_codes=[nt.value for nt in notice_labels],
        department_name=department or None,
        posted_from=posted_from,
        posted_to=posted_to,
        limit=limit,
    )
    return filters


def _render_status(settings) -> None:
    """Render key/cache/budget status in the sidebar."""
    with st.sidebar.expander("Status & cache", expanded=False):
        if settings.has_api_key:
            st.success(f"API key: {settings.masked_api_key()} ({settings.api_key_tier.value})")
        else:
            st.error("No SAM.gov API key set. Add DRAGONPULSE_SAM_API_KEY_BASIC to .env.")

        from dragonpulse.cache.request_budget import RequestBudget

        budget = RequestBudget(settings.cache_dir, settings.daily_request_budget)
        st.metric(
            "Live requests today",
            f"{budget.used_today()}/{settings.daily_request_budget}",
            help="Cached results do not count against this budget.",
        )

        cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
        stats = cache.stats()
        st.caption(
            f"Cache: {stats['fresh']} fresh / {stats['files']} files "
            f"({stats['bytes'] / 1024:.0f} KB)"
        )
        st.caption(f"TTL: {settings.cache_ttl_seconds // 3600}h")
        if st.button("Clear cache", width="stretch"):
            n = cache.clear()
            st.toast(f"Cleared {n} cached files")

        st.caption(f"LLM: {'on' if settings.llm_active else 'off (template mode)'}")
