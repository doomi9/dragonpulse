"""Sidebar filters and status panel."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from dragonpulse.cache.disk_cache import DiskCache
from dragonpulse.config.settings import get_settings
from dragonpulse.models.filters import (
    DEFAULT_NOTICE_TYPE_CODES,
    PRIMARY_NAICS,
    OpportunityFilters,
)
from dragonpulse.ui import theme


def render_sidebar() -> OpportunityFilters:
    """Render the sidebar and return the assembled :class:`OpportunityFilters`.

    Filters are intentionally minimal right now: only **Date range** and **Max
    results** are shown. NAICS codes are hardcoded (237130, 541330) and a
    sensible notice-type default is applied under the hood, so discovery is
    driven by the Knowledge Base (⭐ Priority Picks) rather than manual keywords.
    """
    settings = get_settings()

    theme.render_sidebar_logo()
    st.sidebar.caption("Local-first SAM.gov opportunity intelligence")

    _render_status(settings)

    st.sidebar.header("Search filters")
    st.sidebar.caption(
        f"NAICS locked to **{', '.join(PRIMARY_NAICS)}** for now. Discovery is driven "
        "by your Knowledge Base — see **⭐ Priority Picks**."
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

    filters = OpportunityFilters(
        keyword=None,
        naics_codes=list(PRIMARY_NAICS),
        set_aside_codes=[],
        notice_type_codes=list(DEFAULT_NOTICE_TYPE_CODES),
        department_name=None,
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
