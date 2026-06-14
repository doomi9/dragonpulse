"""Search view: run a search and render the results table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dragonpulse.api.base import (
    SamApiError,
    SamAuthError,
    SamRateLimitError,
)
from dragonpulse.cache.request_budget import RequestBudgetExceeded
from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.ui import state

logger = get_logger(__name__)


def run_search(filters: OpportunityFilters, force_refresh: bool = False) -> None:
    """Execute a search and stash the result in session state."""
    client = state.get_opportunities_client()
    try:
        with st.spinner("Searching SAM.gov (cache-first)…"):
            result = client.search(filters, force_refresh=force_refresh)
        state.set_result(result)
        src = "cache" if result.from_cache else "live API"
        st.toast(f"Found {result.total_records} opportunities (from {src})")
    except SamAuthError as exc:
        st.error(f"Authentication problem: {exc}")
    except SamRateLimitError as exc:
        st.error(f"SAM.gov rate limit hit: {exc}")
    except RequestBudgetExceeded as exc:
        st.warning(f"Daily request budget reached: {exc}")
    except SamApiError as exc:
        st.error(f"SAM.gov API error: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors to UI
        logger.exception("Unexpected search failure")
        st.error(f"Unexpected error: {exc}")


def render_search(filters: OpportunityFilters) -> None:
    """Render the search controls and results table."""
    st.subheader("Discover opportunities")

    col1, col2, col3 = st.columns([1, 1, 3])
    if col1.button("🔍 Search", type="primary", width="stretch"):
        run_search(filters, force_refresh=False)
    if col2.button("↻ Force refresh", width="stretch",
                   help="Bypass cache and spend one live request."):
        run_search(filters, force_refresh=True)

    with st.expander("Active filters (translated to SAM params)", expanded=False):
        st.json(filters.to_query_params())

    result = state.get_result()
    if result is None:
        st.info("Set filters in the sidebar and click **Search** to begin.")
        return

    source = "📁 cache" if result.from_cache else "🌐 live API"
    st.caption(
        f"Showing {result.count} of {result.total_records} total · source: {source}"
        + (f" · fetched {result.fetched_at:%Y-%m-%d %H:%M}" if result.fetched_at else "")
    )

    if not result.opportunities:
        st.warning("No opportunities matched these filters.")
        return

    rows = [opp.to_table_row() for opp in result.opportunities]
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("SAM.gov", display_text="open ↗"),
            "Days Left": st.column_config.NumberColumn("Days Left", format="%d"),
        },
    )

    # Selection for the detail view.
    options = {f"{opp.title or '(untitled)'}  ·  {opp.notice_id}": opp.notice_id
               for opp in result.opportunities}
    label = st.selectbox(
        "Open an opportunity detail",
        options=["—"] + list(options.keys()),
        index=0,
    )
    if label != "—":
        state.select_opportunity(options[label])

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Export results (CSV)",
        data=csv,
        file_name="dragonpulse_results.csv",
        mime="text/csv",
    )
