"""Pricing intelligence view.

Pulls historical SAM.gov **Award Notices** matching the current sidebar filters
(NAICS, keyword, posted-date range), normalizes them to awards, and renders
pricing statistics, a distribution chart, and a ranked table of comparable
awards. Cache-first and budget-aware, like every other live call.
"""

from __future__ import annotations

import streamlit as st

from dragonpulse.api.base import SamApiError, SamAuthError, SamRateLimitError
from dragonpulse.cache.request_budget import RequestBudgetExceeded
from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.processors.pricing import PricingAnalysis, analyze_awards
from dragonpulse.ui import state

logger = get_logger(__name__)


def _run_pricing_search(filters: OpportunityFilters, max_records: int, force: bool) -> None:
    client = state.get_awards_client()
    try:
        with st.spinner("Collecting award history (cache-first)…"):
            result = client.search_awards(
                filters, max_records=max_records, force_refresh=force
            )
        st.session_state[state.KEY_PRICING] = analyze_awards(result)
        st.toast(f"Analyzed {result.total_records} award notices")
    except SamAuthError as exc:
        st.error(f"Authentication problem: {exc}")
    except SamRateLimitError as exc:
        st.error(f"SAM.gov rate limit hit: {exc}")
    except RequestBudgetExceeded as exc:
        st.warning(f"Daily request budget reached: {exc}")
    except SamApiError as exc:
        st.error(f"SAM.gov API error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected pricing failure")
        st.error(f"Unexpected error: {exc}")


def render_pricing(filters: OpportunityFilters) -> None:
    """Render the pricing intelligence tab."""
    st.subheader("💰 Pricing intelligence")
    st.caption(
        "Analyzes historical **Award Notices** (SAM.gov `ptype=a`) matching your "
        "sidebar filters. Set NAICS + a keyword for the most comparable results."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    max_records = c3.slider(
        "Awards to scan",
        min_value=25,
        max_value=300,
        value=100,
        step=25,
        help="Up to 100 awards = 1 live request. Larger scans paginate (more requests).",
    )
    if c1.button("📊 Analyze pricing", type="primary", width="stretch"):
        _run_pricing_search(filters, max_records, force=False)
    if c2.button("↻ Force refresh", width="stretch",
                 help="Bypass cache and spend live request(s)."):
        _run_pricing_search(filters, max_records, force=True)

    with st.expander("Award query (translated to SAM params)", expanded=False):
        award_params = filters.model_copy(update={"notice_type_codes": ["a"]}).to_query_params()
        st.json(award_params)

    analysis: PricingAnalysis = st.session_state.get(state.KEY_PRICING)
    if analysis is None:
        st.info("Click **Analyze pricing** to pull comparable award history.")
        return

    _render_analysis(analysis)


def _render_analysis(analysis: PricingAnalysis) -> None:
    if analysis.total_awards == 0:
        st.warning("No award notices matched these filters. Try widening the date range or NAICS.")
        return

    st.caption(
        f"{analysis.total_awards} award notice(s) found · "
        f"{analysis.priced_awards} with a parseable award amount."
    )

    if not analysis.has_pricing:
        st.warning(
            "Awards were found, but none had a machine-readable dollar amount "
            "(SAM.gov award notices often omit it). Showing the awards table below."
        )
    else:
        stats = analysis.stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Awards priced", stats.get("count", 0))
        m2.metric("Min", f"${stats.get('min', 0):,.0f}")
        m3.metric("Median", f"${stats.get('median', 0):,.0f}")
        m4.metric("Mean", f"${stats.get('mean', 0):,.0f}")
        m5.metric("Max", f"${stats.get('max', 0):,.0f}")

        if analysis.histogram is not None and not analysis.histogram.empty:
            st.markdown("**Award amount distribution**")
            st.bar_chart(analysis.histogram)

    if analysis.awards_table is not None and not analysis.awards_table.empty:
        st.markdown("**Comparable awards**")
        st.dataframe(
            analysis.awards_table,
            width="stretch",
            hide_index=True,
            column_config={
                "Amount": st.column_config.NumberColumn("Amount", format="$%,.0f"),
                "Link": st.column_config.LinkColumn("SAM.gov", display_text="open ↗"),
            },
        )
        csv = analysis.awards_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export awards (CSV)",
            data=csv,
            file_name="dragonpulse_awards.csv",
            mime="text/csv",
        )
