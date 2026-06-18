"""Priority Picks view — the main, Knowledge Base-driven way to discover work.

DragonPulse mines the user's own documents (Capabilities / Technical / Past
Performance) for distinctive keywords, searches SAM.gov (cache-first,
budget-aware), and ranks the results by how well they match the library. It runs
automatically — showing the top 5 best-fit opportunities and refreshing once
every 24 hours — with a manual refresh always available. Each pick is explained
and one click away from the Detail or Proposal tab.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st

from dragonpulse.api.base import SamApiError, SamAuthError
from dragonpulse.config.logging_config import get_logger
from dragonpulse.config.settings import get_settings
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.models.knowledge import DEFAULT_CATEGORIES
from dragonpulse.processors.recommender import DEFAULT_PICK_CATEGORIES, recommend
from dragonpulse.ui import state

logger = get_logger(__name__)

# Show the 5 best-fit opportunities and refresh at most once per day.
TOP_K = 5
MAX_QUERIES = 8
_REFRESH_INTERVAL_S = 24 * 60 * 60
# Priority Picks uses its own internal window (not the Discover date range).
_PICKS_LOOKBACK_DAYS = 120


def _picks_filters(filters: OpportunityFilters) -> OpportunityFilters:
    """Filters for Priority Picks: inherit NAICS etc. but use an internal window."""
    today = date.today()
    return filters.model_copy(
        update={
            "posted_from": today - timedelta(days=_PICKS_LOOKBACK_DAYS),
            "posted_to": today,
        }
    )


def _time_until_budget_reset() -> "tuple[int, int]":
    """Hours and minutes until the daily budget resets (next local midnight)."""
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    remaining = tomorrow - now
    total_minutes = max(0, int(remaining.total_seconds() // 60))
    return total_minutes // 60, total_minutes % 60


# --------------------------------------------------------------------------- #
# 24-hour refresh bookkeeping (persisted on disk so it survives app restarts)
# --------------------------------------------------------------------------- #
def _last_run_path() -> Path:
    return get_settings().cache_dir / "priority_picks_last_run.txt"


def _read_last_run() -> Optional[float]:
    try:
        return float(_last_run_path().read_text().strip())
    except Exception:  # noqa: BLE001 - missing/corrupt file = "never run"
        return None


def _write_last_run(ts: float) -> None:
    try:
        path = _last_run_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(ts))
    except Exception:  # noqa: BLE001 - best-effort; never block the UI
        logger.debug("Could not persist Priority Picks timestamp", exc_info=True)


def _ago(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86_400:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86_400)}d ago"


def _signature(filters: OpportunityFilters, categories, top_k, max_queries) -> str:
    """Stable key describing the inputs to a recommendation run."""
    return json.dumps(
        {
            "categories": sorted(categories),
            "top_k": top_k,
            "max_queries": max_queries,
            "posted_from": filters.posted_from.isoformat(),
            "posted_to": filters.posted_to.isoformat(),
            "naics": sorted(filters.naics_codes),
            "set_aside": sorted(filters.set_aside_codes),
            "notice_type": sorted(filters.notice_type_codes),
            "department": filters.department_name or "",
        },
        sort_keys=True,
    )


def render_priority_picks(filters: OpportunityFilters) -> None:
    """Render the Priority Picks tab."""
    st.subheader("⭐ Priority Picks")
    st.caption(
        "Your main way to discover opportunities. DragonPulse reads your Knowledge "
        "Base, auto-generates the search keywords that describe your work, searches "
        "SAM.gov, and ranks the best-fit opportunities — no manual keywords needed. "
        "It refreshes automatically every 24 hours."
    )

    kb = state.get_knowledge_base()
    docs = kb.list_documents()
    if not docs:
        st.info(
            "Your Knowledge Base is empty. Add a few past proposals, capability "
            "statements, or performance write-ups in the **📚 Knowledge Base** tab, "
            "then come back here for recommendations.",
            icon="📚",
        )
        return

    # Priority Picks runs over its own internal window (last ~120 days); the
    # Discover date range is intentionally not exposed here.
    filters = _picks_filters(filters)

    available_cats = kb.categories()
    auto_cats = [c for c in DEFAULT_PICK_CATEGORIES if c in available_cats] or list(
        available_cats
    )

    # Power users can broaden/narrow the source categories; defaults need no input.
    with st.expander("Advanced", expanded=False):
        categories = st.multiselect(
            "Learn from these document categories",
            options=sorted(set(available_cats) | set(DEFAULT_CATEGORIES)),
            default=auto_cats,
            help="Defaults to Capabilities, Technical, and Past Performance.",
        )
        st.caption(
            f"Scanning the last **{_PICKS_LOOKBACK_DAYS} days** · NAICS: "
            f"**{', '.join(filters.naics_codes) or 'any'}**. Keywords are generated "
            "automatically from your Knowledge Base."
        )

    selected_cats = categories or auto_cats
    n_in_scope = len(kb.documents_in_categories(selected_cats))
    if n_in_scope == 0:
        st.warning(
            "No documents in the Capabilities / Technical / Past Performance "
            "categories yet. Tag some documents in the Knowledge Base, or pick "
            "different categories under **Advanced**.",
            icon="⚠️",
        )
        return

    sig = _signature(filters, selected_cats, TOP_K, MAX_QUERIES)
    cached = st.session_state.get(state.KEY_PICKS_RESULT)
    last_run = _read_last_run()
    now = time.time()
    stale = last_run is None or (now - last_run) > _REFRESH_INTERVAL_S
    sig_changed = st.session_state.get(state.KEY_PICKS_SIG) != sig

    head = st.columns([3, 1])
    manual = head[1].button(
        "🔄 Refresh now",
        width="stretch",
        help="Re-run immediately and bypass the cache (may spend live requests).",
    )
    if last_run is not None and not stale:
        head[0].caption(
            f"Last updated {_ago(now - last_run)} · auto-refreshes every 24h."
        )
    else:
        head[0].caption("Auto-refreshing from your Knowledge Base…")

    if n_in_scope < 2:
        st.caption(
            "ℹ️ Only 1 document in scope — recommendations improve a lot with more."
        )

    # Auto-run on first view, when the 24h window lapses, when inputs change, or
    # on demand. We record the attempt (sig + timestamp) even on failure so the
    # page never loops re-running a failing query.
    if manual or cached is None or stale or sig_changed:
        result = _run(kb, filters, selected_cats, force=manual)
        st.session_state[state.KEY_PICKS_SIG] = sig
        st.session_state[state.KEY_PICKS_TS] = now
        _write_last_run(now)
        if result is not None:
            st.session_state[state.KEY_PICKS_RESULT] = result
            state.register_opportunities(
                [r.opportunity for r in result.recommendations]
            )
        cached = st.session_state.get(state.KEY_PICKS_RESULT)

    if cached is None:
        st.info("No recommendations yet — try **🔄 Refresh now**.")
        return

    _render_result(cached)


def _run(kb, filters, categories, *, force):
    try:
        with st.spinner("Reading your Knowledge Base and searching SAM.gov…"):
            return recommend(
                kb,
                state.get_opportunities_client(),
                filters,
                categories=categories,
                top_k=TOP_K,
                max_queries=MAX_QUERIES,
                force_refresh=force,
            )
    except SamAuthError as exc:
        st.error(f"Authentication problem: {exc}")
    except SamApiError as exc:
        st.error(f"SAM.gov API error: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors
        logger.exception("Priority Picks run failed")
        st.error(f"Unexpected error: {exc}")
    return None


def _render_result(result) -> None:
    # Daily limit hit and nothing recoverable (cache empty + crawl unavailable):
    # show exactly ONE clean message with the auto-retry countdown.
    if result.budget_exhausted and not result.recommendations:
        hours, minutes = _time_until_budget_reset()
        st.warning(
            "Daily SAM.gov request limit reached. Priority Picks will automatically "
            f"try again in {hours}h {minutes}m.",
            icon="⏳",
        )
        return

    # The keyless public-site search is the primary recall engine and runs every
    # time, so only flag the budget when it was actually hit.
    if result.budget_exhausted and result.crawled:
        st.info(
            "Daily SAM.gov API limit reached — these picks were found on SAM.gov's "
            "public site (no API requests used).",
            icon="🕸️",
        )
    elif result.budget_exhausted:
        st.info(
            "Daily SAM.gov API limit reached — showing the most recent cached results.",
            icon="📁",
        )

    # Suppress secondary warnings when the budget is exhausted to keep one clean
    # message; otherwise surface any informational notes.
    if not result.budget_exhausted:
        for note in result.notes:
            st.warning(note)

    if result.queries:
        with st.expander(
            f"🔑 {len(result.queries)} keyword queries auto-derived from your documents",
            expanded=False,
        ):
            st.write(", ".join(f"`{q}`" for q in result.queries))
            st.caption(
                f"From {result.used_documents} document(s) in: "
                f"{', '.join(result.used_categories)}. "
                f"Considered {result.candidates_considered} candidate opportunities."
            )

    recs = result.recommendations
    if not recs:
        st.info(
            "No recommendations to show yet. Try **🔄 Refresh now**, or add more "
            "Capabilities / Technical documents under **Advanced**."
        )
        return

    st.markdown(f"**Top {len(recs)} matches for your firm**")
    for rank, rec in enumerate(recs, start=1):
        _render_card(rank, rec)


def _render_card(rank: int, rec) -> None:
    opp = rec.opportunity
    days = opp.days_until_deadline()
    with st.container(border=True):
        head = st.columns([6, 1])
        head[0].markdown(f"**{rank}. {opp.title or '(untitled opportunity)'}**")
        head[1].metric("Fit", f"{rec.score:.2f}")

        meta = st.columns(4)
        meta[0].caption(f"🏛️ {opp.agency or '—'}")
        meta[1].caption(f"#️⃣ NAICS {opp.naics_code or '—'}")
        if days is None:
            meta[2].caption("📅 No deadline")
        elif days < 0:
            meta[2].caption(f"⛔ Closed {abs(days)}d ago")
        else:
            meta[2].caption(f"⏳ {days}d left")
        meta[3].caption(f"🏷️ {opp.set_aside_description or opp.set_aside_code or 'No set-aside'}")

        st.markdown(f"**Why this matches:** {rec.why}")

        if rec.evidence:
            with st.expander("Evidence from your documents", expanded=False):
                for ev in rec.evidence:
                    st.caption(f"📄 {ev.citation} · score {ev.score:.3f}")
                    preview = ev.chunk.text.strip()
                    st.text(preview[:280] + ("…" if len(preview) > 280 else ""))

        actions = st.columns([1, 1, 2])
        if actions[0].button("📄 Open in Detail", key=f"pick_detail_{opp.notice_id}",
                             width="stretch"):
            state.register_opportunities([opp])
            state.select_opportunity(opp.notice_id)
            st.toast("Opened in Detail — switch to the 📄 Detail tab.", icon="📄")
        if actions[1].button("📝 Draft proposal", key=f"pick_prop_{opp.notice_id}",
                             width="stretch"):
            state.register_opportunities([opp])
            state.select_opportunity(opp.notice_id)
            st.toast("Selected for drafting — switch to the 📝 Proposals tab.", icon="📝")
        actions[2].markdown(f"[Open on SAM.gov ↗]({opp.sam_url})")
