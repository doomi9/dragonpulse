"""Shared Streamlit state and service wiring.

Centralizes construction of the API clients (cached as Streamlit resources) and
provides typed helpers for reading/writing ``st.session_state`` so the views
stay clean.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from dragonpulse.api.awards import AwardsClient
from dragonpulse.api.opportunities import OpportunitiesClient
from dragonpulse.config.settings import Settings, get_settings
from dragonpulse.models.opportunity import Opportunity, OpportunitySearchResult
from dragonpulse.processors.draft_store import DraftStore
from dragonpulse.processors.knowledge_base import KnowledgeBase

# Session-state keys (centralized to avoid typos).
KEY_RESULT = "search_result"
KEY_SELECTED = "selected_notice_id"
KEY_PROFILE = "company_profile"
KEY_LAST_ERROR = "last_error"
KEY_PRICING = "pricing_analysis"
KEY_KB_HITS = "kb_search_hits"
KEY_KB_UPLOAD_RESULT = "kb_upload_result"  # dict surviving the post-index rerun
KEY_OPP_POOL = "opportunity_pool"  # notice_id -> Opportunity (search + recs + detail)
KEY_PICKS_RESULT = "priority_picks_result"
KEY_PICKS_SIG = "priority_picks_signature"
KEY_PICKS_TS = "priority_picks_last_run_ts"  # epoch seconds of last auto/manual run
KEY_PROPOSAL_GEN = "proposal_generator"
KEY_PROPOSAL_DRAFT = "proposal_draft"
KEY_PROPOSAL_ATTACH = "proposal_attachments_loaded"
KEY_PROPOSAL_AUTOLOAD = "proposal_autoload_attempted"  # set[notice_id]
KEY_PROPOSAL_LOAD_MSG = "proposal_load_message"  # (level, text)
KEY_PROPOSAL_LOADED_FROM = "proposal_loaded_from_saved"  # draft_id current draft came from
KEY_MANUAL_SOLICITATION = "manual_solicitation_text"  # notice_id -> [(filename, text), ...]


@st.cache_resource(show_spinner=False)
def get_opportunities_client() -> OpportunitiesClient:
    """Return a process-wide OpportunitiesClient (cached across reruns)."""
    return OpportunitiesClient()


@st.cache_resource(show_spinner=False)
def get_awards_client() -> AwardsClient:
    """Return a process-wide AwardsClient (cached across reruns)."""
    return AwardsClient()


@st.cache_resource(show_spinner=False)
def get_knowledge_base() -> KnowledgeBase:
    """Return a process-wide KnowledgeBase (cached across reruns)."""
    return KnowledgeBase()


@st.cache_resource(show_spinner=False)
def get_draft_store() -> DraftStore:
    """Return a process-wide DraftStore (cached across reruns)."""
    return DraftStore()


def settings() -> Settings:
    return get_settings()


def set_result(result: OpportunitySearchResult) -> None:
    st.session_state[KEY_RESULT] = result
    st.session_state[KEY_SELECTED] = None
    register_opportunities(result.opportunities)


def get_result() -> Optional[OpportunitySearchResult]:
    return st.session_state.get(KEY_RESULT)


def get_opportunities() -> List[Opportunity]:
    result = get_result()
    return result.opportunities if result else []


def _opportunity_pool() -> dict:
    pool = st.session_state.get(KEY_OPP_POOL)
    if pool is None:
        pool = {}
        st.session_state[KEY_OPP_POOL] = pool
    return pool


def register_opportunities(opportunities: List[Opportunity]) -> None:
    """Add opportunities to a session-wide pool so they resolve across tabs.

    Search results, recommendations, and detail fetches all register here; this
    lets the Detail and Proposal tabs open any opportunity the user has seen —
    not only those in the latest search result.
    """
    pool = _opportunity_pool()
    for opp in opportunities:
        pool[opp.notice_id] = opp


def all_opportunities() -> List[Opportunity]:
    """Latest search results first, then any other pooled opportunities."""
    result_opps = get_opportunities()
    seen = {o.notice_id for o in result_opps}
    extras = [o for nid, o in _opportunity_pool().items() if nid not in seen]
    return result_opps + extras


def select_opportunity(notice_id: Optional[str]) -> None:
    st.session_state[KEY_SELECTED] = notice_id


def get_selected() -> Optional[Opportunity]:
    notice_id = st.session_state.get(KEY_SELECTED)
    if not notice_id:
        return None
    for opp in all_opportunities():
        if opp.notice_id == notice_id:
            return opp
    return None
