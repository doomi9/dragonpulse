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
from dragonpulse.processors.knowledge_base import KnowledgeBase

# Session-state keys (centralized to avoid typos).
KEY_RESULT = "search_result"
KEY_SELECTED = "selected_notice_id"
KEY_PROFILE = "company_profile"
KEY_LAST_ERROR = "last_error"
KEY_PRICING = "pricing_analysis"
KEY_KB_HITS = "kb_search_hits"
KEY_PROPOSAL_GEN = "proposal_generator"
KEY_PROPOSAL_DRAFT = "proposal_draft"
KEY_PROPOSAL_ATTACH = "proposal_attachments_loaded"


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


def settings() -> Settings:
    return get_settings()


def set_result(result: OpportunitySearchResult) -> None:
    st.session_state[KEY_RESULT] = result
    st.session_state[KEY_SELECTED] = None


def get_result() -> Optional[OpportunitySearchResult]:
    return st.session_state.get(KEY_RESULT)


def get_opportunities() -> List[Opportunity]:
    result = get_result()
    return result.opportunities if result else []


def select_opportunity(notice_id: Optional[str]) -> None:
    st.session_state[KEY_SELECTED] = notice_id


def get_selected() -> Optional[Opportunity]:
    notice_id = st.session_state.get(KEY_SELECTED)
    if not notice_id:
        return None
    for opp in get_opportunities():
        if opp.notice_id == notice_id:
            return opp
    return None
