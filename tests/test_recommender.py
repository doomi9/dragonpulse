"""Tests for Priority Picks: query generation, ranking, and budget handling."""

from __future__ import annotations

from dragonpulse.cache.request_budget import RequestBudgetExceeded
from dragonpulse.config.settings import KeyTier, Settings
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.models.opportunity import OpportunitySearchResult
from dragonpulse.processors.embeddings import HashingEmbedding
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.recommender import (
    broad_capability_terms,
    generate_queries,
    recommend,
)


def _no_crawl(*_args, **_kwargs):
    """Stub crawl so tests never touch the network."""
    return []


def _settings(tmp_path) -> Settings:
    return Settings(
        sam_api_key_basic="K",
        api_key_tier=KeyTier.BASIC,
        data_dir=tmp_path,
        rag_embedding_backend="hashing",
        rag_chunk_chars=300,
        rag_chunk_overlap=50,
        llm_enabled=False,
    )


def _kb(settings) -> KnowledgeBase:
    kb = KnowledgeBase(settings=settings, backend=HashingEmbedding(256))
    kb.add_document(
        "Hydro Capabilities.txt",
        "Dragon Infrastructure designs and installs generator exciter replacement "
        "systems for hydroelectric dams. We perform exciter replacement, switchgear "
        "upgrades, and turbine controls for federal hydroelectric facilities.",
        category="Capabilities",
    )
    kb.add_document(
        "Dam Past Performance.txt",
        "We completed an exciter replacement and switchgear modernization for a "
        "hydroelectric dam operated by the Army Corps of Engineers, on schedule.",
        category="Past Performance",
    )
    kb.add_document(
        "Rates.txt",
        "Labor rates and pricing tables for fiscal year billing.",
        category="Pricing",
    )
    return kb


def _opp(notice_id, title, naics="221111"):
    payload = {
        "totalRecords": 1,
        "opportunitiesData": [
            {
                "noticeId": notice_id,
                "title": title,
                "naicsCode": naics,
                "fullParentPathName": "DEPT OF DEFENSE.ARMY CORPS OF ENGINEERS",
                "postedDate": "2026-06-01",
                "responseDeadLine": "2026-12-31T17:00:00-04:00",
            }
        ],
    }
    return OpportunitySearchResult.model_validate(payload).opportunities[0]


class _FakeClient:
    """Returns canned opportunities per keyword; records calls + can raise budget."""

    def __init__(self, by_keyword, *, raise_after=None):
        self.by_keyword = by_keyword
        self.calls = []
        self.network_calls = 0
        self.raise_after = raise_after

    def search(self, filters, *, force_refresh=False, allow_network=True):
        self.calls.append((filters.keyword, allow_network))
        if allow_network:
            if self.raise_after is not None and self.network_calls >= self.raise_after:
                raise RequestBudgetExceeded("budget reached")
            self.network_calls += 1
        opps = self.by_keyword.get(filters.keyword, [])
        return OpportunitySearchResult(opportunities=opps)


# --------------------------------------------------------------------------- #
# Query generation
# --------------------------------------------------------------------------- #
def test_generate_queries_prefers_domain_terms():
    texts = [
        "exciter replacement for hydroelectric dam generator exciter replacement",
        "switchgear upgrade and turbine controls for the hydroelectric dam",
    ] * 3
    queries = generate_queries(texts, max_queries=4)
    assert queries
    joined = " ".join(queries).lower()
    # Domain vocabulary surfaces; generic boilerplate does not.
    assert "exciter" in joined or "hydroelectric" in joined
    assert "the" not in queries
    assert "proposal" not in joined


def test_generate_queries_empty_input():
    assert generate_queries([], max_queries=5) == []


def test_generate_queries_respects_max():
    texts = ["alpha bravo charlie delta echo foxtrot golf hotel " * 5]
    assert len(generate_queries(texts, max_queries=3)) <= 3


# --------------------------------------------------------------------------- #
# Recommend pipeline
# --------------------------------------------------------------------------- #
def test_recommend_ranks_relevant_opportunity_first(tmp_path):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    relevant = _opp("rel1", "Hydroelectric Dam Exciter Replacement and Switchgear")
    noise = _opp("noise1", "Office Janitorial Services", naics="561720")
    # Every query returns both; ranking must float the relevant one to the top.
    client = _FakeClient({q: [relevant, noise] for q in
                          generate_queries(kb.category_texts(["Capabilities",
                                                              "Past Performance"]),
                                           max_queries=5)})
    filters = OpportunityFilters(naics_codes=[])
    result = recommend(
        kb, client, filters,
        categories=["Capabilities", "Past Performance"], top_k=10,
        crawl_fn=_no_crawl,
    )
    assert result.recommendations
    assert result.recommendations[0].opportunity.notice_id == "rel1"
    top = result.recommendations[0]
    assert top.why
    assert top.matched_queries
    assert top.evidence  # grounded in the user's documents


def test_recommend_dedupes_across_queries(tmp_path):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    opp = _opp("dup1", "Hydroelectric Exciter Replacement")
    client = _FakeClient({q: [opp] for q in
                          generate_queries(kb.category_texts(None), max_queries=5)})
    result = recommend(
        kb, client, OpportunityFilters(), categories=None, top_k=10,
        crawl_fn=_no_crawl,
    )
    ids = [r.opportunity.notice_id for r in result.recommendations]
    assert ids.count("dup1") == 1


def test_recommend_empty_category_returns_note(tmp_path):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    client = _FakeClient({})
    result = recommend(
        kb, client, OpportunityFilters(), categories=["Management"], top_k=10,
        crawl_fn=_no_crawl,
    )
    assert result.recommendations == []
    assert any("No documents" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# Broad capability terms + crawl-primary recall
# --------------------------------------------------------------------------- #
def test_broad_capability_terms_surfaces_single_words():
    texts = [
        "substation upgrade and power transmission engineering",
        "power substation engineering and distribution",
    ]
    terms = broad_capability_terms(texts, max_terms=3)
    assert terms
    # Only genuine capability words, no boilerplate or connectors.
    assert all(t in {"substation", "power", "transmission", "engineering",
                     "distribution"} for t in terms)
    assert "and" not in terms


def test_recommend_uses_crawl_even_when_budget_available(tmp_path):
    """The keyless full-text crawl is the primary recall source, not just a
    budget fallback — it must run (and surface picks) even when the keyed API is
    available but returns nothing for the title-only search."""
    settings = _settings(tmp_path)
    kb = _kb(settings)
    crawled = _opp("c1", "Hydroelectric Dam Exciter Replacement")
    client = _FakeClient({})  # keyed API (title search) finds nothing
    result = recommend(
        kb, client, OpportunityFilters(naics_codes=[]),
        categories=["Capabilities", "Past Performance"], top_k=10,
        crawl_fn=lambda *a, **k: [crawled],
    )
    assert result.budget_exhausted is False
    assert result.crawled is True
    assert any(r.opportunity.notice_id == "c1" for r in result.recommendations)


def test_recommend_relaxes_naics_when_results_thin(tmp_path):
    """When a NAICS-filtered crawl is empty, it retries without the NAICS filter."""
    settings = _settings(tmp_path)
    kb = _kb(settings)
    relaxed = _opp("r9", "Hydroelectric Switchgear Modernization", naics="999999")
    seen_naics = []

    def crawl(query, *, naics_codes=None, **kwargs):
        seen_naics.append(naics_codes)
        return [] if naics_codes else [relaxed]

    result = recommend(
        kb, _FakeClient({}), OpportunityFilters(naics_codes=["237130"]),
        categories=["Capabilities", "Past Performance"], top_k=10,
        crawl_fn=crawl,
    )
    assert any(n is None for n in seen_naics)  # relaxed at least once
    assert any(r.opportunity.notice_id == "r9" for r in result.recommendations)


def test_recommend_handles_budget_exhaustion(tmp_path):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    opp = _opp("b1", "Hydroelectric Exciter Replacement")
    queries = generate_queries(kb.category_document_texts(None), max_queries=6)
    # Allow only the first network call; the rest hit the budget.
    client = _FakeClient({q: [opp] for q in queries}, raise_after=1)
    # Stub the crawl so the test never touches the network.
    result = recommend(
        kb, client, OpportunityFilters(), categories=None, top_k=10,
        crawl_fn=lambda *a, **k: [],
    )
    assert result.budget_exhausted is True
    # Still produced a recommendation from the first (successful) query / cache.
    assert any(r.opportunity.notice_id == "b1" for r in result.recommendations)


def test_recommend_crawl_fallback_when_budget_exhausted(tmp_path):
    settings = _settings(tmp_path)
    kb = _kb(settings)
    # Keyed API yields nothing and immediately hits the budget; the keyless crawl
    # supplies a result instead — flagged as crawled, no API spend assumed.
    crawled_opp = _opp("crawl1", "Hydroelectric Switchgear Modernization")
    crawl_calls = {"n": 0}

    def fake_crawl(query, **kwargs):
        crawl_calls["n"] += 1
        return [crawled_opp]

    client = _FakeClient({}, raise_after=0)  # first network call raises budget
    result = recommend(
        kb, client, OpportunityFilters(), categories=None, top_k=10,
        crawl_fn=fake_crawl,
    )
    assert result.budget_exhausted is True
    assert result.crawled is True
    assert crawl_calls["n"] >= 1
    assert any(r.opportunity.notice_id == "crawl1" for r in result.recommendations)
