"""Priority Picks — a smart opportunity recommender driven by the Knowledge Base.

The pipeline is deliberately local-first and budget-aware:

1. **Generate queries** from the user's own documents — distinctive domain
   phrases plus broad single-word capability terms — skipping boilerplate,
   locations, and customer names.
2. **Search SAM.gov, recall-first.** A keyless **full-text** crawl of the public
   site runs for every query (free, matches the whole notice, not just the
   title) and is the main candidate source. The keyed API enriches it: the disk
   cache is always consulted for free, plus a few live requests while budget
   remains. If results are thin, the NAICS filter is relaxed to broaden.
3. **Rank** the de-duplicated candidates by semantic similarity to the library
   (reusing the Knowledge Base's embedding backend), with small bonuses for
   matching multiple queries and the firm's NAICS, and attach a clear "why this
   matches" explanation grounded in the user's actual documents.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import requests

from dragonpulse.api.base import SamApiError
from dragonpulse.api.opportunities import OpportunitiesClient
from dragonpulse.cache.request_budget import RequestBudgetExceeded
from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.models.knowledge import RetrievedChunk
from dragonpulse.models.opportunity import Opportunity
from dragonpulse.processors.knowledge_base import KnowledgeBase
from dragonpulse.processors.sam_scrape import search_opportunities_via_frontend

logger = get_logger(__name__)

# Categories that best describe *what the company does* (vs. pricing/certs).
DEFAULT_PICK_CATEGORIES = ["Technical", "Capabilities", "Past Performance"]

# Generic words that show up in every proposal and make poor search keywords.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "will", "shall", "are",
    "was", "were", "has", "have", "had", "not", "all", "any", "our", "their",
    "you", "your", "they", "them", "its", "his", "her", "she", "him",
    "proposal", "proposals", "government", "contract", "contractor", "contracts",
    "offeror", "offerors", "solicitation", "section", "sections", "page", "pages",
    "volume", "price", "pricing", "cost", "costs", "data", "company", "inc",
    "llc", "corporation", "corp", "may", "must", "should", "would",
    "include", "included", "including", "provided", "provides",
    "requirement", "requirements", "experience", "project",
    "projects", "year",
    "years", "date", "number", "table", "figure", "appendix", "attachment",
    "reference", "references", "task", "tasks", "scope", "statement", "request",
    "following", "described", "based", "ensure", "per", "via", "use", "used",
    "using", "within", "into", "upon", "also", "each", "between", "under",
    "over", "such", "these", "those", "which", "where", "when", "what", "who",
    "able", "ability", "quality", "schedule", "deliver", "delivery",
    "us", "u.s", "usa", "federal", "agency", "department", "office", "program",
    "provide", "providing", "team", "personnel", "staff", "approach",
    "plan", "phase", "effort", "current", "new", "existing", "various", "additional",
    "overall", "total", "general", "specific", "applicable", "appropriate",
    "located", "location", "site", "area", "areas", "region", "facility", "facilities",
}

# Geographic / customer / proper-noun words to keep OUT of queries — these are
# project specifics or buyers, not the firm's capabilities. (Item 3.)
_LOCATION_ORG_STOPWORDS = {
    # cardinal / geographic
    "north", "south", "east", "west", "central", "upper", "lower",
    "county", "city", "town", "river", "lake", "creek", "dam", "valley",
    "mountain", "mountains", "island", "bay", "harbor", "port", "fort",
    "base", "district", "regional", "national",
    # military / customer org words
    "army", "navy", "marine", "marines", "force", "guard", "coast",
    "corps", "engineers", "command", "battalion", "brigade", "wing",
    "usace", "usarmy", "dod", "gsa", "va", "nasa", "doe", "dhs",
    # US state names (and DC)
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "hampshire",
    "jersey", "mexico", "york", "carolina", "dakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhode", "tennessee", "texas", "utah",
    "vermont", "virginia", "washington", "wisconsin", "wyoming", "columbia",
}

# Terms that signal a genuine capability / service line. Queries containing
# these are boosted so recommendations track *what the firm does*. (Item 3.)
_CAPABILITY_HINTS = {
    "construction", "engineering", "installation", "maintenance", "design",
    "integration", "testing", "commissioning", "fabrication", "inspection",
    "modernization", "upgrade", "upgrades", "replacement", "repair", "rehabilitation",
    "operations", "electrical", "mechanical", "civil", "structural", "power",
    "energy", "transmission", "distribution", "substation", "substations",
    "generation", "generator", "generators", "exciter", "excitation", "turbine",
    "switchgear", "relay", "protection", "controls", "control", "scada",
    "automation", "hydroelectric", "hydropower", "infrastructure", "cybersecurity",
    "network", "networking", "software", "hardware", "staffing", "logistics",
    "training", "consulting", "environmental", "welding", "hvac",
    "telecommunications", "fiber", "cabling", "instrumentation", "calibration",
    "procurement", "supply", "lines", "line", "grid", "voltage", "breaker",
    "transformer", "transformers", "motor", "motors", "pump", "pumps", "piping",
}

# A short token used to bridge bigrams that are not themselves stopwords.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")

# Recall tuning. The keyless full-text crawl is the primary source (free + much
# higher recall than the keyed *title-only* search), so we lean on it heavily and
# only spend a few live keyed requests as a bonus.
DEFAULT_MAX_LIVE_KEYED = 3       # cap live keyed requests per run (basic = 9/day)
MIN_CANDIDATES = 8               # below this we relax the NAICS filter to broaden
_NAICS_MATCH_BONUS = 0.05        # small fit bonus when the opp is in the firm NAICS


def _is_query_token(tok: str) -> bool:
    return (
        tok not in _STOPWORDS
        and tok not in _LOCATION_ORG_STOPWORDS
        and len(tok) >= 3
    )


@dataclass
class Recommendation:
    """A single recommended opportunity with its rationale."""

    opportunity: Opportunity
    score: float
    matched_queries: List[str] = field(default_factory=list)
    evidence: List[RetrievedChunk] = field(default_factory=list)
    why: str = ""


@dataclass
class RecommendationResult:
    """The full output of a Priority Picks run."""

    recommendations: List[Recommendation] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    used_documents: int = 0
    used_categories: List[str] = field(default_factory=list)
    candidates_considered: int = 0
    notes: List[str] = field(default_factory=list)
    # Daily keyed-API budget was hit during this run.
    budget_exhausted: bool = False
    # At least one result came from the keyless public-site crawl fallback.
    crawled: bool = False


# --------------------------------------------------------------------------- #
# Query generation
# --------------------------------------------------------------------------- #
def _has_capability(tokens) -> bool:
    return any(t in _CAPABILITY_HINTS for t in tokens)


def generate_queries(texts: List[str], *, max_queries: int = 6) -> List[str]:
    """Derive **broad capability/service** queries from per-document ``texts``.

    ``texts`` should be one entry per document. The generator favors terms that
    recur *across documents* (the firm's general capabilities) and that look like
    service lines (e.g. "engineering services", "power line construction",
    "substation"), while filtering out one-off project names and locations
    (e.g. "libby dam", "army corps"). This makes Priority Picks track *what the
    firm does* rather than a single past project.
    """
    if not texts:
        return []

    n_docs = len(texts)
    uni_df: Dict[str, int] = defaultdict(int)   # document frequency
    uni_tf: Dict[str, int] = defaultdict(int)   # total term frequency
    bi_df: Dict[str, int] = defaultdict(int)
    bi_tf: Dict[str, int] = defaultdict(int)

    for text in texts:
        tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
        content = [t for t in tokens if _is_query_token(t)]
        seen_uni: set = set()
        seen_bi: set = set()
        for tok in content:
            uni_tf[tok] += 1
            if tok not in seen_uni:
                uni_df[tok] += 1
                seen_uni.add(tok)
        for a, b in zip(content, content[1:]):
            if a == b:
                continue
            phrase = f"{a} {b}"
            bi_tf[phrase] += 1
            if phrase not in seen_bi:
                bi_df[phrase] += 1
                seen_bi.add(phrase)

    # With several documents, require cross-document recurrence (broad terms).
    # With only one or two, fall back to in-document frequency + capability hints.
    df_floor = 2 if n_docs >= 3 else 1

    scored: List[tuple] = []
    for phrase, df in bi_df.items():
        toks = set(phrase.split())
        tf = bi_tf[phrase]
        cap = _has_capability(toks)
        if df < df_floor and not (cap and tf >= 3):
            continue
        score = df * 4.0 + tf * 0.4 + (3.0 if cap else 0.0)
        scored.append((score, phrase, toks))
    for term, df in uni_df.items():
        if len(term) < 4:
            continue
        tf = uni_tf[term]
        cap = term in _CAPABILITY_HINTS
        if df < df_floor and not (cap and tf >= 3):
            continue
        # Unigrams are broader/noisier than phrases; weight a touch lower.
        score = df * 3.0 + tf * 0.3 + (2.5 if cap else 0.0)
        scored.append((score, term, {term}))

    scored.sort(key=lambda s: s[0], reverse=True)

    chosen: List[str] = []
    covered_tokens: set = set()
    for _score, phrase, tokens in scored:
        # Skip phrases whose tokens are already represented (reduce redundancy).
        if tokens <= covered_tokens:
            continue
        chosen.append(phrase)
        covered_tokens |= tokens
        if len(chosen) >= max_queries:
            break
    return chosen


def broad_capability_terms(texts: List[str], *, max_terms: int = 3) -> List[str]:
    """Top single-word **capability** terms across the documents.

    These broad, high-recall keywords (e.g. ``substation``, ``engineering``,
    ``power``) complement the more specific phrase queries: they cast a wider net
    in the keyless full-text crawl so Priority Picks surfaces opportunities even
    when no exact phrase matches.
    """
    if not texts:
        return []
    df: Dict[str, int] = defaultdict(int)
    tf: Dict[str, int] = defaultdict(int)
    for text in texts:
        seen: set = set()
        for tok in (t.lower() for t in _TOKEN_RE.findall(text)):
            if tok not in _CAPABILITY_HINTS:
                continue
            tf[tok] += 1
            if tok not in seen:
                df[tok] += 1
                seen.add(tok)
    ranked = sorted(df.keys(), key=lambda t: (df[t], tf[t]), reverse=True)
    return ranked[:max_terms]


# --------------------------------------------------------------------------- #
# Ranking helpers
# --------------------------------------------------------------------------- #
def _opportunity_text(opp: Opportunity) -> str:
    """A compact, embeddable representation of an opportunity."""
    parts = [
        opp.title or "",
        opp.naics_code or "",
        opp.classification_code or "",
        opp.set_aside_description or "",
        opp.office or "",
        opp.agency or "",
    ]
    return " ".join(p for p in parts if p).strip()


def _build_why(
    opp: Opportunity,
    rec_evidence: List[RetrievedChunk],
    matched_queries: List[str],
    firm_naics: set,
) -> str:
    """A short, plain-language, document-grounded explanation of the match."""
    bits: List[str] = []
    if rec_evidence:
        top = rec_evidence[0].chunk
        area = (top.doc_type or top.category or "").strip().lower()
        lead = f"Matches your {area}" if area else "Matches your work"
        bits.append(f"{lead} in “{top.doc_name}”")
        others = sorted(
            {e.chunk.doc_name for e in rec_evidence[1:] if e.chunk.doc_name != top.doc_name}
        )
        if others:
            bits[-1] += f" (also {', '.join(others)})"
    if matched_queries:
        quoted = ", ".join(f"“{q}”" for q in matched_queries[:3])
        bits.append(f"surfaced by {quoted}")
    if opp.naics_code and opp.naics_code in firm_naics:
        bits.append(f"in your NAICS {opp.naics_code}")
    if not bits:
        return "Aligns with your Knowledge Base."
    return "; ".join(bits) + "."


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def recommend(
    kb: KnowledgeBase,
    client: OpportunitiesClient,
    filters: OpportunityFilters,
    *,
    categories: Optional[List[str]] = None,
    top_k: int = 10,
    max_queries: int = 6,
    per_query_limit: int = 25,
    force_refresh: bool = False,
    crawl_fn: Optional[Callable] = None,
    max_live_keyed: int = DEFAULT_MAX_LIVE_KEYED,
) -> RecommendationResult:
    """Recommend SAM.gov opportunities ranked by fit to the Knowledge Base.

    Recall-first strategy (the previous title-only keyed search frequently
    returned nothing):

    1. **Keyless full-text crawl** of SAM.gov's public site (``crawl_fn``,
       defaulting to :func:`search_opportunities_via_frontend`) runs for *every*
       query. It is free (no keyed-API budget) and matches the full notice text,
       not just the title — this is the main source of candidates.
    2. **Keyed API** enriches results: the disk cache is always consulted (free),
       plus up to ``max_live_keyed`` live requests as a bonus while budget remains.
    3. If candidates are still thin, the **NAICS filter is relaxed** to broaden.

    Results are ranked by semantic fit to the Knowledge Base (with small bonuses
    for matching multiple queries and the firm's NAICS). Injectable for testing.
    """
    crawl = crawl_fn or search_opportunities_via_frontend
    cats = categories or DEFAULT_PICK_CATEGORIES
    docs = kb.documents_in_categories(cats)
    result = RecommendationResult(used_categories=cats, used_documents=len(docs))

    # Per-document texts so query generation can favor broad, cross-document
    # capability terms over one-off project specifics.
    doc_texts = kb.category_document_texts(cats)
    if not doc_texts:
        result.notes.append(
            "No documents in the selected categories — add some, or broaden the "
            "category filter."
        )
        return result

    # Specific multi-word phrases first, then broad single-word capability terms
    # for high-recall full-text matching.
    phrase_queries = generate_queries(doc_texts, max_queries=max_queries)
    broad_terms = broad_capability_terms(doc_texts, max_terms=3)
    search_terms: List[str] = list(phrase_queries)
    for term in broad_terms:
        if term not in search_terms:
            search_terms.append(term)
    result.queries = search_terms
    if not search_terms:
        result.notes.append(
            "Could not derive distinctive keywords from these documents."
        )
        return result

    seen: Dict[str, Opportunity] = {}
    matched: Dict[str, set] = defaultdict(set)
    budget_hit = False
    crawl_any = False
    live_keyed = 0
    session = requests.Session()

    def _absorb(opps, query: str) -> None:
        for opp in opps:
            seen.setdefault(opp.notice_id, opp)
            matched[opp.notice_id].add(query)

    def _safe_crawl(query: str, naics: Optional[List[str]]) -> List[Opportunity]:
        try:
            return list(
                crawl(
                    query,
                    naics_codes=naics,
                    posted_from=filters.posted_from,
                    posted_to=filters.posted_to,
                    limit=per_query_limit,
                    session=session,
                )
            )
        except Exception:  # noqa: BLE001 - crawl is strictly best-effort
            logger.info("Crawl failed for %r", query, exc_info=True)
            return []

    for q in search_terms:
        page = filters.model_copy(
            update={"keyword": q, "limit": per_query_limit, "offset": 0}
        )

        # (1) Keyed API — a few live requests as a bonus while budget remains.
        if not budget_hit and live_keyed < max_live_keyed:
            try:
                res = client.search(page, force_refresh=force_refresh, allow_network=True)
                live_keyed += 1
                _absorb(res.opportunities, q)
            except RequestBudgetExceeded:
                budget_hit = True
            except SamApiError as exc:
                logger.info("Keyed query %r error: %s", q, exc)

        # (2) Keyed cache — always free, never spends budget.
        try:
            _absorb(client.search(page, allow_network=False).opportunities, q)
        except SamApiError:
            pass

        # (3) Keyless full-text crawl — primary recall. NAICS first; relax if empty.
        crawl_opps = _safe_crawl(q, list(filters.naics_codes) or None)
        if not crawl_opps and filters.naics_codes:
            crawl_opps = _safe_crawl(q, None)
        if crawl_opps:
            crawl_any = True
            _absorb(crawl_opps, q)

    # (4) Still thin? Broaden by dropping the NAICS filter on the broad terms.
    if len(seen) < min(MIN_CANDIDATES, top_k * 2):
        for q in (broad_terms or search_terms[:3]):
            crawl_opps = _safe_crawl(q, None)
            if crawl_opps:
                crawl_any = True
                _absorb(crawl_opps, q)

    result.budget_exhausted = budget_hit
    result.crawled = crawl_any
    result.candidates_considered = len(seen)
    if not seen:
        if not budget_hit:
            result.notes.append(
                "No matching opportunities found for your capabilities in the last "
                "few months. This is unusual — try the 🔄 Refresh, or add more "
                "Capabilities / Technical documents to sharpen the keywords."
            )
        # When the budget is exhausted and nothing was recoverable, the view
        # renders a single clean countdown message (no extra notes here).
        return result

    firm_naics = set(filters.naics_codes or [])
    recs: List[Recommendation] = []
    for notice_id, opp in seen.items():
        score, evidence = kb.relevance(_opportunity_text(opp), categories=cats, top_n=3)
        query_bonus = 0.03 * (len(matched[notice_id]) - 1)
        naics_bonus = (
            _NAICS_MATCH_BONUS if (opp.naics_code and opp.naics_code in firm_naics) else 0.0
        )
        recs.append(
            Recommendation(
                opportunity=opp,
                score=score + query_bonus + naics_bonus,
                matched_queries=sorted(matched[notice_id]),
                evidence=evidence,
                why=_build_why(opp, evidence, sorted(matched[notice_id]), firm_naics),
            )
        )

    # Highest fit first; tie-break by soonest (non-negative) deadline.
    def _deadline_key(rec: Recommendation) -> float:
        days = rec.opportunity.days_until_deadline()
        if days is None:
            return 1e9
        return days if days >= 0 else 1e8 + abs(days)

    recs.sort(key=lambda r: (-r.score, _deadline_key(r)))
    result.recommendations = recs[:top_k]
    return result
