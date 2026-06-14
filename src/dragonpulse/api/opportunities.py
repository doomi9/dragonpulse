"""Client for the SAM.gov Opportunities v2 API.

Endpoint
--------
``GET https://api.sam.gov/opportunities/v2/search``

Key parameters (see :meth:`OpportunityFilters.to_query_params`):
- ``postedFrom`` / ``postedTo`` (MM/dd/yyyy, both required, max 1-year span)
- ``title``, ``ncode`` (NAICS), ``ptype`` (notice type), ``typeOfSetAside``
- ``deptname``, ``limit`` (<=1000), ``offset``
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from dragonpulse.api.base import SamClient
from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.filters import OpportunityFilters
from dragonpulse.models.opportunity import Opportunity, OpportunitySearchResult

logger = get_logger(__name__)

SEARCH_PATH = "/opportunities/v2/search"


class OpportunitiesClient:
    """High-level, model-returning client for opportunity search."""

    def __init__(self, sam_client: Optional[SamClient] = None) -> None:
        self.client = sam_client or SamClient()

    def search(
        self,
        filters: OpportunityFilters,
        *,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False,
        allow_network: bool = True,
    ) -> OpportunitySearchResult:
        """Run a single search page and return parsed results.

        The result records whether it was served from cache and when it was
        fetched, for transparent display in the UI.
        """
        params = filters.to_query_params()
        payload, from_cache = self.client.get_json(
            SEARCH_PATH,
            params,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
            allow_network=allow_network,
        )
        result = OpportunitySearchResult.model_validate(payload)
        result.from_cache = from_cache
        result.fetched_at = datetime.now()
        logger.info(
            "Opportunities search: %d of %d total (cache=%s)",
            result.count,
            result.total_records,
            from_cache,
        )
        return result

    def get_by_notice_id(
        self,
        notice_id: str,
        *,
        posted_from: Optional[str] = None,
        posted_to: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[Opportunity]:
        """Fetch a single opportunity by its notice ID.

        The v2 API supports a ``noticeid`` parameter but still requires the
        posted date range. Callers should pass a range that brackets the
        notice's posted date; otherwise a wide default is used.
        """
        params = {"noticeid": notice_id, "limit": 1, "offset": 0}
        if posted_from:
            params["postedFrom"] = posted_from
        if posted_to:
            params["postedTo"] = posted_to
        payload, from_cache = self.client.get_json(
            SEARCH_PATH, params, ttl_seconds=ttl_seconds
        )
        result = OpportunitySearchResult.model_validate(payload)
        result.from_cache = from_cache
        return result.opportunities[0] if result.opportunities else None

    def iter_all(
        self,
        filters: OpportunityFilters,
        *,
        max_records: int = 200,
        page_size: int = 100,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Iterator[Opportunity]:
        """Paginate through results up to ``max_records``.

        Each page is a separate (cached) request. With the basic key, prefer a
        small ``max_records`` to stay within the daily budget. Cached pages do
        not count against the budget.
        """
        offset = 0
        yielded = 0
        page_filters = filters.model_copy(update={"limit": min(page_size, 1000), "offset": 0})
        while yielded < max_records:
            page_filters = page_filters.model_copy(update={"offset": offset})
            result = self.search(
                page_filters,
                ttl_seconds=ttl_seconds,
                force_refresh=force_refresh,
            )
            if not result.opportunities:
                break
            for opp in result.opportunities:
                yield opp
                yielded += 1
                if yielded >= max_records:
                    return
            offset += result.count
            if not result.has_more:
                break
