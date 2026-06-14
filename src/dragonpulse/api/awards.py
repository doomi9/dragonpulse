"""Client for historical contract award data.

Design note
-----------
SAM.gov surfaces award information as *Award Notices* within the Opportunities
v2 API (``ptype=a``), each carrying an ``award`` sub-object (number, amount,
awardee). DragonPulse therefore implements the "Contract Awards" capability on
top of that same endpoint and key, normalizing results into :class:`Award`
records suitable for the future pricing-analyzer module.

This keeps the MVP to a single credential and a single, official API surface.
If a dedicated awards endpoint becomes available later, only this module needs
to change — the models and UI are already award-shaped.
"""

from __future__ import annotations

from typing import List, Optional

from dragonpulse.api.base import SamClient
from dragonpulse.api.opportunities import OpportunitiesClient
from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.award import Award, AwardSearchResult
from dragonpulse.models.common import NoticeType
from dragonpulse.models.filters import OpportunityFilters

logger = get_logger(__name__)


class AwardsClient:
    """Fetch and normalize historical awards (via award-notice opportunities)."""

    def __init__(
        self,
        sam_client: Optional[SamClient] = None,
        opportunities_client: Optional[OpportunitiesClient] = None,
    ) -> None:
        client = sam_client or SamClient()
        self.opportunities = opportunities_client or OpportunitiesClient(client)

    def search_awards(
        self,
        filters: OpportunityFilters,
        *,
        max_records: int = 100,
        ttl_seconds: Optional[int] = None,
        force_refresh: bool = False,
    ) -> AwardSearchResult:
        """Search award notices matching ``filters`` and normalize to awards.

        The ``notice_type_codes`` on the incoming filters are forced to
        ``Award Notice`` so the caller cannot accidentally pull non-award data.
        """
        award_filters = filters.model_copy(
            update={"notice_type_codes": [NoticeType.AWARD_NOTICE.value]}
        )
        awards: List[Award] = []
        for opp in self.opportunities.iter_all(
            award_filters,
            max_records=max_records,
            ttl_seconds=ttl_seconds,
            force_refresh=force_refresh,
        ):
            awards.append(Award.from_opportunity(opp))

        logger.info("Collected %d award records", len(awards))
        return AwardSearchResult(awards=awards, total_records=len(awards))
