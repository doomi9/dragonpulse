"""Award models.

DragonPulse sources historical award data from the SAM.gov Opportunities API
filtered to *Award Notices* (``ptype=a``). Each award notice carries an
``award`` sub-object (number, amount, awardee). This provides real,
key-compatible pricing signal for the future pricing-analyzer module without
introducing a second credential or data source.

The models below normalize that data into an analysis-friendly shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from dragonpulse.models.opportunity import Opportunity


class Award(BaseModel):
    """A normalized historical award derived from an award-notice opportunity."""

    model_config = ConfigDict(extra="ignore")

    notice_id: str
    title: Optional[str] = None
    agency: Optional[str] = None
    office: Optional[str] = None
    naics_code: Optional[str] = None
    set_aside_description: Optional[str] = None

    award_number: Optional[str] = None
    award_amount: Optional[float] = None
    award_amount_raw: Optional[str] = None
    award_date: Optional[str] = None
    awardee_name: Optional[str] = None
    awardee_uei: Optional[str] = None

    posted_date: Optional[datetime] = None
    sam_url: Optional[str] = None

    @classmethod
    def from_opportunity(cls, opp: Opportunity) -> "Award":
        """Build an :class:`Award` from an award-notice :class:`Opportunity`."""
        amount_raw = opp.award.amount if opp.award else None
        return cls(
            notice_id=opp.notice_id,
            title=opp.title,
            agency=opp.agency,
            office=opp.office,
            naics_code=opp.naics_code,
            set_aside_description=opp.set_aside_description,
            award_number=opp.award.number if opp.award else None,
            award_amount=_to_float(amount_raw),
            award_amount_raw=amount_raw,
            award_date=opp.award.date if opp.award else None,
            awardee_name=opp.award.awardee_name if opp.award else None,
            awardee_uei=opp.award.awardee_uei if opp.award else None,
            posted_date=opp.posted_date,
            sam_url=opp.sam_url,
        )


def _to_float(value: Optional[str]) -> Optional[float]:
    """Parse a currency-ish string like ``"$1,234.56"`` into a float."""
    if value is None:
        return None
    cleaned = str(value).replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


class AwardSearchResult(BaseModel):
    """Collection of normalized awards plus simple summary statistics."""

    awards: List[Award] = Field(default_factory=list)
    total_records: int = 0
    from_cache: bool = False

    @property
    def amounts(self) -> List[float]:
        return [a.award_amount for a in self.awards if a.award_amount is not None]

    def summary(self) -> dict:
        """Return basic pricing statistics (count, min, max, mean, median)."""
        amounts = sorted(self.amounts)
        n = len(amounts)
        if n == 0:
            return {"count": 0}
        mean = sum(amounts) / n
        mid = n // 2
        median = amounts[mid] if n % 2 else (amounts[mid - 1] + amounts[mid]) / 2
        return {
            "count": n,
            "min": amounts[0],
            "max": amounts[-1],
            "mean": mean,
            "median": median,
        }
