"""Pricing analysis over historical awards.

Turns an :class:`AwardSearchResult` (collected from SAM.gov Award Notices) into
analysis-friendly structures: summary statistics, a histogram suitable for
charting, and a ranked list of comparable awards.

All math is plain Python / pandas — no external services. Every figure is
traceable back to the underlying award records (which carry their own
``sam_url``), preserving DragonPulse's grounding principle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from dragonpulse.config.logging_config import get_logger
from dragonpulse.models.award import Award, AwardSearchResult

logger = get_logger(__name__)


@dataclass
class PricingAnalysis:
    """Result of analyzing a set of awards for pricing intelligence."""

    total_awards: int
    priced_awards: int  # awards that actually had a parseable amount
    stats: dict = field(default_factory=dict)  # count/min/max/mean/median
    histogram: Optional[pd.DataFrame] = None  # columns: range, count
    awards_table: Optional[pd.DataFrame] = None
    awards: List[Award] = field(default_factory=list)

    @property
    def has_pricing(self) -> bool:
        return self.priced_awards > 0


def _histogram(amounts: List[float], bins: int = 8) -> pd.DataFrame:
    """Bucket amounts into ``bins`` ranges and count them.

    Returns a DataFrame indexed by a human-readable range label with a single
    ``count`` column (ready for ``st.bar_chart``).
    """
    if not amounts:
        return pd.DataFrame({"count": []})
    series = pd.Series(amounts)
    lo, hi = series.min(), series.max()
    if lo == hi:
        # All identical -> single bucket avoids a degenerate cut.
        label = f"${lo:,.0f}"
        return pd.DataFrame({"count": [len(amounts)]}, index=[label])
    cut = pd.cut(series, bins=bins)
    counts = cut.value_counts().sort_index()
    labels = [f"${int(iv.left):,}–${int(iv.right):,}" for iv in counts.index]
    return pd.DataFrame({"count": counts.values}, index=labels)


def _awards_table(awards: List[Award]) -> pd.DataFrame:
    rows = []
    for a in awards:
        rows.append(
            {
                "Awardee": a.awardee_name or "—",
                "Amount": a.award_amount,
                "Award #": a.award_number or "—",
                "Award Date": a.award_date or "—",
                "Agency": a.agency or "—",
                "NAICS": a.naics_code or "—",
                "Title": a.title or "—",
                "Link": a.sam_url or "",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Most expensive first; unpriced rows sink to the bottom.
        df = df.sort_values("Amount", ascending=False, na_position="last").reset_index(drop=True)
    return df


def analyze_awards(result: AwardSearchResult) -> PricingAnalysis:
    """Compute a :class:`PricingAnalysis` from collected awards."""
    awards = result.awards
    priced = [a for a in awards if a.award_amount is not None]
    stats = result.summary()
    logger.info(
        "Pricing analysis: %d awards, %d with parseable amounts", len(awards), len(priced)
    )
    return PricingAnalysis(
        total_awards=len(awards),
        priced_awards=len(priced),
        stats=stats,
        histogram=_histogram([a.award_amount for a in priced]),
        awards_table=_awards_table(awards),
        awards=awards,
    )
