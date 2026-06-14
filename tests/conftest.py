"""Pytest fixtures and path setup for DragonPulse tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure the src/ package is importable.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def sample_opportunity_payload() -> Dict[str, Any]:
    """A realistic (trimmed) Opportunities v2 search payload for one notice.

    Mirrors the shape of the live API including bare-string ``resourceLinks``
    and a list of ``pointOfContact`` entries.
    """
    return {
        "totalRecords": 1,
        "limit": 25,
        "offset": 0,
        "opportunitiesData": [
            {
                "noticeId": "abc123def456",
                "title": "Cybersecurity Support Services",
                "solicitationNumber": "W912-25-R-0042",
                "fullParentPathName": "DEPT OF DEFENSE.DEPT OF THE ARMY.ACC",
                "fullParentPathCode": "057.2100.W912",
                "type": "Combined Synopsis/Solicitation",
                "baseType": "Combined Synopsis/Solicitation",
                "postedDate": "2026-06-01",
                "responseDeadLine": "2026-07-15T17:00:00-04:00",
                "archiveDate": "2026-08-15",
                "naicsCode": "541512",
                "classificationCode": "D310",
                "typeOfSetAside": "SDVOSBC",
                "typeOfSetAsideDescription": "Service-Disabled Veteran-Owned Small Business",
                "active": "Yes",
                "pointOfContact": [
                    {
                        "type": "primary",
                        "fullName": "Jane Contracting Officer",
                        "title": "Contract Specialist",
                        "email": "jane.co@army.mil",
                        "phone": "555-123-4567",
                    },
                    {
                        "type": "secondary",
                        "fullName": "John Specialist",
                        "email": "john.s@army.mil",
                    },
                ],
                "officeAddress": {"city": "Aberdeen", "state": "MD", "zip": "21005"},
                "placeOfPerformance": {"city": "Aberdeen", "state": "MD"},
                "description": "https://api.sam.gov/opportunities/v2/desc/abc123",
                "uiLink": "https://sam.gov/opp/abc123def456/view",
                "resourceLinks": [
                    "https://sam.gov/api/prod/opps/v3/opportunities/resources/files/uuid1/download",
                    "https://sam.gov/api/prod/opps/v3/opportunities/resources/files/uuid2/download",
                ],
            }
        ],
    }


@pytest.fixture()
def sample_award_payload() -> Dict[str, Any]:
    """A trimmed Award Notice payload (ptype=a) with an ``award`` sub-object."""
    return {
        "totalRecords": 1,
        "limit": 25,
        "offset": 0,
        "opportunitiesData": [
            {
                "noticeId": "award999",
                "title": "IT Services Award",
                "type": "Award Notice",
                "fullParentPathName": "GENERAL SERVICES ADMINISTRATION",
                "naicsCode": "541512",
                "postedDate": "2026-05-20",
                "award": {
                    "number": "GS-35F-0001",
                    "amount": "$1,250,000.00",
                    "date": "2026-05-19",
                    "awardee": {"name": "Acme Federal LLC", "ueiSAM": "ABC123XYZ"},
                },
            }
        ],
    }
