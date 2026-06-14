"""DragonPulse — local-first SAM.gov opportunity intelligence for government contractors.

Package layout
--------------
- ``config``     : settings, environment loading, logging setup.
- ``cache``      : disk-based response caching (TTL + param hashing).
- ``models``     : Pydantic models for SAM.gov opportunities and awards.
- ``api``        : SAM.gov API clients (Opportunities v2, Contract Awards).
- ``processors`` : business logic (checklists, outreach drafts, attachments, LLM).
- ``ui``         : Streamlit views (sidebar filters, search table, detail view).
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
