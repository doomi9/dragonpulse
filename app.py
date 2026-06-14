"""DragonPulse — Streamlit entry point.

Run with:
    streamlit run app.py

This file wires the package on ``sys.path`` (so the app runs without an editable
install), configures logging, and lays out the top-level tabbed navigation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Make the src/ package importable without requiring `pip install -e .` ---
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402

from dragonpulse import __version__  # noqa: E402
from dragonpulse.config.logging_config import configure_logging  # noqa: E402
from dragonpulse.config.settings import get_settings  # noqa: E402
from dragonpulse.ui import detail_view, placeholders, pricing_view, search_view  # noqa: E402
from dragonpulse.ui.sidebar import render_sidebar  # noqa: E402


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level.value)

    st.set_page_config(
        page_title="DragonPulse",
        page_icon="🐉",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    filters = render_sidebar()

    st.title("🐉 DragonPulse")
    st.caption(
        f"v{__version__} · Local-first SAM.gov opportunity intelligence · "
        "official APIs only · cache-first"
    )

    if not settings.has_api_key:
        st.warning(
            "No SAM.gov API key configured yet. Copy `.env.example` to `.env` and set "
            "`DRAGONPULSE_SAM_API_KEY_BASIC`. You can still explore the UI; searches "
            "will fail until a key is set.",
            icon="🔑",
        )

    tab_discover, tab_detail, tab_pricing, tab_rag, tab_proposal = st.tabs(
        ["🔍 Discover", "📄 Detail", "💰 Pricing", "📚 Knowledge Base", "📝 Proposals"]
    )

    with tab_discover:
        search_view.render_search(filters)
    with tab_detail:
        detail_view.render_detail()
    with tab_pricing:
        pricing_view.render_pricing(filters)
    with tab_rag:
        placeholders.render_rag()
    with tab_proposal:
        placeholders.render_proposal()


if __name__ == "__main__":
    main()
