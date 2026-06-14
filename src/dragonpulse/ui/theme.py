"""Branding and HeroUI-inspired dark theme for the DragonPulse dashboard.

HeroUI (heroui.com) is a React component library and cannot run inside
Streamlit. This module instead reproduces its *look and feel* — a dark surface
palette, blue accent, soft rounded "cards", subtle borders, and modern
typography — via injected CSS, and centralizes the Dragon Infrastructure logo.

Everything here is purely presentational; no business logic lives in this file.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from typing import Optional

import streamlit as st

# Repo root: .../dragonpulse (this file is src/dragonpulse/ui/theme.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGO_PATH = _PROJECT_ROOT / "assets" / "dragonpulse-logo.png"

# --- HeroUI-inspired palette (tuned to the blue/black Dragon logo) -----------
PALETTE = {
    "bg": "#0a0d14",            # app background (near-black)
    "surface": "#111620",       # cards / sidebar
    "surface_2": "#161c28",     # elevated surface
    "border": "#222b3a",        # hairline borders
    "primary": "#2f7be6",       # dragon blue
    "primary_hover": "#4790f0",
    "primary_soft": "rgba(47,123,230,0.15)",
    "text": "#e6edf3",          # primary text
    "text_muted": "#9aa7b8",    # secondary text
    "success": "#2ecc71",
    "warning": "#f5a623",
    "danger": "#e5484d",
}


def _detect_mime(data: bytes) -> str:
    """Detect image MIME from magic bytes (the asset may be JPEG or PNG)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


@lru_cache(maxsize=1)
def _logo_data_uri() -> Optional[str]:
    """Return the logo as a base64 data URI (cached), or None if missing."""
    if not LOGO_PATH.exists():
        return None
    raw = LOGO_PATH.read_bytes()
    mime = _detect_mime(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def page_icon() -> str:
    """Page/tab icon: the logo file path if available, else a dragon emoji."""
    return str(LOGO_PATH) if LOGO_PATH.exists() else "🐉"


def apply_theme() -> None:
    """Inject the global dark, HeroUI-style CSS. Call once near app start."""
    p = PALETTE
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --dp-bg: {p['bg']};
        --dp-surface: {p['surface']};
        --dp-surface-2: {p['surface_2']};
        --dp-border: {p['border']};
        --dp-primary: {p['primary']};
        --dp-primary-hover: {p['primary_hover']};
        --dp-text: {p['text']};
        --dp-text-muted: {p['text_muted']};
    }}

    /* Base surfaces */
    .stApp {{
        background: radial-gradient(1200px 600px at 80% -10%,
                    rgba(47,123,230,0.10), transparent 60%), var(--dp-bg);
        color: var(--dp-text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background: var(--dp-surface);
        border-right: 1px solid var(--dp-border);
    }}

    h1, h2, h3, h4 {{ color: var(--dp-text); font-weight: 700; letter-spacing: -0.01em; }}
    p, span, label, li {{ color: var(--dp-text); }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--dp-text-muted) !important; }}

    /* HeroUI-style buttons: rounded, blue, soft shadow */
    .stButton > button {{
        border-radius: 12px;
        border: 1px solid var(--dp-border);
        background: var(--dp-surface-2);
        color: var(--dp-text);
        font-weight: 600;
        transition: all 0.15s ease;
    }}
    .stButton > button:hover {{
        border-color: var(--dp-primary);
        color: #fff;
        transform: translateY(-1px);
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(180deg, var(--dp-primary-hover), var(--dp-primary));
        border: none;
        box-shadow: 0 6px 18px rgba(47,123,230,0.35);
    }}
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 8px 24px rgba(47,123,230,0.5);
    }}
    .stDownloadButton > button {{ border-radius: 12px; }}

    /* Cards / bordered containers */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--dp-surface);
        border: 1px solid var(--dp-border) !important;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}

    /* Metrics as HeroUI stat cards */
    [data-testid="stMetric"] {{
        background: var(--dp-surface-2);
        border: 1px solid var(--dp-border);
        border-radius: 16px;
        padding: 14px 18px;
    }}
    [data-testid="stMetricValue"] {{ color: var(--dp-text); font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: var(--dp-text-muted); }}

    /* Tabs: pill style */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: var(--dp-surface);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid var(--dp-border);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 8px 16px;
        color: var(--dp-text-muted);
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--dp-surface-2);
        color: var(--dp-text) !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background: var(--dp-primary); }}

    /* Inputs */
    .stTextInput input, .stDateInput input, .stNumberInput input,
    [data-baseweb="select"] > div {{
        background: var(--dp-surface-2) !important;
        border-radius: 10px !important;
        border: 1px solid var(--dp-border) !important;
        color: var(--dp-text) !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--dp-border);
        border-radius: 14px;
        overflow: hidden;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        border: 1px solid var(--dp-border);
        border-radius: 14px;
        background: var(--dp-surface);
    }}

    /* Hide default Streamlit chrome for a cleaner dashboard */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero_header(version: str) -> None:
    """Render the branded hero header with the Dragon Infrastructure logo."""
    uri = _logo_data_uri()
    logo_html = (
        f'<img src="{uri}" alt="DragonPulse" '
        'style="height:74px;border-radius:14px;border:1px solid var(--dp-border);'
        'background:#000;padding:6px;box-shadow:0 6px 20px rgba(0,0,0,0.4);" />'
        if uri
        else '<div style="font-size:48px;">🐉</div>'
    )
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:20px;
                    padding:20px 24px;margin-bottom:8px;
                    background:linear-gradient(90deg, rgba(47,123,230,0.12), transparent);
                    border:1px solid var(--dp-border);border-radius:18px;">
            {logo_html}
            <div>
                <div style="font-size:30px;font-weight:800;letter-spacing:-0.02em;
                            color:var(--dp-text);line-height:1.1;">DragonPulse</div>
                <div style="color:var(--dp-text-muted);font-size:13.5px;margin-top:4px;">
                    v{version} · SAM.gov opportunity intelligence ·
                    <span style="color:var(--dp-primary);">Dragon Infrastructure</span> ·
                    local-first · cache-first
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_logo() -> None:
    """Render the logo at the top of the sidebar."""
    uri = _logo_data_uri()
    if not uri:
        return
    st.sidebar.markdown(
        f"""
        <div style="text-align:center;padding:8px 4px 14px 4px;">
            <img src="{uri}" alt="DragonPulse"
                 style="width:100%;max-width:220px;border-radius:12px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
