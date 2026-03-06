"""
main_app.py — Unified Med-AI Platform

Entry point combining:
  - AI Silent Disease Predictor (Face & Voice biomarkers)
  - BioFusion AI (EEG, ECG, EMG clinical intelligence)

Run:
    python -m streamlit run main_app.py
"""
from __future__ import annotations
import os
import sys
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Med-AI Unified Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "nav_target" not in st.session_state:
    st.session_state["nav_target"] = "home"

# ── Sidebar navigation ─────────────────────────────────────────────────────────
_NAV_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 230px !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.3rem !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(26,115,232,0.25), rgba(124,58,237,0.2)) !important;
    border-color: rgba(26,115,232,0.4) !important;
    color: #93c5fd !important;
}
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(_NAV_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem 0;text-align:center;">
      <span style="font-size:2rem;">⚕️</span><br>
      <span style="font-size:0.65rem;font-weight:700;letter-spacing:2px;
                  text-transform:uppercase;color:#475569;">Unified Med-AI</span><br>
      <span style="font-size:1rem;font-weight:700;color:#e2e8f0;">Platform</span>
    </div>
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:0 0 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.68rem;letter-spacing:1.2px;text-transform:uppercase;color:#334155;margin:0 0 0.5rem 0;">Navigation</p>',
                unsafe_allow_html=True)

    current = st.session_state.get("nav_target", "home")

    if st.button("🏠   Home", key="nav_home",
                 type="primary" if current == "home" else "secondary"):
        st.session_state["nav_target"] = "home"
        st.rerun()

    if st.button("🧬   Face & Voice Predictor", key="nav_face",
                 type="primary" if current == "silent_disease" else "secondary"):
        st.session_state["nav_target"] = "silent_disease"
        st.rerun()

    if st.button("🧠   BioFusion AI (EEG·ECG·EMG)", key="nav_bio",
                 type="primary" if current == "biofusion_ai" else "secondary"):
        st.session_state["nav_target"] = "biofusion_ai"
        st.rerun()

    st.markdown("""
    <hr style="border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0 1rem 0;">
    <p style="font-size:0.7rem;color:#1e293b;text-align:center;line-height:1.7;">
      For clinical decision support only.<br>Not a diagnostic replacement.<br>
      <span style="color:#1e293b;">© 2026 Med-AI Platform</span>
    </p>
    """, unsafe_allow_html=True)

# ── Route to the correct page ──────────────────────────────────────────────────
target = st.session_state.get("nav_target", "home")

if target == "home":
    from home_page import render_home
    render_home()

elif target == "silent_disease":
    # The cloned repo app — inject sys.path so it can find its own modules
    SILENT_ROOT = os.path.join(PROJECT_ROOT, "Ai-in-bio-medical")
    if SILENT_ROOT not in sys.path:
        sys.path.insert(0, SILENT_ROOT)

    # Dynamically load the modules from the cloned repo
    import importlib.util

    def _run_module(filepath: str) -> None:
        """Execute a Python file in the current interpreter context,
        skipping the st.set_page_config call (already set)."""
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        # Remove the set_page_config call so it doesn't conflict
        source = source.replace("st.set_page_config(", "_SKIP_set_page_config(")
        # Provide a dummy function for the skipped call
        _globals = {
            "__file__": filepath,
            "__name__": "__main__",
            "_SKIP_set_page_config": lambda **kw: None,
        }
        exec(compile(source, filepath, "exec"), _globals)  # noqa: S102

    _run_module(os.path.join(SILENT_ROOT, "app.py"))

elif target == "biofusion_ai":
    # BioFusion app — already in sys.path via PROJECT_ROOT
    BIOFUSION_ROOT = PROJECT_ROOT
    if BIOFUSION_ROOT not in sys.path:
        sys.path.insert(0, BIOFUSION_ROOT)

    import importlib.util

    def _run_module_bf(filepath: str) -> None:
        with open(filepath, "r", encoding="utf-8") as fh:
            source = fh.read()
        source = source.replace("st.set_page_config(", "_SKIP_set_page_config(")
        _globals = {
            "__file__": filepath,
            "__name__": "__main__",
            "_SKIP_set_page_config": lambda **kw: None,
        }
        exec(compile(source, filepath, "exec"), _globals)  # noqa: S102

    _run_module_bf(os.path.join(PROJECT_ROOT, "biofusion_ai", "app.py"))
