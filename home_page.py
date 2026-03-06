"""
home_page.py — Unified Med-AI Platform Home Page

Landing page that lets the user choose between:
  1. AI Silent Disease Predictor (Face & Voice)
  2. BioFusion AI (EEG, ECG, EMG)
"""
from __future__ import annotations
import streamlit as st

_HOME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1526 40%, #0a1628 70%, #0f1f35 100%) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem 2rem;
    position: relative;
}
.hero-logo {
    font-size: 4rem;
    margin-bottom: 0.5rem;
    display: block;
    animation: float 4s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
.hero-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1a73e8, #7c3aed);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero h1 {
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    color: #f1f5f9 !important;
    line-height: 1.1 !important;
    margin: 0 0 1rem 0 !important;
    letter-spacing: -1.5px;
}
.hero h1 span {
    background: linear-gradient(90deg, #1a73e8, #7c3aed, #06b6d4);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem !important;
    color: #94a3b8 !important;
    max-width: 560px;
    margin: 0 auto 2.5rem auto !important;
    line-height: 1.7 !important;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-bottom: 3rem;
    flex-wrap: wrap;
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Module cards ── */
.module-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    max-width: 960px;
    margin: 0 auto 3rem auto;
    padding: 0 1rem;
}
@media (max-width: 768px) {
    .module-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.2rem !important; }
}

.module-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
    cursor: pointer;
    text-decoration: none;
}
.module-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 20px;
    opacity: 0;
    transition: opacity 0.3s ease;
}
.module-card.blue::before {
    background: linear-gradient(135deg, rgba(26,115,232,0.12), rgba(6,182,212,0.08));
}
.module-card.purple::before {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(236,72,153,0.08));
}
.module-card:hover::before { opacity: 1; }
.module-card:hover {
    transform: translateY(-6px);
    border-color: rgba(255,255,255,0.18);
    box-shadow: 0 24px 60px -12px rgba(0,0,0,0.6);
}
.module-card.blue:hover { border-color: rgba(26,115,232,0.4); }
.module-card.purple:hover { border-color: rgba(124,58,237,0.4); }

.card-icon {
    font-size: 3rem;
    margin-bottom: 1.25rem;
    display: block;
    position: relative;
}
.card-badge {
    position: absolute;
    top: 1.25rem;
    right: 1.25rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    text-transform: uppercase;
}
.badge-blue  { background: rgba(26,115,232,0.2);  color: #60a5fa; border: 1px solid rgba(26,115,232,0.3); }
.badge-purple{ background: rgba(124,58,237,0.2);  color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }

.card-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.card-desc {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.6;
    margin: 0 0 1.5rem 0;
}

/* Modality pills */
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.5rem; }
.pill {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
}
.pill-blue   { background: rgba(26,115,232,0.15);  color: #93c5fd; }
.pill-purple { background: rgba(124,58,237,0.15);  color: #c4b5fd; }
.pill-cyan   { background: rgba(6,182,212,0.15);   color: #67e8f9; }
.pill-pink   { background: rgba(236,72,153,0.15);  color: #f9a8d4; }
.pill-green  { background: rgba(34,197,94,0.15);   color: #86efac; }
.pill-orange { background: rgba(251,146,60,0.15);  color: #fdba74; }

/* Launch button inside card */
.card-cta {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.88rem;
    font-weight: 700;
    padding: 0.65rem 1.4rem;
    border-radius: 10px;
    transition: all 0.25s ease;
    letter-spacing: 0.2px;
    text-decoration: none;
}
.cta-blue {
    background: linear-gradient(135deg, #1a73e8, #0ea5e9);
    color: white;
    box-shadow: 0 4px 16px rgba(26,115,232,0.35);
}
.cta-blue:hover  { box-shadow: 0 6px 24px rgba(26,115,232,0.5); transform: translateY(-1px); }
.cta-purple {
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    color: white;
    box-shadow: 0 4px 16px rgba(124,58,237,0.35);
}
.cta-purple:hover{ box-shadow: 0 6px 24px rgba(124,58,237,0.5); transform: translateY(-1px); }

/* ── Feature list inside cards ── */
.feature-list { list-style: none; padding: 0; margin: 0; }
.feature-list li {
    font-size: 0.82rem;
    color: #94a3b8;
    padding: 0.35rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.feature-list li::before {
    content: '✓';
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}
.blue .feature-list li::before  { color: #60a5fa; }
.purple .feature-list li::before { color: #a78bfa; }

/* ── Divider ── */
.card-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1.25rem 0;
}

/* ── Footer ── */
.home-footer {
    text-align: center;
    padding: 1.5rem 1rem 2rem 1rem;
    color: #334155;
    font-size: 0.78rem;
    line-height: 1.8;
}
.home-footer a { color: #475569; text-decoration: none; }

/* Glow orbs */
.glow-orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(80px);
    opacity: 0.15;
}
.orb-blue {
    width: 400px; height: 400px;
    background: #1a73e8;
    top: -100px; left: -100px;
}
.orb-purple {
    width: 350px; height: 350px;
    background: #7c3aed;
    bottom: -80px; right: -80px;
}
.orb-cyan {
    width: 250px; height: 250px;
    background: #06b6d4;
    top: 50%; right: 20%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* Streamlit button styling */
div[data-testid="column"] .stButton > button {
    width: 100%;
    padding: 1.1rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: none !important;
    letter-spacing: 0.2px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
</style>
"""

def render_home():
    st.markdown(_HOME_CSS, unsafe_allow_html=True)

    # Glow orbs
    st.markdown("""
    <div class="glow-orb orb-blue"></div>
    <div class="glow-orb orb-purple"></div>
    <div class="glow-orb orb-cyan"></div>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown("""
    <div class="hero">
        <span class="hero-logo">⚕️</span>
        <div class="hero-pill">Unified Med-AI Platform</div>
        <h1>Preventive Healthcare<br><span>Powered by AI</span></h1>
        <p class="hero-sub">
            Two powerful diagnostic modules — one unified platform. 
            Detect silent diseases from face &amp; voice biomarkers, 
            or perform deep biosignal analysis with EEG, ECG &amp; EMG.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats strip
    st.markdown("""
    <div class="stats-strip">
        <div class="stat-item">
            <div class="stat-value">9+</div>
            <div class="stat-label">Biomarkers</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">3</div>
            <div class="stat-label">Signal Types</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">77.9%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">Real-time</div>
            <div class="stat-label">Analysis</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">0</div>
            <div class="stat-label">Data Stored</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Module selection cards
    st.markdown('<div class="module-grid">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="module-card blue">
            <span class="card-badge badge-blue">● Live</span>
            <span class="card-icon">🧬</span>
            <div class="card-title">AI Silent Disease<br>Predictor</div>
            <p class="card-desc">
                Non-invasive preventive health screening using facial landmark 
                analysis and vocal stress markers fused through a trained 
                RandomForest ML model.
            </p>
            <div class="pill-row">
                <span class="pill pill-blue">👁 Face Biomarkers</span>
                <span class="pill pill-cyan">🎤 Voice Analysis</span>
                <span class="pill pill-green">🤖 ML Fusion Engine</span>
                <span class="pill pill-blue">📊 Risk Scoring</span>
            </div>
            <div class="card-divider"></div>
            <ul class="feature-list">
                <li>Eye Aspect Ratio &amp; Blink Instability</li>
                <li>Facial Symmetry via MediaPipe FaceMesh</li>
                <li>MFCCs · Pitch Instability · RMS Energy</li>
                <li>PDF Health Report Download</li>
                <li>Digital Twin 6-Month Projection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧬  Launch Face & Voice Predictor", key="btn_face_voice", 
                     use_container_width=True):
            st.session_state["nav_target"] = "silent_disease"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="module-card purple">
            <span class="card-badge badge-purple">● Live</span>
            <span class="card-icon">🧠</span>
            <div class="card-title">BioFusion AI<br>Clinical Intelligence</div>
            <p class="card-desc">
                Deep multimodal biosignal analysis across EEG, ECG, and EMG 
                modalities with cross-organ coupling indices and clinical 
                alert generation.
            </p>
            <div class="pill-row">
                <span class="pill pill-purple">🧠 EEG · Seizure</span>
                <span class="pill pill-pink">❤️ ECG · Arrhythmia</span>
                <span class="pill pill-orange">💪 EMG · Neuromuscular</span>
                <span class="pill pill-purple">🔀 Fusion Engine</span>
            </div>
            <div class="card-divider"></div>
            <ul class="feature-list">
                <li>EEGNet · ResNet18-1D · CNN-LSTM models</li>
                <li>HRV Analysis &amp; Seizure Probability</li>
                <li>Neuro-Cardiac-Muscular Coupling Indices</li>
                <li>Clinical Alert Generation</li>
                <li>AI Interpretation &amp; Recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧠  Launch BioFusion AI (EEG·ECG·EMG)", key="btn_biofusion",
                     use_container_width=True):
            st.session_state["nav_target"] = "biofusion_ai"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="home-footer">
        <strong style="color:#475569;">⚕️ Unified Med-AI Platform</strong><br>
        For research and clinical decision support only. Not a replacement for professional medical diagnosis.<br>
        <span style="color:#1e293b;">© 2026 Med-AI Platform · All data processed in-memory · Zero data persistence</span>
    </div>
    """, unsafe_allow_html=True)
