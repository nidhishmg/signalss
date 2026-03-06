"""
dashboard_ui.py — Clean clinical UI components for BioFusion AI
"""
from __future__ import annotations
import streamlit as st
from typing import Any, Dict, List

DASHBOARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --blue:       #2563EB;
    --blue-light: #EFF6FF;
    --teal:       #0D9488;
    --teal-light: #F0FDFA;
    --red:        #DC2626;
    --red-light:  #FEF2F2;
    --amber:      #D97706;
    --amber-light:#FFFBEB;
    --green:      #16A34A;
    --green-light:#F0FDF4;
    --gray-50:    #F8FAFC;
    --gray-100:   #F1F5F9;
    --gray-200:   #E2E8F0;
    --gray-400:   #94A3B8;
    --gray-600:   #475569;
    --gray-800:   #1E293B;
    --white:      #FFFFFF;
    --shadow-sm:  0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow:     0 4px 12px rgba(0,0,0,0.06);
    --radius:     8px;
}

/* ── Reset ── */
div.stApp {
    background: var(--gray-50) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
h1,h2,h3,h4,h5,p,span,li,div {
    font-family: 'Inter', -apple-system, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Header ── */
.app-header {
    background: var(--white);
    border-bottom: 1px solid var(--gray-200);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}
.app-header .system-tag {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--teal);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.app-header h1 {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--gray-800) !important;
    margin: 0 0 0.25rem 0 !important;
    letter-spacing: -0.3px;
}
.app-header .subtitle {
    font-size: 0.85rem;
    color: var(--gray-400);
    margin: 0;
}

/* ── Patient strip ── */
.patient-strip {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 0.7rem 1.25rem;
    display: flex;
    gap: 2rem;
    align-items: center;
    margin-bottom: 1.25rem;
    font-size: 0.82rem;
    color: var(--gray-600);
}
.patient-strip .field { display: flex; align-items: center; gap: 6px; }
.patient-strip .label { color: var(--gray-400); font-weight: 500; }
.patient-strip .value { color: var(--gray-800); font-weight: 600; }
.demo-badge {
    margin-left: auto;
    background: var(--teal-light);
    color: var(--teal);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    border: 1px solid rgba(13,148,136,0.2);
}

/* ── Section headings ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--gray-400);
    margin: 0 0 0.75rem 0;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    box-shadow: var(--shadow-sm);
}
.metric-card .mc-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--gray-400);
    margin: 0 0 0.5rem 0;
}
.metric-card .mc-value {
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1;
    margin: 0 0 0.3rem 0;
}
.metric-card .mc-sub {
    font-size: 0.78rem;
    color: var(--gray-400);
    margin: 0;
}
.mc-border-blue  { border-top: 3px solid var(--blue);  }
.mc-border-red   { border-top: 3px solid var(--red);   }
.mc-border-amber { border-top: 3px solid var(--amber); }
.mc-border-teal  { border-top: 3px solid var(--teal);  }
.color-blue  { color: var(--blue);  }
.color-red   { color: var(--red);   }
.color-amber { color: var(--amber); }
.color-teal  { color: var(--teal);  }
.color-green { color: var(--green); }

/* ── Feature table ── */
.feat-table {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    overflow: hidden;
}
.feat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--gray-100);
    font-size: 0.84rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-row:hover { background: var(--gray-50); }
.feat-name { color: var(--gray-600); font-weight: 500; }
.feat-val  { font-family: 'SF Mono', 'Fira Code', monospace; color: var(--blue); font-weight: 600; font-size: 0.83rem; }

/* ── Alert boxes ── */
.alert {
    padding: 0.75rem 1rem;
    border-radius: var(--radius);
    border-left: 3px solid;
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
    font-weight: 500;
    line-height: 1.5;
}
.alert-critical { background: var(--red-light);   border-color: var(--red);   color: #7F1D1D; }
.alert-warning  { background: var(--amber-light);  border-color: var(--amber); color: #78350F; }
.alert-ok       { background: var(--green-light);  border-color: var(--green); color: #14532D; }
.alert-info     { background: var(--blue-light);   border-color: var(--blue);  color: #1E3A8A; }

/* ── Welcome card ── */
.welcome-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 4rem 2rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.welcome-card h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--gray-800) !important;
    margin: 1rem 0 0.5rem 0 !important;
}
.welcome-card p {
    font-size: 0.88rem;
    color: var(--gray-400);
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.7;
}
.modality-pill {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    border-radius: 6px;
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--gray-600);
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
}

/* ── Recommendation list ── */
.rec-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    overflow: hidden;
}
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--gray-100);
    font-size: 0.84rem;
    color: var(--gray-600);
    line-height: 1.5;
}
.rec-item:last-child { border-bottom: none; }
.rec-num {
    background: var(--blue-light);
    color: var(--blue);
    border-radius: 4px;
    width: 20px; height: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 700;
    flex-shrink: 0;
}

/* ── Interpretation box ── */
.interp-box {
    padding: 1rem 1.25rem;
    border-radius: var(--radius);
    font-size: 0.88rem;
    line-height: 1.7;
    border-left: 3px solid;
}
.interp-high     { background: var(--red-light);   border-color: var(--red);   color: #7F1D1D; }
.interp-moderate { background: var(--amber-light);  border-color: var(--amber); color: #78350F; }
.interp-low      { background: var(--green-light);  border-color: var(--green); color: #14532D; }

/* ── Sidebar overrides ── */
section[data-testid="stSidebar"] > div {
    background: var(--white) !important;
    border-right: 1px solid var(--gray-200);
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--gray-400);
    margin: 1.2rem 0 0.5rem 0;
}
.stButton > button {
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background: var(--blue) !important;
    border: 1px solid var(--blue) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
}
.stButton > button:not([kind="primary"]) {
    background: var(--white) !important;
    border: 1px solid var(--gray-200) !important;
    color: var(--gray-600) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--blue) !important;
    color: var(--blue) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--gray-200) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--gray-400) !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important;
    font-weight: 600 !important;
}
</style>
"""


def inject_css() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown("""
<div class="app-header">
  <div class="system-tag">Clinical Decision Support &nbsp;&middot;&nbsp; AI</div>
  <h1>Multimodal Neuro-Cardio-Muscular Clinical Intelligence System</h1>
  <p class="subtitle">Simultaneous EEG &middot; ECG &middot; EMG biosignal analysis with multimodal risk fusion</p>
</div>""", unsafe_allow_html=True)


def render_patient_strip(patient_id: str, age: int, gender: str,
                          demo: bool = False) -> None:
    demo_badge = '<span class="demo-badge">Demo Mode</span>' if demo else ""
    st.markdown(f"""
<div class="patient-strip">
  <div class="field"><span class="label">Patient ID</span><span class="value">{patient_id}</span></div>
  <div class="field"><span class="label">Age</span><span class="value">{age}</span></div>
  <div class="field"><span class="label">Gender</span><span class="value">{gender}</span></div>
  {demo_badge}
</div>""", unsafe_allow_html=True)


def render_metric_cards(eeg: Dict, ecg: Dict, emg: Dict, fusion: Dict) -> None:
    sp  = eeg.get("seizure_probability", 0.0)
    ap  = ecg.get("arrhythmia_probability", 0.0)
    mas = emg.get("muscle_abnormality_score", 0.0)
    rp  = fusion.get("risk_percent", 0.0)
    rl  = fusion.get("risk_level", "—")

    def _color(v: float) -> str:
        return "red" if v >= 0.7 else "amber" if v >= 0.3 else "green"

    sp_c  = _color(sp)
    ap_c  = _color(ap)
    mas_c = _color(mas)
    fus_c = {"HIGH": "red", "MODERATE": "amber", "LOW": "green"}.get(rl, "blue")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
<div class="metric-card mc-border-blue">
  <p class="mc-label">EEG &mdash; Seizure Probability</p>
  <p class="mc-value color-{sp_c}">{sp*100:.1f}%</p>
  <p class="mc-sub">RandomForest classifier</p>
</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class="metric-card mc-border-red">
  <p class="mc-label">ECG &mdash; Arrhythmia Probability</p>
  <p class="mc-value color-{ap_c}">{ap*100:.1f}%</p>
  <p class="mc-sub">HR {ecg.get('heart_rate', 0):.0f} bpm &middot; XGBoost</p>
</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
<div class="metric-card mc-border-amber">
  <p class="mc-label">EMG &mdash; Muscular Abnormality</p>
  <p class="mc-value color-{mas_c}">{mas*100:.1f}%</p>
  <p class="mc-sub">SVM (RBF kernel)</p>
</div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
<div class="metric-card mc-border-teal">
  <p class="mc-label">Fusion &mdash; Combined Risk</p>
  <p class="mc-value color-{fus_c}">{rp:.1f}%</p>
  <p class="mc-sub">Risk level: <strong>{rl}</strong></p>
</div>""", unsafe_allow_html=True)


def _feat_table(rows: list[tuple[str, str]]) -> None:
    html = '<div class="feat-table">'
    for name, val in rows:
        html += f'<div class="feat-row"><span class="feat-name">{name}</span><span class="feat-val">{val}</span></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_eeg_features(eeg: Dict) -> None:
    bands = eeg.get("band_powers", {})
    rows: list[tuple[str, str]] = [
        ("Seizure Probability",   f"{eeg.get('seizure_probability', 0)*100:.2f}%"),
        ("Spectral Entropy",      f"{eeg.get('spectral_entropy', 0):.4f}"),
        ("Spike Rate",            f"{eeg.get('spike_rate', 0):.3f} spk/s"),
        ("Signal Quality",        f"{eeg.get('signal_quality', 0)*100:.1f}%"),
    ]
    for name, key in [("Delta (0.5–4 Hz)", "delta"), ("Theta (4–8 Hz)", "theta"),
                       ("Alpha (8–13 Hz)", "alpha"), ("Beta (13–30 Hz)", "beta"),
                       ("Gamma (30–40 Hz)", "gamma")]:
        rows.append((f"Band Power — {name}", f"{bands.get(key, 0):.4f}"))
    _feat_table(rows)


def render_ecg_features(ecg: Dict) -> None:
    _feat_table([
        ("Arrhythmia Probability",  f"{ecg.get('arrhythmia_probability', 0)*100:.2f}%"),
        ("Heart Rate",              f"{ecg.get('heart_rate', 0):.1f} bpm"),
        ("RR Interval Mean",        f"{ecg.get('rr_mean_ms', 0):.1f} ms"),
        ("RR Interval Std Dev",     f"{ecg.get('rr_std_ms', 0):.2f} ms"),
        ("RMSSD",                   f"{ecg.get('rmssd_ms', 0):.2f} ms"),
        ("LF / HF Ratio",           f"{ecg.get('lf_hf_ratio', 0):.3f}"),
        ("QRS Duration",            f"{ecg.get('qrs_duration_ms', 0):.1f} ms"),
        ("R-Peaks Detected",        f"{ecg.get('r_peaks_count', 0)}"),
    ])


def render_emg_features(emg: Dict) -> None:
    _feat_table([
        ("Abnormality Score",       f"{emg.get('muscle_abnormality_score', 0)*100:.2f}%"),
        ("Mean Absolute Value",     f"{emg.get('mav', 0):.4f}"),
        ("RMS Energy",              f"{emg.get('rms_energy', 0):.4f}"),
        ("Waveform Length",         f"{emg.get('waveform_length', 0):.4f}"),
        ("Zero Crossing Rate",      f"{emg.get('zero_crossing_rate', 0):.4f}"),
        ("Slope Sign Changes",      f"{emg.get('slope_sign_changes', 0):.4f}"),
        ("Median Frequency",        f"{emg.get('median_frequency_hz', 0):.1f} Hz"),
        ("Mean Frequency",          f"{emg.get('mean_frequency_hz', 0):.1f} Hz"),
    ])


def render_clinical_alerts(alerts: List[str]) -> None:
    for alert in alerts:
        upper = alert.upper()
        if any(k in upper for k in ("HIGH", "URGENT", "SUDEP", "CRITICAL", "MULTI-ORGAN")):
            cls = "alert-critical"
        elif any(k in upper for k in ("MODERATE", "COUPLING", "WARNING", "ICTAL")):
            cls = "alert-warning"
        elif "NO CRITICAL" in upper or alert.startswith("No "):
            cls = "alert-ok"
        else:
            cls = "alert-info"
        # Strip leading emoji characters
        clean = alert.lstrip("⚠️🔴🟠✅").strip()
        st.markdown(f'<div class="alert {cls}">{clean}</div>', unsafe_allow_html=True)


def render_recommendations(recs: List[str]) -> None:
    st.markdown('<p class="section-title">Clinical Recommendations</p>', unsafe_allow_html=True)
    html = '<div class="rec-card">'
    for i, rec in enumerate(recs, 1):
        html += f'<div class="rec-item"><div class="rec-num">{i}</div><span>{rec}</span></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
