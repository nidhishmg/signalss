"""
app.py — BioFusion AI · Doctor Dashboard (Clean UI)

Multimodal Neuro-Cardio-Muscular Clinical Intelligence System

Run:
  streamlit run app.py
"""
from __future__ import annotations
import io, os, sys, time
import numpy as np
import streamlit as st

# Path resolution removed natively

from biofusion_ai.modules.eeg_analysis  import analyze_eeg
from biofusion_ai.modules.ecg_analysis  import analyze_ecg
from biofusion_ai.modules.emg_analysis  import analyze_emg
from biofusion_ai.modules.fusion_engine import compute_risk

from biofusion_ai.frontend.dashboard_ui import (
    inject_css, render_header, render_patient_strip,
    render_metric_cards, render_eeg_features, render_ecg_features,
    render_emg_features, render_clinical_alerts, render_recommendations,
)
from biofusion_ai.frontend.visualization import (
    eeg_chart, ecg_chart, emg_chart, fusion_gauge, cross_organ_radar,
)

# ── Page config ───────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="BioFusion AI — Clinical Intelligence",
#     page_icon="+",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )
inject_css()

# ── Session state ─────────────────────────────────────────────────────────────
def _init() -> None:
    for k, v in {
        "eeg_result": None, "ecg_result": None,
        "emg_result": None, "fusion_result": None,
        "analysed": False, "demo_mode": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:1rem 0 0.5rem 0;">
  <p style="font-size:0.7rem;font-weight:600;letter-spacing:1.2px;
            text-transform:uppercase;color:#94A3B8;margin:0 0 0.3rem 0;">
    Clinical Intelligence
  </p>
  <p style="font-size:1rem;font-weight:700;color:#1E293B;margin:0;">
    BioFusion AI
  </p>
</div>
<hr style="border:none;border-top:1px solid #E2E8F0;margin:0.5rem 0 1rem 0;">
""", unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Patient Information</p>', unsafe_allow_html=True)
    patient_id = st.text_input("Patient ID", value="PT-20240001", label_visibility="collapsed",
                                placeholder="Patient ID")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", 1, 120, 45, label_visibility="visible")
    with col_b:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], label_visibility="visible")

    st.markdown('<p class="sidebar-label" style="margin-top:1.2rem;">Biosignal Upload</p>',
                unsafe_allow_html=True)
    eeg_file = st.file_uploader("EEG (.edf / .csv)",  type=["edf","csv","txt"])
    ecg_file = st.file_uploader("ECG (.csv / .dat)",  type=["csv","txt","dat"])
    emg_file = st.file_uploader("EMG (.csv)",         type=["csv","txt"])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        demo_clicked = st.button("Demo", use_container_width=True)
    with c2:
        run_clicked  = st.button("Run Analysis", type="primary", use_container_width=True)

    st.markdown("""
<hr style="border:none;border-top:1px solid #E2E8F0;margin:1.5rem 0 0.8rem 0;">
<p style="font-size:0.72rem;color:#CBD5E1;text-align:center;line-height:1.6;">
  For clinical decision support only.<br>Not a diagnostic replacement.
</p>""", unsafe_allow_html=True)

    # ── Model status ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="sidebar-label">Model Status</p>', unsafe_allow_html=True)
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _models = {
        "EEGNet": os.path.join(_base_dir, "models", "eeg_model.pt"),
        "ResNet18-1D": os.path.join(_base_dir, "models", "ecg_model.pt"),
        "CNN-LSTM": os.path.join(_base_dir, "models", "emg_model.pt"),
    }
    for name, path in _models.items():
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            st.markdown(
                f'<div style="font-size:0.78rem;padding:0.2rem 0;color:#16A34A;">'  
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:#16A34A;margin-right:6px;"></span>'
                f'<strong>{name}</strong> &nbsp; <span style="color:#94A3B8;">{size_kb:.0f} KB</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="font-size:0.78rem;padding:0.2rem 0;color:#D97706;">'  
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                f'background:#D97706;margin-right:6px;"></span>'
                f'<strong>{name}</strong> &nbsp; <span style="color:#94A3B8;">will train on first run</span></div>',
                unsafe_allow_html=True
            )

# ── Main content ──────────────────────────────────────────────────────────────
render_header()
render_patient_strip(patient_id, age, gender, demo=st.session_state.demo_mode)

# ── Analysis runner ───────────────────────────────────────────────────────────
def _run(demo: bool = False, seed: int = 42) -> None:
    bar = st.progress(0, text="Initialising…")

    bar.progress(5,  text="Preprocessing EEG…")
    if not demo and eeg_file:
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", f"_tmp_{eeg_file.name}")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as f: f.write(eeg_file.getvalue())
        if eeg_file.name.endswith(".edf"):
            st.session_state.eeg_result = analyze_eeg(file_path=tmp.replace(".edf",""))
        else:
            try:
                arr = np.loadtxt(tmp, delimiter=",", skiprows=1, usecols=0,
                                  max_rows=100_000).astype(np.float32)
                st.session_state.eeg_result = analyze_eeg(signal_data=arr)
            except Exception:
                st.session_state.eeg_result = analyze_eeg(deterministic_seed=seed)
    else:
        st.session_state.eeg_result = analyze_eeg(deterministic_seed=seed)

    bar.progress(30, text="EEG features extracted.")
    time.sleep(0.2)

    bar.progress(35, text="Preprocessing ECG…")
    if not demo and ecg_file:
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", f"_tmp_{ecg_file.name}")
        with open(tmp, "wb") as f: f.write(ecg_file.getvalue())
        try:
            arr = np.loadtxt(tmp, delimiter=",", skiprows=1, usecols=0,
                              max_rows=200_000).astype(np.float32)
            st.session_state.ecg_result = analyze_ecg(signal_data=arr)
        except Exception:
            st.session_state.ecg_result = analyze_ecg(deterministic_seed=seed + 1)
    else:
        st.session_state.ecg_result = analyze_ecg(deterministic_seed=seed + 1)

    bar.progress(60, text="ECG HRV features computed.")
    time.sleep(0.2)

    bar.progress(65, text="Preprocessing EMG…")
    if not demo and emg_file:
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", f"_tmp_{emg_file.name}")
        with open(tmp, "wb") as f: f.write(emg_file.getvalue())
        st.session_state.emg_result = analyze_emg(file_path=tmp, deterministic_seed=seed + 2)
    else:
        st.session_state.emg_result = analyze_emg(deterministic_seed=seed + 2)

    bar.progress(85, text="EMG motor-unit features extracted.")
    time.sleep(0.2)

    bar.progress(92, text="Running multimodal fusion…")
    st.session_state.fusion_result = compute_risk(
        st.session_state.eeg_result,
        st.session_state.ecg_result,
        st.session_state.emg_result,
    )
    bar.progress(100, text="Analysis complete.")
    time.sleep(0.4)
    bar.empty()
    st.session_state.analysed   = True
    st.session_state.demo_mode  = demo

if run_clicked:
    _run(demo=False)
if demo_clicked or (st.session_state.demo_mode and not st.session_state.analysed):
    _run(demo=True, seed=42)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.analysed:
    eeg_r = st.session_state.eeg_result
    ecg_r = st.session_state.ecg_result
    emg_r = st.session_state.emg_result
    fus_r = st.session_state.fusion_result

    # Metric cards
    st.markdown('<p class="section-title">Biosignal Analysis Summary</p>', unsafe_allow_html=True)
    render_metric_cards(eeg_r, ecg_r, emg_r, fus_r)
    st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:1.5rem 0;'>",
                unsafe_allow_html=True)

    # Waveform tabs
    tab_eeg, tab_ecg, tab_emg, tab_fus = st.tabs(
        ["EEG Waveform", "ECG Waveform", "EMG Waveform", "Fusion Report"]
    )

    with tab_eeg:
        col_chart, col_feats = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(eeg_chart(eeg_r), use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True,
                                    "displaylogo": False})
        with col_feats:
            st.markdown('<p class="section-title" style="margin-top:0.3rem;">EEG Feature Profile</p>',
                        unsafe_allow_html=True)
            render_eeg_features(eeg_r)
            st.caption(f"Source: {eeg_r.get('model_used', '—')}")

    with tab_ecg:
        col_chart, col_feats = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(ecg_chart(ecg_r), use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True,
                                    "displaylogo": False})
        with col_feats:
            st.markdown('<p class="section-title" style="margin-top:0.3rem;">HRV Feature Profile</p>',
                        unsafe_allow_html=True)
            render_ecg_features(ecg_r)
            st.caption(f"Source: {ecg_r.get('model_used', '—')}")

    with tab_emg:
        col_chart, col_feats = st.columns([2, 1])
        with col_chart:
            st.plotly_chart(emg_chart(emg_r), use_container_width=True,
                            config={"displayModeBar": True, "scrollZoom": True,
                                    "displaylogo": False})
        with col_feats:
            st.markdown('<p class="section-title" style="margin-top:0.3rem;">Motor-Unit Feature Profile</p>',
                        unsafe_allow_html=True)
            render_emg_features(emg_r)
            st.caption(f"Source: {emg_r.get('model_used', '—')}")

    with tab_fus:
        # Gauge + radar side by side
        g_col, r_col = st.columns(2)
        with g_col:
            st.plotly_chart(fusion_gauge(fus_r["risk_percent"], fus_r["risk_level"]),
                            use_container_width=True, config={"displayModeBar": False})
        with r_col:
            st.plotly_chart(cross_organ_radar(fus_r),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:1rem 0;'>",
                    unsafe_allow_html=True)

        # Cross-organ coupling indices
        st.markdown('<p class="section-title">Cross-Organ Coupling Indices</p>',
                    unsafe_allow_html=True)
        cross = fus_r.get("cross_organ_indices", {})
        ci1, ci2, ci3 = st.columns(3)
        for col, label, key in [
            (ci1, "Neuro-Cardiac Coupling",   "neuro_cardiac_coupling"),
            (ci2, "Neuro-Muscular Coupling",  "neuro_muscular_coupling"),
            (ci3, "Cardio-Muscular Coupling", "cardio_muscular_coupling"),
        ]:
            val = cross.get(key, 0.0)
            delta_txt = "Elevated" if val * 100 > 55 else "Within range"
            col.metric(label, f"{val*100:.1f}%", delta=delta_txt)

        st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:1rem 0;'>",
                    unsafe_allow_html=True)

        # Clinical alerts
        st.markdown('<p class="section-title">Clinical Alerts</p>', unsafe_allow_html=True)
        render_clinical_alerts(fus_r.get("clinical_alerts", []))

        st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:1rem 0;'>",
                    unsafe_allow_html=True)

        # Interpretation
        st.markdown('<p class="section-title">AI Interpretation</p>', unsafe_allow_html=True)
        level = fus_r["risk_level"]
        cls = {"HIGH": "interp-high", "MODERATE": "interp-moderate",
               "LOW": "interp-low"}.get(level, "interp-low")
        st.markdown(
            f'<div class="interp-box {cls}">{fus_r.get("interpretation","")}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        render_recommendations(fus_r.get("recommendations", []))

        # Footnote
        fw = fus_r.get("fusion_weights", {})
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            f"Fusion: risk = {fw.get('eeg',0.4):.0%} EEG + "
            f"{fw.get('ecg',0.4):.0%} ECG + {fw.get('emg',0.2):.0%} EMG  "
            f"|  Thresholds: 0–30% Low · 30–70% Moderate · 70–100% High"
        )

        col_reset, _ = st.columns([1, 6])
        with col_reset:
            if st.button("Reset", use_container_width=True):
                for k in ["eeg_result","ecg_result","emg_result","fusion_result","analysed","demo_mode"]:
                    st.session_state[k] = None if "result" in k else False
                st.rerun()

else:
    # ── Welcome state ──────────────────────────────────────────────────────
    st.markdown("""
<div class="welcome-card">
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="40" height="40" rx="8" fill="#EFF6FF"/>
    <path d="M8 24L14 16L18 22L22 12L28 20L32 17" stroke="#2563EB" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <h3>Ready for biosignal analysis</h3>
  <p>
    Upload EEG, ECG, and EMG signals via the sidebar, then click
    <strong>Run Analysis</strong>. Or use <strong>Demo</strong> for an instant
    run with synthetic physiological data.
  </p>
  <br>
  <div style="display:flex;justify-content:center;gap:0.75rem;flex-wrap:wrap;margin-top:0.5rem;">
    <span class="modality-pill">EEG &mdash; Seizure Detection</span>
    <span class="modality-pill">ECG &mdash; Arrhythmia Detection</span>
    <span class="modality-pill">EMG &mdash; Neuromuscular Patterns</span>
    <span class="modality-pill">Multimodal Fusion</span>
  </div>
</div>""", unsafe_allow_html=True)
