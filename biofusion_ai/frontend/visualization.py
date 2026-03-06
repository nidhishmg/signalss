"""
visualization.py — Clean Plotly signal charts for BioFusion AI
"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import plotly.graph_objects as go

# Shared design tokens
_FONT    = "Inter, -apple-system, sans-serif"
_BG      = "rgba(0,0,0,0)"
_PLOT_BG = "#FAFBFF"
_GRID    = "#E2E8F0"
_TEXT    = "#475569"
_BLUE    = "#2563EB"
_RED     = "#DC2626"
_AMBER   = "#D97706"
_GREEN   = "#16A34A"
_TEAL    = "#0D9488"
_GRAY    = "#94A3B8"


def _risk_color(value: float, invert: bool = False) -> str:
    v = 1.0 - value if invert else value
    if v >= 0.7: return _RED
    if v >= 0.3: return _AMBER
    return _GREEN


def _base(height: int = 280) -> dict:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(family=_FONT, size=11, color=_TEXT),
        margin=dict(l=50, r=16, t=36, b=36),
        height=height,
        hovermode="x unified",
        xaxis=dict(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            zeroline=False, tickfont=dict(size=10),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            zeroline=False, tickfont=dict(size=10),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
    )


def eeg_chart(result: Dict[str, Any]) -> go.Figure:
    raw  = result.get("raw_signal", [])
    fs   = result.get("fs", 256)
    sp   = result.get("seizure_probability", 0.0)

    raw_arr = np.array(raw if raw else [0.0], dtype=np.float32)
    t = np.arange(len(raw_arr)) / fs
    color = _risk_color(sp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=raw_arr,
        mode="lines",
        name="EEG",
        line=dict(color=color, width=0.9),
        hovertemplate="%{x:.3f}s &nbsp; %{y:.3f} µV<extra></extra>",
    ))
    fig.add_annotation(
        x=1, y=1, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"Seizure: <b>{sp*100:.1f}%</b>",
        showarrow=False,
        font=dict(size=11, color=color, family=_FONT),
        bgcolor="white", bordercolor=_GRID, borderwidth=1, borderpad=5,
    )
    layout = _base()
    layout["title"] = dict(text="EEG — Brain Electrical Activity", font=dict(size=13, color="#1E293B"), x=0)
    layout["xaxis"]["title"] = dict(text="Time (s)", font=dict(size=10))
    layout["yaxis"]["title"] = dict(text="µV", font=dict(size=10))
    fig.update_layout(**layout)
    return fig


def ecg_chart(result: Dict[str, Any]) -> go.Figure:
    raw     = result.get("raw_signal", [])
    fs      = result.get("fs", 360)
    ap      = result.get("arrhythmia_probability", 0.0)
    r_peaks = result.get("r_peaks", [])
    hr      = result.get("heart_rate", 0.0)

    raw_arr = np.array(raw if raw else [0.0], dtype=np.float32)
    t = np.arange(len(raw_arr)) / fs
    r_arr = [rp for rp in r_peaks if rp < len(raw_arr)]
    color = _risk_color(ap)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=raw_arr,
        mode="lines", name="ECG",
        line=dict(color=color, width=1.1),
        hovertemplate="%{x:.3f}s &nbsp; %{y:.4f} mV<extra></extra>",
    ))
    if r_arr:
        fig.add_trace(go.Scatter(
            x=[r / fs for r in r_arr],
            y=[raw_arr[r] for r in r_arr],
            mode="markers", name="R-peaks",
            marker=dict(symbol="triangle-up", size=7, color=_RED, opacity=0.85),
            hovertemplate="R-peak @%{x:.3f}s<extra></extra>",
        ))
    fig.add_annotation(
        x=1, y=1, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"HR {hr:.0f} bpm &nbsp;|&nbsp; Arrhythmia: <b>{ap*100:.1f}%</b>",
        showarrow=False,
        font=dict(size=11, color=color, family=_FONT),
        bgcolor="white", bordercolor=_GRID, borderwidth=1, borderpad=5,
    )
    layout = _base()
    layout["title"] = dict(text="ECG — Cardiac Electrical Activity", font=dict(size=13, color="#1E293B"), x=0)
    layout["xaxis"]["title"] = dict(text="Time (s)", font=dict(size=10))
    layout["yaxis"]["title"] = dict(text="mV", font=dict(size=10))
    fig.update_layout(**layout)
    return fig


def emg_chart(result: Dict[str, Any]) -> go.Figure:
    raw  = result.get("raw_signal", [])
    fs   = result.get("fs", 2000)
    mas  = result.get("muscle_abnormality_score", 0.0)
    mav  = result.get("mav", 0.0)

    raw_arr = np.array(raw if raw else [0.0], dtype=np.float32)
    t = np.arange(len(raw_arr)) / fs
    win = max(1, int(0.05 * fs))
    env = np.array([
        np.sqrt(np.mean(raw_arr[max(0, i - win): i + win + 1] ** 2))
        for i in range(len(raw_arr))
    ])
    color = _risk_color(mas)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=raw_arr,
        mode="lines", name="EMG signal",
        line=dict(color="#CBD5E1", width=0.7),
        hovertemplate="%{x:.4f}s &nbsp; %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=env,
        mode="lines", name="RMS envelope",
        line=dict(color=color, width=2.0),
        hovertemplate="RMS %{y:.4f}<extra></extra>",
    ))
    fig.add_annotation(
        x=1, y=1, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"MAV {mav:.4f} &nbsp;|&nbsp; Abnormality: <b>{mas*100:.1f}%</b>",
        showarrow=False,
        font=dict(size=11, color=color, family=_FONT),
        bgcolor="white", bordercolor=_GRID, borderwidth=1, borderpad=5,
    )
    layout = _base()
    layout["title"] = dict(text="EMG — Muscle Electrical Activity", font=dict(size=13, color="#1E293B"), x=0)
    layout["xaxis"]["title"] = dict(text="Time (s)", font=dict(size=10))
    layout["yaxis"]["title"] = dict(text="mV", font=dict(size=10))
    fig.update_layout(**layout)
    return fig


def fusion_gauge(risk_percent: float, risk_level: str) -> go.Figure:
    color = _RED if risk_level == "HIGH" else _AMBER if risk_level == "MODERATE" else _GREEN
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        number={"suffix": "%", "font": {"size": 40, "family": _FONT, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1, "tickcolor": _GRAY,
                      "tickfont": {"size": 10, "family": _FONT}},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#F0FDF4"},
                {"range": [30, 70], "color": "#FFFBEB"},
                {"range": [70, 100],"color": "#FEF2F2"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=24, r=24, t=16, b=8),
        paper_bgcolor=_BG,
        font=dict(family=_FONT),
        title=dict(
            text=f"Combined Risk &nbsp;&middot;&nbsp; <span style='color:{color};font-weight:700;'>{risk_level}</span>",
            font=dict(size=12, color="#1E293B"), x=0.5, xanchor="center",
        ),
    )
    return fig


def cross_organ_radar(fusion_result: Dict[str, Any]) -> go.Figure:
    comps = fusion_result.get("component_scores", {})
    cross = fusion_result.get("cross_organ_indices", {})

    categories = [
        "Seizure Risk", "Arrhythmia Risk", "Muscular Abnormality",
        "Neuro-Cardiac", "Neuro-Muscular", "Cardio-Muscular",
    ]
    values = [
        comps.get("eeg_seizure_probability", 0),
        comps.get("ecg_arrhythmia_probability", 0),
        comps.get("emg_abnormality_score", 0),
        cross.get("neuro_cardiac_coupling", 0),
        cross.get("neuro_muscular_coupling", 0),
        cross.get("cardio_muscular_coupling", 0),
    ]
    pct = [v * 100 for v in values]

    fig = go.Figure(go.Scatterpolar(
        r=pct + [pct[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(37,99,235,0.08)",
        line=dict(color=_BLUE, width=1.8),
        name="Risk Profile",
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, family=_FONT),
                            gridcolor=_GRID),
            angularaxis=dict(tickfont=dict(size=10, family=_FONT),
                             gridcolor=_GRID),
        ),
        showlegend=False,
        height=260,
        margin=dict(l=48, r=48, t=32, b=32),
        paper_bgcolor=_BG,
        font=dict(family=_FONT),
        title=dict(text="Cross-Organ Risk Profile",
                   font=dict(size=12, color="#1E293B"), x=0.5, xanchor="center"),
    )
    return fig
