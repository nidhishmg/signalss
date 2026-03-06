"""
fusion_engine.py — Multimodal Biosignal Fusion Engine

Combines predictions from EEG, ECG, EMG modules using weighted fusion.

Risk formula:
  risk_score = 0.40 * seizure_probability
             + 0.40 * arrhythmia_probability
             + 0.20 * muscle_abnormality_score

Thresholds:
  0.00 – 0.30 → LOW
  0.30 – 0.70 → MODERATE
  0.70 – 1.00 → HIGH

Additional cross-organ analysis:
  • Neuro-cardiac coupling index
  • Neuro-muscular coupling index
  • General systemic risk narrative
"""

from __future__ import annotations

from typing import Any, Dict, List

# Fusion weights (must sum to 1.0)
_W_EEG = 0.40
_W_ECG = 0.40
_W_EMG = 0.20

_LOW_THRESHOLD      = 0.30
_MODERATE_THRESHOLD = 0.70

# Clinical cross-organ risk multipliers
_NEURO_CARDIAC_THRESHOLD  = 0.55   # both EEG + ECG high → elevated SUDEP risk
_NEURO_MUSCULAR_THRESHOLD = 0.50   # both EEG + EMG high → ictal-motor coupling


def compute_risk(
    eeg_result: Dict[str, Any],
    ecg_result: Dict[str, Any],
    emg_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Fuse three signal-domain predictions into a unified risk assessment.

    Parameters
    ----------
    eeg_result : dict from eeg_analysis.analyze_eeg()
    ecg_result : dict from ecg_analysis.analyze_ecg()
    emg_result : dict from emg_analysis.analyze_emg()

    Returns
    -------
    dict
        risk_score, risk_level, component_scores, cross_organ_indices,
        clinical_alerts, interpretation, recommendations.
    """
    sp  = float(eeg_result.get("seizure_probability",      0.0))
    ap  = float(ecg_result.get("arrhythmia_probability",   0.0))
    mas = float(emg_result.get("muscle_abnormality_score", 0.0))

    # Weighted linear fusion
    risk_score = _W_EEG * sp + _W_ECG * ap + _W_EMG * mas
    risk_score = max(0.0, min(1.0, risk_score))

    risk_level = (
        "HIGH"     if risk_score >= _MODERATE_THRESHOLD
        else "MODERATE" if risk_score >= _LOW_THRESHOLD
        else "LOW"
    )

    # ── Cross-organ coupling indices ──────────────────────────────────────
    neuro_cardiac   = (sp + ap) / 2.0   # Neuro-Cardiac Coupling (SUDEP risk proxy)
    neuro_muscular  = (sp + mas) / 2.0  # Ictal-Motor Coupling
    cardio_muscular = (ap + mas) / 2.0  # Cardiac-Muscular Stress

    # ── Clinical alerts ───────────────────────────────────────────────────
    alerts: List[str] = []

    if sp >= 0.7:
        alerts.append("⚠️ HIGH SEIZURE RISK — Ictal EEG pattern detected.")
    if ap >= 0.7:
        alerts.append("⚠️ HIGH ARRHYTHMIA RISK — Severe RR irregularity detected.")
    if mas >= 0.7:
        alerts.append("⚠️ HIGH MUSCULAR ABNORMALITY — Reduced MU firing rate pattern.")

    if neuro_cardiac >= _NEURO_CARDIAC_THRESHOLD:
        alerts.append(
            "🔴 SUDEP RISK FLAG — Simultaneous neuro-cardiac instability detected. "
            "Urgent cardiology + neurology review recommended."
        )
    if neuro_muscular >= _NEURO_MUSCULAR_THRESHOLD:
        alerts.append(
            "🟠 Ictal-Motor Coupling detected — possible tonic/clonic seizure propagation."
        )
    if sp >= 0.5 and ap >= 0.5 and mas >= 0.5:
        alerts.append(
            "🔴 MULTI-ORGAN INSTABILITY — All three modalities show concurrent abnormal patterns. "
            "Immediate clinical review required."
        )

    if not alerts:
        alerts.append("✅ No critical cross-organ alerts at this time.")

    # ── Interpretation text ───────────────────────────────────────────────
    if risk_level == "HIGH":
        interpretation = (
            "The multimodal fusion engine has identified significant abnormalities across "
            "two or more biosignal domains. The patient exhibits high-risk patterns consistent "
            "with concurrent neuro-cardiac or neuro-muscular pathology. Immediate specialist "
            "review is clinically indicated."
        )
        recommendations = [
            "Urgent neurology and cardiology consultation",
            "Continuous ECG + EEG monitoring (ICU/HDU)",
            "Anti-epileptic therapy review if applicable",
            "Cardiac rhythm management assessment",
            "Neuromuscular evaluation (EMG/NCS formal study)",
        ]
    elif risk_level == "MODERATE":
        interpretation = (
            "Moderate biosignal deviations detected across one or more modalities. "
            "The pattern may represent early-stage physiological stress, medication side effects, "
            "or sub-clinical disease progression. Clinical correlation with patient history "
            "is strongly recommended."
        )
        recommendations = [
            "Outpatient neurology review within 1 week",
            "24-hour Holter monitoring for cardiac rhythm",
            "Repeat biosignal scan in 48–72 hours",
            "Review current medications for CNS/cardiac effects",
            "Patient education on warning signs",
        ]
    else:
        interpretation = (
            "All three biosignal domains show patterns within expected physiological ranges. "
            "No critical abnormalities detected at this time. Routine monitoring is appropriate."
        )
        recommendations = [
            "Continue routine clinical follow-up",
            "Repeat screening in 3–6 months",
            "Maintain cardiovascular health practices",
            "Document baseline for longitudinal comparison",
        ]

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "risk_percent": round(risk_score * 100, 2),
        "component_scores": {
            "eeg_seizure_probability":    round(sp, 4),
            "ecg_arrhythmia_probability": round(ap, 4),
            "emg_abnormality_score":      round(mas, 4),
        },
        "fusion_weights": {
            "eeg": _W_EEG,
            "ecg": _W_ECG,
            "emg": _W_EMG,
        },
        "cross_organ_indices": {
            "neuro_cardiac_coupling":   round(neuro_cardiac, 4),
            "neuro_muscular_coupling":  round(neuro_muscular, 4),
            "cardio_muscular_coupling": round(cardio_muscular, 4),
        },
        "clinical_alerts": alerts,
        "interpretation": interpretation,
        "recommendations": recommendations,
    }
