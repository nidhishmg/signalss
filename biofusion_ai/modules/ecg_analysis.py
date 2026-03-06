"""
ecg_analysis.py — ECG Signal Processing + ResNet18-1D Classifier

Pipeline:
  Load wfdb / CSV / synthetic
  → Baseline removal (high-pass 0.5 Hz)
  → Bandpass (0.5–40 Hz)
  → Window into 1-s segments (360 samples @ 360 Hz)
  → ResNet18-1D predicts arrhythmia probability per window
  → Average → arrhythmia_probability

HRV display features (for dashboard):
  heart rate, RR intervals, RMSSD, SDNN, LF/HF ratio, QRS duration
  (computed via scipy R-peak detection — separate from model)

Model saved to: models/ecg_model.pt
"""

from __future__ import annotations
import os, sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(_ROOT, "models", "ecg_model.pt")

# sys.path.insert(0, _ROOT)
from biofusion_ai.models.architectures.resnet1d import ResNet1D

FS_DEFAULT  = 360
WINDOW_SAMP = 360       # 1-s window at 360 Hz
OVERLAP     = 0.5
DEVICE      = torch.device("cpu")
torch.set_num_threads(16)   # use all available CPU cores


# ── Model cache ───────────────────────────────────────────────────────────────
class _Cache:
    model: Optional[ResNet1D] = None
    loaded: bool = False
_cache = _Cache()


def _load_or_train() -> None:
    if _cache.loaded:
        return
    net = ResNet1D(in_channels=1, num_classes=1, base_channels=16).to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(state)
        net.eval()
    else:
        _train(net)
    _cache.model = net
    _cache.loaded = True


# ── Synthetic ECG ─────────────────────────────────────────────────────────────
def _qrs_kernel(fs: int) -> np.ndarray:
    dur  = int(0.08 * fs)
    half = dur // 2
    k    = np.zeros(dur)
    k[:half]  = np.linspace(0, 1, half)
    k[half:]  = np.linspace(1, 0, dur - half)
    return k


def _synthetic_ecg(duration: float = 30.0, fs: int = FS_DEFAULT,
                   arrhythmia: bool = False,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n   = int(duration * fs)
    sig = np.zeros(n)
    qrs = _qrs_kernel(fs)
    t   = np.arange(n) / fs
    pos = 0
    while pos < n:
        rr = int((rng.uniform(0.35, 1.1) if arrhythmia
                  else rng.normal(0.80, 0.04)) * fs)
        p_pos = pos + rr - int(0.15 * fs)
        if 0 <= p_pos < n:
            pw = np.sin(np.linspace(0, np.pi, int(0.08 * fs))) * 0.15
            sig[p_pos: p_pos + len(pw)] += pw[:n - p_pos]
        q_pos = pos + rr - len(qrs) // 2
        if 0 <= q_pos < n:
            amp = 1.0 + (0.4 * rng.standard_normal() if arrhythmia else 0.0)
            sig[q_pos: q_pos + len(qrs)] += amp * qrs[:n - q_pos]
        t_pos = pos + rr + int(0.12 * fs)
        if 0 <= t_pos < n:
            tw = np.sin(np.linspace(0, np.pi, int(0.16 * fs))) * 0.3
            sig[t_pos: t_pos + len(tw)] += tw[:n - t_pos]
        pos += rr
    sig += 0.05 * rng.standard_normal(n)
    return sig.astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _preprocess(raw: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    nyq = fs / 2.0
    # High-pass for baseline removal
    sos_hp = sp_signal.butter(2, 0.5 / nyq, btype="high", output="sos")
    sig    = sp_signal.sosfiltfilt(sos_hp, raw)
    # Bandpass
    sos_bp = sp_signal.butter(4, [0.5 / nyq, 40.0 / nyq], btype="band", output="sos")
    sig    = sp_signal.sosfiltfilt(sos_bp, sig)
    return sig.astype(np.float32)


def _segment(sig: np.ndarray, win: int = WINDOW_SAMP,
             overlap: float = OVERLAP) -> List[np.ndarray]:
    step = int(win * (1 - overlap))
    return [sig[i: i + win] for i in range(0, len(sig) - win + 1, step)]


# ── R-peak detection (HRV display features only) ─────────────────────────────
def _detect_r_peaks(sig: np.ndarray, fs: int) -> np.ndarray:
    diff_sq = np.diff(sig) ** 2
    win     = int(0.15 * fs)
    mwa     = np.convolve(diff_sq, np.ones(win) / win, mode="same")
    thr     = 0.5 * np.max(mwa)
    peaks, _ = sp_signal.find_peaks(mwa, height=thr,
                                     distance=int(0.25 * fs))
    return peaks


def _hrv_features(sig: np.ndarray, fs: int) -> Dict[str, float]:
    r = _detect_r_peaks(sig, fs)
    if len(r) < 2:
        return {"heart_rate": 75.0, "rr_mean_ms": 800.0, "rr_std_ms": 20.0,
                "rmssd_ms": 25.0, "lf_hf_ratio": 1.5, "qrs_duration_ms": 80.0,
                "r_peaks_count": 0}
    rr_ms = np.diff(r) / fs * 1000.0
    hr    = 60_000.0 / (np.mean(rr_ms) + 1e-9)
    if len(rr_ms) >= 8:
        f, p = sp_signal.welch(rr_ms, fs=1.0, nperseg=min(8, len(rr_ms)))
        lf   = float(np.trapz(p[(f >= 0.04) & (f <= 0.15)]))
        hf   = float(np.trapz(p[(f > 0.15) & (f <= 0.40)]))
        lfhf = lf / (hf + 1e-9)
    else:
        lfhf = 1.5
    return {
        "heart_rate":      round(float(hr), 1),
        "rr_mean_ms":      round(float(np.mean(rr_ms)), 1),
        "rr_std_ms":       round(float(np.std(rr_ms)), 2),
        "rmssd_ms":        round(float(np.sqrt(np.mean(np.diff(rr_ms)**2))), 2),
        "lf_hf_ratio":     round(float(lfhf), 3),
        "qrs_duration_ms": round(float(np.mean(np.diff(r[:30])) / fs * 1000), 1),
        "r_peaks_count":   int(len(r)),
        "r_peaks":         r[:100].tolist(),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def _train(net: ResNet1D, epochs: int = 10) -> None:
    rng = np.random.default_rng(42)
    segs_list, labels_list = [], []

    for label, flag in [(0, False), (1, True)]:
        for _ in range(20):
            raw  = _synthetic_ecg(60, FS_DEFAULT, flag, rng)
            proc = _preprocess(raw, FS_DEFAULT)
            for s in _segment(proc, WINDOW_SAMP):
                segs_list.append(s)
                labels_list.append(label)

    X      = torch.tensor(np.array(segs_list), dtype=torch.float32)
    y      = torch.tensor(labels_list, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

    net.train()
    opt  = optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.BCELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()

    net.eval()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_ecg(
    signal_data: Optional[np.ndarray] = None,
    fs: int = FS_DEFAULT,
    file_path: Optional[str] = None,
    deterministic_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyse ECG with ResNet18-1D → arrhythmia_probability [0–1].

    Returns dict with keys: arrhythmia_probability, risk_level, heart_rate,
    rr_mean_ms, rr_std_ms, rmssd_ms, lf_hf_ratio, qrs_duration_ms,
    r_peaks_count, model_used.
    """
    _load_or_train()

    raw_signal: Optional[np.ndarray] = None
    model_used = "simulation"

    if file_path and os.path.isfile(str(file_path) + ".hea"):
        try:
            import wfdb
            rec        = wfdb.rdrecord(file_path)
            raw_signal = rec.p_signal[:, 0].astype(np.float32)
            fs         = int(rec.fs)
            model_used = "real-wfdb"
        except Exception:
            pass

    if raw_signal is None and file_path and str(file_path).endswith(".csv") and \
            os.path.isfile(file_path):
        try:
            import pandas as pd
            raw_signal = pd.read_csv(file_path).iloc[:, 0].values.astype(np.float32)
            model_used = "real-csv"
        except Exception:
            pass

    if raw_signal is None and signal_data is not None and len(signal_data) > 0:
        raw_signal = signal_data
        model_used = "provided-array"

    if raw_signal is None:
        rng        = np.random.default_rng(deterministic_seed)
        raw_signal = _synthetic_ecg(60, FS_DEFAULT, rng.random() < 0.35, rng)
        model_used = "simulation"
        fs         = FS_DEFAULT

    proc = _preprocess(raw_signal, fs)
    segs = _segment(proc, WINDOW_SAMP) or [proc[:WINDOW_SAMP]]

    probs: List[float] = []
    with torch.no_grad():
        for s in segs:
            w = np.zeros(WINDOW_SAMP, dtype=np.float32)
            w[:min(len(s), WINDOW_SAMP)] = s[:WINDOW_SAMP]
            probs.append(float(_cache.model(torch.tensor(w).unsqueeze(0)).item()))

    arrhythmia_probability = float(np.mean(probs))
    hrv = _hrv_features(proc, fs)

    risk_level = ("HIGH" if arrhythmia_probability >= 0.7 else
                  "MODERATE" if arrhythmia_probability >= 0.3 else "LOW")

    return {
        "arrhythmia_probability": round(arrhythmia_probability, 4),
        "risk_level":             risk_level,
        "model_used":             f"ResNet18-1D ({model_used})",
        "raw_signal":             raw_signal[:int(10 * fs)].tolist(),
        "fs":                     fs,
        **hrv,
    }
