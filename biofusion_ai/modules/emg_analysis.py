"""
emg_analysis.py — EMG Signal Processing + CNN-LSTM Classifier

Pipeline:
  Load CSV / 1-D array / synthetic
  → Bandpass (20–450 Hz)
  → Notch (50 Hz power-line)
  → Window into 0.5-s segments (1000 samples @ 2000 Hz)
  → CNN-LSTM predicts abnormality probability per window
  → Average → muscle_abnormality_score

Display features (scipy — for dashboard):
  MAV, RMS energy, waveform length, ZCR, SSC, median/mean frequency
  (computed separately, not used by model)

Model saved to: models/emg_model.pt
"""

from __future__ import annotations
import os, sys
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal as sp_signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, "models", "emg_model.pt")

# sys.path.insert(0, _ROOT)
from biofusion_ai.models.architectures.cnn_lstm import CNNLSTM

FS_DEFAULT  = 2000
WINDOW_SAMP = 1000      # 0.5-s window at 2000 Hz
OVERLAP     = 0.5
DEVICE      = torch.device("cpu")
torch.set_num_threads(16)


# ── Model cache ───────────────────────────────────────────────────────────────
class _Cache:
    model: Optional[CNNLSTM] = None
    loaded: bool = False
_cache = _Cache()


def _load_or_train() -> None:
    if _cache.loaded:
        return
    net = CNNLSTM(in_channels=1, hidden_size=128, num_layers=2,
                  bidirectional=True, dropout=0.3).to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(state)
        net.eval()
    else:
        _train(net)
    _cache.model = net
    _cache.loaded = True


# ── Synthetic EMG ─────────────────────────────────────────────────────────────
def _synthetic_emg(duration: float = 10.0, fs: int = FS_DEFAULT,
                   abnormal: bool = False,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n   = int(duration * fs)
    sig = np.zeros(n)
    t   = np.arange(n) / fs

    burst_rate = rng.uniform(2.0, 5.0)  if abnormal else rng.uniform(8.0, 15.0)
    burst_amp  = rng.uniform(0.3, 0.6)  if abnormal else rng.uniform(0.8, 1.2)
    burst_dur  = int(0.08 * fs)         if abnormal else int(0.03 * fs)

    step = int(fs / burst_rate)
    for pos in range(0, n, step):
        jitter = int(rng.uniform(-step * 0.1, step * 0.1))
        start  = max(0, pos + jitter)
        end    = min(n, start + burst_dur)
        if start >= n:
            continue
        carrier  = np.sin(2 * np.pi * rng.uniform(120, 320) * t[start:end])
        env      = np.hanning(end - start)
        sig[start:end] += burst_amp * carrier * env

    sig += 0.05 * rng.standard_normal(n)
    return sig.astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _preprocess(raw: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    nyq  = fs / 2.0
    hi   = min(450.0, nyq - 1.0)
    sos  = sp_signal.butter(4, [20.0 / nyq, hi / nyq], btype="band", output="sos")
    sig  = sp_signal.sosfiltfilt(sos, raw)
    b, a = sp_signal.iirnotch(50.0, 30.0, fs)
    sig  = sp_signal.lfilter(b, a, sig)
    return sig.astype(np.float32)


def _segment(sig: np.ndarray, win: int = WINDOW_SAMP,
             overlap: float = OVERLAP) -> List[np.ndarray]:
    step = int(win * (1 - overlap))
    return [sig[i: i + win] for i in range(0, len(sig) - win + 1, step)]


# ── Display features (scipy — dashboard only) ─────────────────────────────────
def _display_features(sig: np.ndarray, fs: int) -> Dict[str, float]:
    def mav(s): return float(np.mean(np.abs(s)))
    def rms(s): return float(np.sqrt(np.mean(s ** 2)))
    def wl(s):  return float(np.sum(np.abs(np.diff(s))))
    def zcr(s): return float(np.sum(np.diff(np.sign(s)) != 0)) / len(s)
    def ssc(s):
        d = np.diff(s)
        return float(np.sum(np.diff(np.sign(d)) != 0)) / max(1, len(d))
    def mnf(s):
        f, p = sp_signal.welch(s, fs=fs, nperseg=min(256, len(s)))
        return float(np.sum(f * p) / (np.sum(p) + 1e-9))
    def mdf(s):
        f, p = sp_signal.welch(s, fs=fs, nperseg=min(256, len(s)))
        cp   = np.cumsum(p)
        idx  = np.searchsorted(cp, cp[-1] / 2.0)
        return float(f[min(idx, len(f)-1)])

    sample = sig[:WINDOW_SAMP] if len(sig) >= WINDOW_SAMP else sig
    return {
        "mav":                round(mav(sample), 4),
        "rms_energy":         round(rms(sample), 4),
        "waveform_length":    round(wl(sample), 4),
        "zero_crossing_rate": round(zcr(sample), 4),
        "slope_sign_changes": round(ssc(sample), 4),
        "median_frequency_hz":round(mdf(sample), 2),
        "mean_frequency_hz":  round(mnf(sample), 2),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def _train(net: CNNLSTM, epochs: int = 10) -> None:
    rng = np.random.default_rng(42)
    segs_list, labels_list = [], []

    emg_dir = os.path.join(_ROOT, "datasets", "emg")
    
    if os.path.isfile(os.path.join(emg_dir, "emg_healthy.hea")):
        print("\\n[INFO] Training CNN-LSTM on REAL clinical EMG records from PhysioNet...")
        import wfdb
        
        records = [
            ("emg_healthy", 0),
            ("emg_myopathy", 1),
            ("emg_neuropathy", 1)
        ]
        
        for rec_name, label in records:
            try:
                rec_path = os.path.join(emg_dir, rec_name)
                record = wfdb.rdrecord(rec_path)
                raw_sig = record.p_signal[:, 0].astype(np.float32)
                fs_real = int(record.fs)
                
                proc = _preprocess(raw_sig, fs_real)
                for s in _segment(proc, WINDOW_SAMP):
                    w = np.zeros(WINDOW_SAMP, dtype=np.float32)
                    w[:min(len(s), WINDOW_SAMP)] = s[:WINDOW_SAMP]
                    if np.isfinite(w).all():
                        segs_list.append(w)
                        labels_list.append(label)
            except Exception as e:
                print(f"Skipping {rec_name}: {e}")
                
    else:
        print("\\n[INFO] No real EMG data found. Training CNN-LSTM on synthetic data...")
        for label, flag in [(0, False), (1, True)]:
            for _ in range(20):
                raw  = _synthetic_emg(10, FS_DEFAULT, flag, rng)
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
    print("[INFO] CNN-LSTM trained and saved.")


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_emg(
    signal_data: Optional[np.ndarray] = None,
    fs: int = FS_DEFAULT,
    file_path: Optional[str] = None,
    deterministic_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyse EMG with CNN-LSTM → muscle_abnormality_score [0–1].

    Returns dict with keys: muscle_abnormality_score, risk_level, mav,
    rms_energy, waveform_length, zero_crossing_rate, slope_sign_changes,
    median_frequency_hz, mean_frequency_hz, model_used.
    """
    _load_or_train()

    raw_signal: Optional[np.ndarray] = None
    model_used = "simulation"

    if file_path and os.path.isfile(str(file_path)):
        try:
            import pandas as pd
            raw_signal = pd.read_csv(file_path).iloc[:, 0].values.astype(np.float32)
            model_used = "real-csv"
        except Exception:
            try:
                raw_signal = np.loadtxt(file_path, delimiter=",",
                                         skiprows=1, usecols=0).astype(np.float32)
                model_used = "real-csv"
            except Exception:
                pass

    if raw_signal is None and signal_data is not None and len(signal_data) > 0:
        raw_signal = signal_data
        model_used = "provided-array"

    if raw_signal is None:
        rng        = np.random.default_rng(deterministic_seed)
        raw_signal = _synthetic_emg(10, FS_DEFAULT, rng.random() < 0.35, rng)
        model_used = "simulation"
        fs         = FS_DEFAULT

    proc = _preprocess(raw_signal, fs)
    segs = _segment(proc, WINDOW_SAMP) or [proc[:WINDOW_SAMP]]

    probs: List[float] = []
    with torch.no_grad():
        for s in segs:
            w = np.zeros(WINDOW_SAMP, dtype=np.float32)
            w[:min(len(s), WINDOW_SAMP)] = s[:WINDOW_SAMP]
            if not np.isfinite(w).all():
                continue
            probs.append(float(_cache.model(torch.tensor(w).unsqueeze(0)).item()))

    if not probs:
        probs = [0.1]
    muscle_abnormality_score = float(np.mean(probs))
    display = _display_features(proc, fs)

    risk_level = ("HIGH" if muscle_abnormality_score >= 0.7 else
                  "MODERATE" if muscle_abnormality_score >= 0.3 else "LOW")

    return {
        "muscle_abnormality_score": round(muscle_abnormality_score, 4),
        "risk_level":               risk_level,
        "model_used":               f"CNN-LSTM ({model_used})",
        "raw_signal":               raw_signal[:int(2 * fs)].tolist(),
        "fs":                       fs,
        **display,
    }
