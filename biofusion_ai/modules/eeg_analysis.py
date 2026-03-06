"""
eeg_analysis.py — EEG Signal Processing + EEGNet Classifier

Pipeline:
  Load EDF (MNE) / 1-D array / synthetic signal
  → Bandpass filter (0.5–40 Hz)
  → Artifact removal
  → Normalise
  → Window into 2-s segments (512 samples @ 256 Hz)
  → EEGNet predicts seizure probability per window
  → Average across windows → seizure_probability

Display features (computed separately via scipy for the dashboard):
  band powers, spectral entropy, spike rate, wavelet energy

Model saved to: models/eeg_model.pt
"""

from __future__ import annotations
import os, sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import entropy as sp_entropy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(_ROOT, "models", "eeg_model.pt")

# sys.path.insert(0, _ROOT)
from biofusion_ai.models.architectures.eegnet import EEGNet

FS_DEFAULT   = 256
WINDOW_SAMP  = 512          # 2-s window at 256 Hz
OVERLAP      = 0.5
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
DEVICE = torch.device("cpu")
torch.set_num_threads(16)


# ── Model cache ───────────────────────────────────────────────────────────────
class _Cache:
    model: Optional[EEGNet] = None
    loaded: bool = False
_cache = _Cache()


def _load_or_train() -> None:
    if _cache.loaded:
        return
    net = EEGNet(in_channels=1, F1=8, D=2, F2=16,
                 kernel_length=64, dropout=0.5).to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(state)
        net.eval()
    else:
        _train(net)
    _cache.model = net
    _cache.loaded = True


# ── Synthetic EEG ─────────────────────────────────────────────────────────────
def _synthetic_eeg(duration: float = 30.0, fs: int = FS_DEFAULT,
                   seizure: bool = False,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    t   = np.linspace(0, duration, int(duration * fs))
    sig = np.zeros_like(t)
    if seizure:
        sig += 12.0 * np.sin(2 * np.pi * 3.0 * t)
        sig += 5.0  * np.sin(2 * np.pi * 1.5 * t)
        sig += 3.0  * rng.standard_normal(len(t))
    else:
        sig += 4.0 * np.sin(2 * np.pi * 10.0 * t)
        sig += 2.0 * np.sin(2 * np.pi * 20.0 * t)
        sig += 1.0 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _bandpass(sig: np.ndarray, lo: float, hi: float, fs: int) -> np.ndarray:
    nyq = fs / 2.0
    sos = sp_signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, sig).astype(np.float32)

def _preprocess(raw: np.ndarray, fs: int = FS_DEFAULT) -> np.ndarray:
    sig = _bandpass(raw, 0.5, 40.0, fs)
    thr = 6.0 * float(np.std(sig)) + 1e-9
    sig = np.clip(sig, -thr, thr)
    sig = (sig / (np.std(sig) + 1e-9)).astype(np.float32)
    return sig

def _segment(sig: np.ndarray, win: int = WINDOW_SAMP,
             overlap: float = OVERLAP) -> List[np.ndarray]:
    step = int(win * (1 - overlap))
    return [sig[i: i + win] for i in range(0, len(sig) - win + 1, step)]


# ── Display features (scipy — for dashboard metrics only) ────────────────────
def _band_power(seg: np.ndarray, fs: int, lo: float, hi: float) -> float:
    f, p = sp_signal.welch(seg, fs=fs, nperseg=min(256, len(seg)))
    idx  = (f >= lo) & (f <= hi)
    return float(np.trapz(p[idx], f[idx]))

def _spectral_entropy(seg: np.ndarray, fs: int) -> float:
    _, p = sp_signal.welch(seg, fs=fs, nperseg=min(256, len(seg)))
    pn   = p / (p.sum() + 1e-9)
    return float(sp_entropy(pn + 1e-12))

def _spike_rate(seg: np.ndarray, fs: int) -> float:
    thr    = 2.5 * float(np.std(seg))
    peaks, _ = sp_signal.find_peaks(np.abs(seg), height=thr,
                                     distance=int(0.1 * fs))
    return float(len(peaks) / (len(seg) / fs + 1e-9))


# ── Training ──────────────────────────────────────────────────────────────────
import glob

def _train(net: EEGNet, epochs: int = 10) -> None:
    rng = np.random.default_rng(42)
    segs_list, labels_list = [], []
    
    eeg_dir = os.path.join(_ROOT, "datasets", "eeg")
    edf_files = glob.glob(os.path.join(eeg_dir, "*.edf"))
    
    if edf_files:
        print(f"\\n[INFO] Training EEGNet on {len(edf_files)} REAL patient EDF records from CHB-MIT...")
        import mne
        for path in edf_files:
            basename = os.path.basename(path)
            # CHB-MIT chb01 annotated seizure intervals (in seconds)
            seizure_ranges = {
                "chb01_03.edf": (2996, 3036),
                "chb01_04.edf": (1467, 1494)
            }
            try:
                raw_mne = mne.io.read_raw_edf(path, preload=True, verbose=False)
                # Channel 0 is usually FP1-F7
                raw_sig = raw_mne.get_data()[0]
                fs_real = int(raw_mne.info["sfreq"])
                
                if basename in seizure_ranges:
                    label = 1
                    s_start, s_end = seizure_ranges[basename]
                    # Extract 1 minute before/after the seizure
                    extract_start = max(0, (s_start - 30) * fs_real)
                    extract_end = min(len(raw_sig), (s_end + 30) * fs_real)
                    raw_chunk = raw_sig[extract_start:extract_end]
                else:
                    label = 0
                    # Normal recording: take 2 minutes from the middle
                    mid = len(raw_sig) // 2
                    raw_chunk = raw_sig[mid:mid + 120 * fs_real]

                proc = _preprocess(raw_chunk, fs_real)
                for s in _segment(proc, WINDOW_SAMP):
                    w = np.zeros(WINDOW_SAMP, dtype=np.float32)
                    w[:min(len(s), WINDOW_SAMP)] = s[:WINDOW_SAMP]
                    segs_list.append(w)
                    labels_list.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    else:
        print("\\n[INFO] No real EDF data found. Training EEGNet on synthetic data...")
        for label, flag in [(0, False), (1, True)]:
            for _ in range(20):          # 20 synthetic signals per class
                raw  = _synthetic_eeg(30, FS_DEFAULT, flag, rng)
                proc = _preprocess(raw, FS_DEFAULT)
                for s in _segment(proc, WINDOW_SAMP):
                    segs_list.append(s)
                    labels_list.append(label)

    X = torch.tensor(np.array(segs_list), dtype=torch.float32)
    y = torch.tensor(labels_list, dtype=torch.float32)
    ds     = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    net.train()
    opt      = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            opt.step()

    net.eval()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    print("[INFO] EEGNet trained and saved.")


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_eeg(
    signal_data: Optional[np.ndarray] = None,
    fs: int = FS_DEFAULT,
    file_path: Optional[str] = None,
    deterministic_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyse EEG with EEGNet → seizure_probability [0–1].

    Returns dict with keys: seizure_probability, risk_level, band_powers,
    spectral_entropy, spike_rate, signal_quality, model_used.
    """
    _load_or_train()

    raw_signal: Optional[np.ndarray] = None
    model_used = "simulation"

    if file_path and os.path.isfile(str(file_path) + ".edf"):
        try:
            import mne
            raw     = mne.io.read_raw_edf(file_path + ".edf", preload=True, verbose=False)
            raw_signal = raw.get_data()[0]
            fs         = int(raw.info["sfreq"])
            model_used = "real-edf"
        except Exception:
            pass

    if raw_signal is None and signal_data is not None and len(signal_data) > 0:
        raw_signal = signal_data
        model_used = "provided-array"

    if raw_signal is None:
        rng       = np.random.default_rng(deterministic_seed)
        is_seiz   = rng.random() < 0.35
        raw_signal = _synthetic_eeg(30, FS_DEFAULT, is_seiz, rng)
        model_used = "simulation"
        fs         = FS_DEFAULT

    proc = _preprocess(raw_signal, fs)
    segs = _segment(proc, WINDOW_SAMP) or [proc[:WINDOW_SAMP]]

    # Pad/trim each window to WINDOW_SAMP and run EEGNet
    probs: List[float] = []
    with torch.no_grad():
        for s in segs:
            w = np.zeros(WINDOW_SAMP, dtype=np.float32)
            w[:min(len(s), WINDOW_SAMP)] = s[:WINDOW_SAMP]
            t = torch.tensor(w).unsqueeze(0)          # (1, 512)
            probs.append(float(_cache.model(t).item()))

    seizure_probability = float(np.mean(probs))

    # Display features on a representative window
    sample = segs[len(segs) // 2]
    band_powers = {k: round(_band_power(sample, fs, lo, hi), 4)
                   for k, (lo, hi) in BANDS.items()}

    risk_level = ("HIGH" if seizure_probability >= 0.7 else
                  "MODERATE" if seizure_probability >= 0.3 else "LOW")

    return {
        "seizure_probability": round(seizure_probability, 4),
        "risk_level":          risk_level,
        "band_powers":         band_powers,
        "spectral_entropy":    round(_spectral_entropy(sample, fs), 4),
        "spike_rate":          round(_spike_rate(sample, fs), 4),
        "signal_quality":      round(float(np.isfinite(raw_signal).mean()), 4),
        "model_used":          f"EEGNet ({model_used})",
        "raw_signal":          raw_signal[:int(10 * fs)].tolist(),
        "fs":                  fs,
    }
