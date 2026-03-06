import sys
sys.path.insert(0, 'biofusion_ai')

from modules.eeg_analysis  import analyze_eeg
from modules.ecg_analysis  import analyze_ecg
from modules.emg_analysis  import analyze_emg
from modules.fusion_engine import compute_risk

print("Running EEG (simulation)...")
eeg = analyze_eeg(deterministic_seed=42)
print("  seizure_probability:", eeg["seizure_probability"])
print("  risk_level:         ", eeg["risk_level"])

print("Running ECG (simulation)...")
ecg = analyze_ecg(deterministic_seed=43)
print("  arrhythmia_probability:", ecg["arrhythmia_probability"])
print("  heart_rate:            ", ecg["heart_rate"], "bpm")

print("Running EMG (simulation)...")
emg = analyze_emg(deterministic_seed=44)
print("  muscle_abnormality_score:", emg["muscle_abnormality_score"])

print("Running Fusion...")
fusion = compute_risk(eeg, ecg, emg)
print("  risk_score:  ", fusion["risk_score"])
print("  risk_level:  ", fusion["risk_level"])
print("  risk_percent:", fusion["risk_percent"])
print("  alerts:", fusion["clinical_alerts"])
print()
print("FULL PIPELINE OK")
