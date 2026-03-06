import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'biofusion_ai'))

# Remove old .pt files to force retrain with new slim architecture
for f in ['models/eeg_model.pt', 'models/ecg_model.pt', 'models/emg_model.pt']:
    p = os.path.join('biofusion_ai', f)
    if os.path.isfile(p):
        os.remove(p)

print("=" * 50)
t0 = time.time()
print("Training EEGNet...")
from modules.eeg_analysis import analyze_eeg
eeg = analyze_eeg(deterministic_seed=42)
t1 = time.time()
print(f"  Done in {t1-t0:.1f}s | seizure_prob={eeg['seizure_probability']} | {eeg['model_used']}")

print("Training ResNet18-1D (base_channels=16)...")
from modules.ecg_analysis import analyze_ecg
ecg = analyze_ecg(deterministic_seed=43)
t2 = time.time()
print(f"  Done in {t2-t1:.1f}s | arrhythmia_prob={ecg['arrhythmia_probability']} | {ecg['model_used']}")

print("Training CNN-LSTM...")
from modules.emg_analysis import analyze_emg
emg = analyze_emg(deterministic_seed=44)
t3 = time.time()
print(f"  Done in {t3-t2:.1f}s | abnormality_score={emg['muscle_abnormality_score']} | {emg['model_used']}")

print("Fusion...")
from modules.fusion_engine import compute_risk
fusion = compute_risk(eeg, ecg, emg)
print(f"  risk_score={fusion['risk_score']} | level={fusion['risk_level']}")

print("=" * 50)
print(f"TOTAL TIME: {time.time()-t0:.1f}s")
print()
for f in ['models/eeg_model.pt', 'models/ecg_model.pt', 'models/emg_model.pt']:
    p = os.path.join('biofusion_ai', f)
    if os.path.isfile(p):
        size_mb = round(os.path.getsize(p)/1024/1024, 2)
        print(f"  {f}: {size_mb} MB saved")
