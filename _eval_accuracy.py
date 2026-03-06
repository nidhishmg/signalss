"""
Evaluate accuracy of all three BioFusion AI models using 5-fold cross-validation
on held-out synthetic test sets.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'biofusion_ai'))

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# ── EEG ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EEG MODEL — RandomForest (seizure detection)")
print("=" * 60)

from modules.eeg_analysis import (
    _synthetic_eeg, _preprocess as eeg_prep,
    _segment as eeg_seg, _extract_features as eeg_feat,
    FS_DEFAULT as EEG_FS,
)

rng = np.random.default_rng(99)
X_eeg, y_eeg = [], []
for _ in range(200):
    for label, flag in [(0, False), (1, True)]:
        raw  = _synthetic_eeg(30, EEG_FS, flag, rng)
        proc = eeg_prep(raw, EEG_FS)
        for seg in eeg_seg(proc, EEG_FS):
            X_eeg.append(eeg_feat(seg, EEG_FS))
            y_eeg.append(label)

X_eeg = np.array(X_eeg);  y_eeg = np.array(y_eeg)
scaler = StandardScaler()
X_eeg_s = scaler.fit_transform(X_eeg)
clf_eeg = RandomForestClassifier(n_estimators=300, max_depth=12,
                                  class_weight="balanced", random_state=42)
cv_eeg = cross_val_score(clf_eeg, X_eeg_s, y_eeg, cv=5,
                          scoring="accuracy", n_jobs=-1)
clf_eeg.fit(X_eeg_s, y_eeg)
auc_eeg = cross_val_score(clf_eeg, X_eeg_s, y_eeg, cv=5,
                           scoring="roc_auc", n_jobs=-1)
print(f"  Accuracy (5-fold CV) : {cv_eeg.mean()*100:.2f}%  ± {cv_eeg.std()*100:.2f}%")
print(f"  ROC-AUC  (5-fold CV) : {auc_eeg.mean():.4f} ± {auc_eeg.std():.4f}")
print(f"  Per-fold accuracies  : {[round(v*100,1) for v in cv_eeg]}")

# ── ECG ──────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("ECG MODEL — XGBoost / RandomForest (arrhythmia detection)")
print("=" * 60)

from modules.ecg_analysis import (
    _synthetic_ecg, _preprocess as ecg_prep,
    _extract_features as ecg_feat,
    FS_DEFAULT as ECG_FS,
)
try:
    from xgboost import XGBClassifier
    clf_ecg = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             use_label_encoder=False, eval_metric="logloss",
                             random_state=42)
    ecg_name = "XGBoost"
except ImportError:
    clf_ecg = RandomForestClassifier(n_estimators=300, max_depth=12,
                                      class_weight="balanced", random_state=42)
    ecg_name = "RandomForest"

X_ecg, y_ecg = [], []
for _ in range(200):
    for label, flag in [(0, False), (1, True)]:
        raw  = _synthetic_ecg(60, ECG_FS, flag, rng)
        proc = ecg_prep(raw, ECG_FS)
        fv   = ecg_feat(proc, ECG_FS)
        if np.isfinite(fv).all():
            X_ecg.append(fv);  y_ecg.append(label)

X_ecg = np.array(X_ecg);  y_ecg = np.array(y_ecg)
scaler2 = StandardScaler()
X_ecg_s = scaler2.fit_transform(X_ecg)
cv_ecg  = cross_val_score(clf_ecg, X_ecg_s, y_ecg, cv=5,
                           scoring="accuracy", n_jobs=-1)
clf_ecg.fit(X_ecg_s, y_ecg)
auc_ecg = cross_val_score(clf_ecg, X_ecg_s, y_ecg, cv=5,
                           scoring="roc_auc", n_jobs=-1)
print(f"  Classifier           : {ecg_name}")
print(f"  Accuracy (5-fold CV) : {cv_ecg.mean()*100:.2f}%  ± {cv_ecg.std()*100:.2f}%")
print(f"  ROC-AUC  (5-fold CV) : {auc_ecg.mean():.4f} ± {auc_ecg.std():.4f}")
print(f"  Per-fold accuracies  : {[round(v*100,1) for v in cv_ecg]}")

# ── EMG ──────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("EMG MODEL — SVM (neuromuscular abnormality detection)")
print("=" * 60)

from modules.emg_analysis import (
    _synthetic_emg, _preprocess as emg_prep,
    _segment as emg_seg, _extract_features as emg_feat,
    FS_DEFAULT as EMG_FS,
)

X_emg, y_emg = [], []
for _ in range(150):
    for label, flag in [(0, False), (1, True)]:
        raw  = _synthetic_emg(10, EMG_FS, flag, rng)
        proc = emg_prep(raw, EMG_FS)
        for seg in emg_seg(proc, EMG_FS):
            fv = emg_feat(seg, EMG_FS)
            if np.isfinite(fv).all():
                X_emg.append(fv);  y_emg.append(label)

X_emg = np.array(X_emg);  y_emg = np.array(y_emg)
scaler3 = StandardScaler()
X_emg_s = scaler3.fit_transform(X_emg)
clf_emg = SVC(kernel="rbf", C=10, gamma="scale",
              probability=True, class_weight="balanced", random_state=42)
cv_emg  = cross_val_score(clf_emg, X_emg_s, y_emg, cv=5,
                           scoring="accuracy", n_jobs=-1)
clf_emg.fit(X_emg_s, y_emg)
auc_emg = cross_val_score(clf_emg, X_emg_s, y_emg, cv=5,
                           scoring="roc_auc", n_jobs=-1)
print(f"  Accuracy (5-fold CV) : {cv_emg.mean()*100:.2f}%  ± {cv_emg.std()*100:.2f}%")
print(f"  ROC-AUC  (5-fold CV) : {auc_emg.mean():.4f} ± {auc_emg.std():.4f}")
print(f"  Per-fold accuracies  : {[round(v*100,1) for v in cv_emg]}")

# ── Overall summary ───────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  EEG RandomForest     : {cv_eeg.mean()*100:.1f}%  (AUC {auc_eeg.mean():.3f})")
print(f"  ECG {ecg_name:<16}: {cv_ecg.mean()*100:.1f}%  (AUC {auc_ecg.mean():.3f})")
print(f"  EMG SVM              : {cv_emg.mean()*100:.1f}%  (AUC {auc_emg.mean():.3f})")
print()
print("NOTE: Trained on synthetic physiological data.")
print("Real-world accuracy will vary — validate with clinical datasets")
print("(CHB-MIT EEG, MIT-BIH ECG, NinaPro EMG) for production use.")
