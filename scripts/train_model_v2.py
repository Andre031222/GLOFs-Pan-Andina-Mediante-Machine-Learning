#!/usr/bin/env python3
"""
scripts/train_model_v2.py
-------------------------
Reentrenamiento del modelo de susceptibilidad GLOF usando training_data_v2.csv
(n+=20, buffer graduado 5km/10km para eventos historicos).

Genera los mismos archivos de salida que NB15 (modelos, OOF, LOCO, calibracion)
de modo que todos los scripts generate_fig*.py y compute_jackknife.py funcionen
sin cambios.

Uso:
    glof/Scripts/python scripts/train_model_v2.py 2>&1 | tee logs/train_v2_standalone.log
"""

import os, sys, json, time, warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5

# GPU detection
USE_GPU = False
try:
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=3 * 1024**3)
    USE_GPU = True
    print('[OK] GPU: RTX 3050')
except Exception:
    print('[INFO] CPU mode')

XGB_DEVICE  = 'cuda' if USE_GPU else 'cpu'
LGBM_DEVICE = 'gpu'  if USE_GPU else 'cpu'

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
print('=' * 70)
print('GLOF MODEL TRAINING v2 — n+=20')
print(f'Inicio: {datetime.now():%Y-%m-%d %H:%M:%S}')
print('=' * 70)

training_path = DATA_DIR / 'labeled' / 'training_data_v2.csv'
df = pd.read_csv(training_path)
if 'had_glof' in df.columns and 'glof_label' not in df.columns:
    df = df.rename(columns={'had_glof': 'glof_label'})

area_to_country = {
    'cordillera_blanca'      : 'Peru',
    'cordillera_vilcanota'   : 'Peru',
    'cordillera_huayhuash'   : 'Peru',
    'cordillera_raura'       : 'Peru',
    'cordillera_central'     : 'Peru',
    'cordillera_huanzo'      : 'Peru',
    'cordillera_urubamba'    : 'Peru',
    'carabaya'               : 'Peru',
    'bolivia_cordillera_real': 'Bolivia',
    'apolobamba'             : 'Bolivia',
    'ecuador_antisana'       : 'Ecuador',
    'chile_andes_centrales'  : 'Chile',
    'patagonia_norte'        : 'Chile',
    'patagonia_sur'          : 'Chile',
}
if 'country' not in df.columns:
    df['country'] = df['area_name'].map(area_to_country).fillna('Unknown')

print(f'Total lagunas  : {len(df):,}')
print(f'Con GLOF (pos) : {int(df["glof_label"].sum()):,}')
print(f'Sin GLOF (neg) : {int((df["glof_label"]==0).sum()):,}')
ratio = (1 - df["glof_label"].mean()) / df["glof_label"].mean()
print(f'Ratio desbalance: 1:{ratio:.0f}')
print()
cs = df.groupby('country')['glof_label'].agg(['sum','count'])
cs.columns = ['GLOFs','Total']
print(cs.to_string())

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
exclude_cols = [
    'lake_id', 'glof_label', 'country', 'cordillera',
    'geometry', 'year', 'source', 'area_name', 'scene_date',
    'glof_volume_m3', 'glof_match_dist_m',
    'glof_date', 'glof_trigger', 'glof_lake_name',
]
feature_cols = [c for c in df.columns
                if c not in exclude_cols
                and df[c].dtype in ['float64','float32','int64','int32']]

X = df[feature_cols].copy()
y = df['glof_label'].astype(int)

X = X.fillna(X.median(numeric_only=True)).fillna(0)

# Log transforms
log_cols = [c for c in ['area_m2','perimeter_m','volume_m3','equivalent_diameter_m',
                         'potential_energy','depth_m','dam_height',
                         'area_2017_est','area_2025_est','total_growth_m2'] if c in X.columns]
for c in log_cols:
    X[f'log_{c}'] = np.log1p(X[c].clip(lower=0))

if 'total_growth_m2' in X.columns and 'area_2017_est' in X.columns:
    X['rel_growth'] = X['total_growth_m2'] / (X['area_2017_est'] + 1)
if 'slope_mean' in X.columns and 'log_area_m2' in X.columns:
    X['slope_x_logarea'] = X['slope_mean'] * X['log_area_m2']
if 'dist_glacier_m' in df.columns:
    X['no_glacier'] = (df['dist_glacier_m'].isna()).astype(int)

feature_cols = list(X.columns)
nan_count = X.isna().sum().sum()
print(f'\nFeatures totales: {len(feature_cols)} | NaN restantes: {nan_count}')

groups = df['country'].values
n_pos  = int(y.sum())
n_neg  = int((y == 0).sum())
scale_pos_weight = n_neg / n_pos
print(f'n+={n_pos} n-={n_neg} scale_pos_weight={scale_pos_weight:.0f}')

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

# =============================================================================
# 3. SMOTE + MODELOS
# =============================================================================
smote_def = SMOTETomek(
    smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, n_pos - 1)),
    random_state=RANDOM_STATE
)

models = {
    'BalancedRF': BalancedRandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        sampling_strategy='auto', replacement=True,
        n_jobs=-1, random_state=RANDOM_STATE,
    ),
    'EasyEnsemble': EasyEnsembleClassifier(
        n_estimators=50, sampling_strategy='auto',
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE,
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        device=XGB_DEVICE, eval_metric='auc',
        random_state=RANDOM_STATE, verbosity=0,
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        device=LGBM_DEVICE,
        random_state=RANDOM_STATE, verbosity=-1,
    ),
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=2000,
        C=0.01, solver='lbfgs', random_state=RANDOM_STATE,
    ),
}

NO_SMOTE_MODELS = {'BalancedRF', 'EasyEnsemble'}

# =============================================================================
# 4. OOF CROSS-VALIDATION
# =============================================================================
def optimal_threshold(y_true, y_prob, method='youden'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results     = {}
results_oof = {}

for name, mdl in models.items():
    print(f'\n--- {name} ---')
    t0 = time.time()

    if name in NO_SMOTE_MODELS:
        X_use, y_use = X.values, y.values
    else:
        X_use, y_use = X_sc, y.values

    y_prob_oof = np.zeros(len(y))

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_use, y_use)):
        X_tr, X_va = X_use[train_idx], X_use[val_idx]
        y_tr, y_va = y_use[train_idx], y_use[val_idx]

        if name not in NO_SMOTE_MODELS and n_pos >= 2:
            try:
                X_tr, y_tr = smote_def.fit_resample(X_tr, y_tr)
            except Exception as e:
                print(f'  SMOTE fold {fold_i} skipped: {e}')

        clone_mdl = joblib.load.__class__  # trick: we need to clone
        import copy
        m = copy.deepcopy(mdl)
        m.fit(X_tr, y_tr)
        y_prob_oof[val_idx] = m.predict_proba(X_va)[:, 1]

    auc  = roc_auc_score(y_use, y_prob_oof)
    pr   = average_precision_score(y_use, y_prob_oof)
    t_yo = optimal_threshold(y_use, y_prob_oof)
    y_pred = (y_prob_oof >= t_yo).astype(int)
    mcc  = matthews_corrcoef(y_use, y_pred)
    f1   = f1_score(y_use, y_pred, zero_division=0)

    # Lift at Youden threshold
    flagged_frac = float(y_pred.mean())
    lift = (float(y_pred[y_use==1].mean()) / flagged_frac) if flagged_frac > 0 else 0.0

    elapsed = time.time() - t0
    print(f'  AUC={auc:.4f} PR={pr:.4f} Lift={lift:.1f}x MCC={mcc:.4f} F1={f1:.4f} ({elapsed:.0f}s)')

    results[name] = {
        'roc_auc': auc, 'pr_auc': pr, 'lift': lift,
        'mcc': mcc, 'f1': f1,
        'threshold_youden': t_yo,
        'flagged_frac': flagged_frac,
    }
    results_oof[name] = {
        'y_true': y_use.tolist(),
        'y_prob_oof': y_prob_oof.tolist(),
    }

# =============================================================================
# 5. BEST MODEL SELECTION + FINAL TRAINING
# =============================================================================
# Sort by ROC-AUC
best_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f'\nBest by AUC: {best_name} ({results[best_name]["roc_auc"]:.4f})')

# BRF override: if BRF has good Lift, prefer it
brf_auc  = results.get('BalancedRF', {}).get('roc_auc', 0)
brf_lift = results.get('BalancedRF', {}).get('lift', 0)
if brf_lift >= 5.0:
    best_name = 'BalancedRF'
    print(f'Override: BalancedRF selected (Lift={brf_lift:.1f}x)')

# Train final model on full dataset
print(f'\nTraining final model: {best_name} on full dataset...')
if best_name in NO_SMOTE_MODELS:
    X_final, y_final = X.values, y.values
else:
    X_final, y_final = X_sc, y.values
    try:
        X_final, y_final = smote_def.fit_resample(X_final, y_final)
    except Exception as e:
        print(f'  SMOTE skipped: {e}')
        X_final, y_final = X_sc, y.values

import copy
best_mdl = copy.deepcopy(models[best_name])
best_mdl.fit(X_final, y_final)

# Calibration
best_oof = results_oof[best_name]['y_prob_oof']
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(best_oof, y.values)
best_oof_cal = ir.transform(best_oof)

# =============================================================================
# 6. SAVE MODELS + OOF + COMPARISON CSV
# =============================================================================
joblib.dump(best_mdl, MODELS_DIR / 'best_model.joblib')
joblib.dump(best_mdl, MODELS_DIR / 'balancedrf.joblib')
joblib.dump(scaler,   MODELS_DIR / 'scaler.joblib')
joblib.dump(ir,       MODELS_DIR / 'best_model_calibrated.joblib')

# Save all individual models
for nm, mdl_def in models.items():
    key = nm.lower().replace(' ','_').replace(' ','_')
    fname = {'balancedrf':'balancedrf','easyensemble':'easyensemble',
             'random_forest':'random_forest','xgboost':'xgboost',
             'lightgbm':'lightgbm','logistic_regression':'logistic_regression'}.get(key, key)
    try:
        import copy
        m_final = copy.deepcopy(mdl_def)
        if nm in NO_SMOTE_MODELS:
            m_final.fit(X.values, y.values)
        else:
            Xf, yf = X_sc, y.values
            try:
                Xf, yf = smote_def.fit_resample(Xf, yf)
            except Exception:
                pass
            m_final.fit(Xf, yf)
        joblib.dump(m_final, MODELS_DIR / f'{fname}.joblib')
    except Exception as e:
        print(f'  Could not save {nm}: {e}')

# OOF predictions (for jackknife + figures)
best_oof_arr = np.array(results_oof[best_name]['y_prob_oof'])
lgbm_oof = np.array(results_oof.get('LightGBM', results_oof[best_name])['y_prob_oof'])
oof_out = {
    'y_true':          y.tolist(),
    'y_score_brf':     best_oof_arr.tolist(),
    'y_score_lgbm':    lgbm_oof.tolist(),
    'y_score_cal':     best_oof_cal.tolist(),
}
with open(MODELS_DIR / 'oof_predictions.json', 'w') as f:
    json.dump(oof_out, f)

# Calibrated OOF
cal_out = {
    'y_true':           y.tolist(),
    'y_score_raw':      best_oof_arr.tolist(),
    'y_score_isotonic': best_oof_cal.tolist(),
}
with open(MODELS_DIR / 'calibrated_oof.json', 'w') as f:
    json.dump(cal_out, f)

# Feature names
with open(MODELS_DIR / 'feature_names.json', 'w') as f:
    json.dump(feature_cols, f)

# Model comparison CSV
rows = []
for nm, r in sorted(results.items(), key=lambda x: -x[1]['roc_auc']):
    rows.append({
        'Modelo': nm,
        'ROC-AUC (OOF)': round(r['roc_auc'], 4),
        'PR-AUC  (OOF)': round(r['pr_auc'], 4),
        'LiftxRandom':   f"{r['lift']:.1f}x",
        'MCC (OOF)':     round(r['mcc'], 4),
        'F1  (OOF)':     round(r['f1'], 4),
    })
pd.DataFrame(rows).to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
print('\nModel comparison:')
for r in rows:
    print(f"  {r['Modelo']:<22} AUC={r['ROC-AUC (OOF)']:.4f} Lift={r['LiftxRandom']}")

# Best model results for robustness script compatibility
best_r = results[best_name]
t_yo   = best_r['threshold_youden']
y_pred_final = (best_oof_arr >= t_yo).astype(int)
recall_at_youden = float(y_pred_final[y.values==1].mean())
flagged = int(y_pred_final.sum())

# Optimal thresholds
thresholds_out = {
    best_name: {'youden': t_yo, 'flagged': flagged,
                'flagged_frac': float(flagged / len(y)),
                'recall_at_youden': recall_at_youden}
}
with open(MODELS_DIR / 'optimal_thresholds.json', 'w') as f:
    json.dump(thresholds_out, f)

print(f'\nYouden threshold: {t_yo:.4f}')
print(f'Flagged: {flagged}/{len(y)} ({100*flagged/len(y):.1f}%)')
print(f'Recall at Youden: {recall_at_youden*100:.0f}%')

# =============================================================================
# 7. LOCO-CV
# =============================================================================
print('\n' + '=' * 70)
print('LOCO-CV (Leave-One-Country-Out)')
print('=' * 70)

logo = LeaveOneGroupOut()
loco_results = []

for train_idx, test_idx in logo.split(X.values, y.values, groups):
    country_test = groups[test_idx[0]]
    X_tr_loco = X.values[train_idx]
    y_tr_loco = y.values[train_idx]
    X_te_loco = X.values[test_idx]
    y_te_loco = y.values[test_idx]

    n_pos_fold = int(y_tr_loco.sum())
    if n_pos_fold < 2:
        print(f'  [{country_test}] Skip: only {n_pos_fold} positive(s) in train')
        loco_results.append({
            'country': country_test, 'n_test': len(test_idx),
            'n_pos_test': int(y_te_loco.sum()),
            'n_pos_train': n_pos_fold,
            'auc': None, 'note': 'insufficient_train_pos'
        })
        continue

    import copy
    m = copy.deepcopy(models['BalancedRF'])
    m.fit(X_tr_loco, y_tr_loco)
    y_prob_te = m.predict_proba(X_te_loco)[:, 1]
    n_pos_test = int(y_te_loco.sum())

    if n_pos_test < 2:
        # Can't compute AUC; report FPR at threshold
        fpr_val = float((y_prob_te[y_te_loco==0] >= t_yo).mean())
        print(f'  [{country_test}] n_test={len(test_idx)} n+_test={n_pos_test} '
              f'FPR@{t_yo:.3f}={fpr_val*100:.1f}%  (null-class specificity domain)')
        loco_results.append({
            'country': country_test, 'n_test': len(test_idx),
            'n_pos_test': n_pos_test, 'n_pos_train': n_pos_fold,
            'auc': None, 'fpr_at_threshold': fpr_val,
            'note': 'null_class'
        })
    else:
        auc = roc_auc_score(y_te_loco, y_prob_te)
        print(f'  [{country_test}] n_test={len(test_idx)} n+_test={n_pos_test} AUC={auc:.4f}')
        loco_results.append({
            'country': country_test, 'n_test': len(test_idx),
            'n_pos_test': n_pos_test, 'n_pos_train': n_pos_fold,
            'auc': auc, 'note': ''
        })

pd.DataFrame(loco_results).to_csv(MODELS_DIR / 'loco_validation.csv', index=False)

# =============================================================================
# 8. BOOTSTRAP CI
# =============================================================================
print('\nBootstrap CI (2000 resamples)...')
np.random.seed(RANDOM_STATE)
auc_boots = []
y_arr = y.values
for _ in range(2000):
    idx = np.random.choice(len(y_arr), len(y_arr), replace=True)
    if y_arr[idx].sum() < 2:
        continue
    auc_boots.append(roc_auc_score(y_arr[idx], best_oof_arr[idx]))
ci_low, ci_high = np.percentile(auc_boots, [2.5, 97.5])
print(f'AUC={best_r["roc_auc"]:.4f}  95% CI=[{ci_low:.3f}, {ci_high:.3f}]')

robustness = {
    'roc_auc_mean': best_r['roc_auc'],
    'roc_auc': best_r['roc_auc'],
    'ci_low': ci_low, 'ci_high': ci_high,
    'n_bootstrap': 2000,
    'best_model': best_name,
}
with open(MODELS_DIR / 'robustness_results.json', 'w') as f:
    json.dump(robustness, f, indent=2)

# =============================================================================
# 9. SUMMARY
# =============================================================================
print('\n' + '=' * 70)
print('TRAINING COMPLETE')
print('=' * 70)
print(f'Best model  : {best_name}')
print(f'ROC-AUC     : {best_r["roc_auc"]:.4f}')
print(f'95% CI      : [{ci_low:.3f}, {ci_high:.3f}]')
print(f'Lift        : {best_r["lift"]:.1f}x')
print(f'n+          : {n_pos}  (class ratio 1:{n_neg//n_pos})')
print(f'Threshold   : {t_yo:.4f}  Recall={recall_at_youden*100:.0f}%  Flagged={flagged}/{len(y)}')
print(f'Finished    : {datetime.now():%Y-%m-%d %H:%M:%S}')
