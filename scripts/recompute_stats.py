#!/usr/bin/env python3
"""
recompute_stats.py
------------------
Recomputes all statistical validations from scratch and saves results.

Produces:
  models/robustness_results.json   — updated with bootstrap_aucs + permutation_aucs arrays
  models/oof_predictions.json      — OOF y_true, y_score_brf, y_score_lgbm for DeLong
  models/statistical_tests.json    — updated Mann-Whitney with medians for non-GLOF
  models/delong_test.json          — DeLong z, p, CIs for BalancedRF vs LightGBM

Run from project root with the glof venv:
    glof/Scripts/python recompute_stats.py
"""

import json, sys, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'

print("=" * 60)
print("GLOF — Recompute Statistical Validations")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")

df = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
with open(MODELS_DIR / 'feature_names.json', encoding='utf-8') as f:
    feature_cols = json.load(f)

excl = ['lake_id', 'had_glof', 'country', 'cordillera', 'geometry', 'year',
        'source', 'area_name', 'scene_date', 'glof_volume_m3', 'glof_match_dist_m',
        'glof_date', 'glof_trigger', 'glof_lake_name']
num_cols = [c for c in df.columns if c not in excl
            and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

X_raw = df[num_cols].copy().fillna(df[num_cols].median()).fillna(0)

# Engineered features (same as generate_figures.py)
for col in ['area_m2', 'perimeter_m', 'volume_m3', 'equivalent_diameter_m',
            'potential_energy', 'depth_m', 'dam_height',
            'area_2017_est', 'area_2025_est', 'total_growth_m2']:
    if col in X_raw.columns:
        X_raw[f'log_{col}'] = np.log1p(X_raw[col].clip(lower=0))
if 'total_growth_m2' in X_raw.columns and 'area_2017_est' in X_raw.columns:
    X_raw['rel_growth'] = X_raw['total_growth_m2'] / (X_raw['area_2017_est'] + 1)
if 'slope_mean' in X_raw.columns and 'log_area_m2' in X_raw.columns:
    X_raw['slope_x_logarea'] = X_raw['slope_mean'] * X_raw['log_area_m2']
if 'dist_glacier_m' in df.columns:
    X_raw['no_glacier'] = df['dist_glacier_m'].isna().astype(int)

feat_avail = [c for c in feature_cols if c in X_raw.columns]
X = X_raw[feat_avail].values
y = df['had_glof'].values.astype(int)

n_pos = y.sum()
n_tot = len(y)
prevalence = n_pos / n_tot
print(f"  Samples: {n_tot:,}  |  GLOF+: {n_pos}  |  Prevalence: {prevalence:.5f}")
print(f"  Features used: {len(feat_avail)}/{len(feature_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. OOF PREDICTIONS (BalancedRF + LightGBM) — 5-fold stratified
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Running 5-fold OOF cross-validation...")

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
import lightgbm as lgb

n_pos_total = int(y.sum())

# Exact parameters from notebook 15_model_training.ipynb (RANDOM_STATE=42, N_SPLITS=5)
brf_params = dict(n_estimators=500, max_depth=None, min_samples_leaf=1,
                  sampling_strategy='auto', replacement=True,
                  n_jobs=-1, random_state=42)
lgbm_clf = lgb.LGBMClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    is_unbalance=True, device='cpu',
    random_state=42, verbose=-1
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_brf  = np.zeros(n_tot)
oof_lgbm = np.zeros(n_tot)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    t0 = time.time()
    X_tr, X_va = X[train_idx], X[val_idx]
    y_tr, y_va = y[train_idx], y[val_idx]

    # BalancedRF — internal resampling, no SMOTE
    brf = BalancedRandomForestClassifier(**brf_params)
    brf.fit(X_tr, y_tr)
    oof_brf[val_idx] = brf.predict_proba(X_va)[:, 1]

    # LightGBM — with SMOTETomek (same as original notebook)
    n_pos_fold = int(y_tr.sum())
    k_neighbors = min(5, n_pos_fold - 1) if n_pos_fold > 1 else 1
    smt = SMOTETomek(smote=SMOTE(random_state=42, k_neighbors=k_neighbors),
                     random_state=42)
    pipe_lgbm = ImbPipeline([('smote', smt), ('clf', clone(lgbm_clf))])
    pipe_lgbm.fit(X_tr, y_tr)
    oof_lgbm[val_idx] = pipe_lgbm.predict_proba(X_va)[:, 1]

    auc_b = roc_auc_score(y_va, oof_brf[val_idx])
    auc_l = roc_auc_score(y_va, oof_lgbm[val_idx])
    print(f"  Fold {fold}: BRF={auc_b:.4f}  LGBM={auc_l:.4f}  ({time.time()-t0:.1f}s)")

auc_brf_oof  = roc_auc_score(y, oof_brf)
auc_lgbm_oof = roc_auc_score(y, oof_lgbm)
print(f"\n  OOF BRF  AUC = {auc_brf_oof:.4f}")
print(f"  OOF LGBM AUC = {auc_lgbm_oof:.4f}")

# Save OOF predictions
oof_preds = {
    'y_true':      y.tolist(),
    'y_score_brf': oof_brf.tolist(),
    'y_score_lgbm': oof_lgbm.tolist(),
    'auc_brf':     auc_brf_oof,
    'auc_lgbm':    auc_lgbm_oof,
}
with open(MODELS_DIR / 'oof_predictions.json', 'w', encoding='utf-8') as f:
    json.dump(oof_preds, f)
print("  Saved: models/oof_predictions.json")

# ─────────────────────────────────────────────────────────────────────────────
# 3. BOOTSTRAP CI (2000 resamples on OOF BRF predictions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Bootstrap CI (2,000 resamples)...")

rng = np.random.default_rng(0)
bootstrap_aucs = []
n_boot = 2000
for _ in range(n_boot):
    idx = rng.integers(0, n_tot, n_tot)
    y_b = y[idx]
    if y_b.sum() < 2 or y_b.sum() == len(y_b):
        continue
    bootstrap_aucs.append(roc_auc_score(y_b, oof_brf[idx]))

bootstrap_aucs = np.array(bootstrap_aucs)
ci_low  = float(np.percentile(bootstrap_aucs, 2.5))
ci_high = float(np.percentile(bootstrap_aucs, 97.5))
print(f"  n_valid resamples: {len(bootstrap_aucs)}")
print(f"  95% CI = [{ci_low:.4f}, {ci_high:.4f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABEL-PERMUTATION TEST (5000 shuffles)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Label-permutation test (5,000 shuffles)...")

rng2 = np.random.default_rng(1)
permutation_aucs = []
n_perm = 5000
for _ in range(n_perm):
    y_perm = rng2.permutation(y)
    permutation_aucs.append(roc_auc_score(y_perm, oof_brf))

permutation_aucs = np.array(permutation_aucs)
p_value = float((permutation_aucs >= auc_brf_oof).mean())
p_report = p_value if p_value > 0 else 1.0 / n_perm
print(f"  Observed AUC     : {auc_brf_oof:.4f}")
print(f"  Null mean ± std  : {permutation_aucs.mean():.4f} ± {permutation_aucs.std():.4f}")
print(f"  p-value          : {p_value} → reported as p < {p_report:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. DELONG TEST (BalancedRF vs LightGBM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] DeLong test: BalancedRF vs LightGBM...")


def delong_roc_variance(ground_truth, predictions):
    """Compute DeLong AUC variance using structural components."""
    n_p = ground_truth.sum()
    n_n = (1 - ground_truth).sum()

    score_p = predictions[ground_truth == 1]
    score_n = predictions[ground_truth == 0]

    def psi(a, b):
        if a > b: return 1.0
        if a == b: return 0.5
        return 0.0

    # V10[i] = mean_{j} psi(Xi, Yj) for positive sample i
    V10 = np.array([np.mean([psi(s, sn) for sn in score_n]) for s in score_p])
    # V01[j] = mean_{i} psi(Xi, Yj) for negative sample j
    V01 = np.array([np.mean([psi(sp, s) for sp in score_p]) for s in score_n])

    var = (np.var(V10, ddof=1) / n_p + np.var(V01, ddof=1) / n_n)
    return var


def fast_delong_variance(y_true, y_score):
    """Faster vectorized version."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_p = len(pos_scores)
    n_n = len(neg_scores)

    # V10[i]: for each positive, what fraction of negatives it beats
    V10 = np.array([(np.sum(pos_scores[i] > neg_scores) +
                     0.5 * np.sum(pos_scores[i] == neg_scores)) / n_n
                    for i in range(n_p)])
    # V01[j]: for each negative, fraction of positives it's beaten by
    V01 = np.array([(np.sum(pos_scores > neg_scores[j]) +
                     0.5 * np.sum(pos_scores == neg_scores[j])) / n_p
                    for j in range(n_n)])

    var_auc = np.var(V10, ddof=1) / n_p + np.var(V01, ddof=1) / n_n
    return var_auc, V10, V01


def delong_paired_test(y_true, score1, score2):
    """
    DeLong test for correlated (paired) ROC curves.
    Returns (z_stat, p_value, auc1, auc2).
    """
    auc1 = roc_auc_score(y_true, score1)
    auc2 = roc_auc_score(y_true, score2)

    pos = y_true == 1
    neg = y_true == 0
    n_p = pos.sum()
    n_n = neg.sum()

    sp1, sn1 = score1[pos], score1[neg]
    sp2, sn2 = score2[pos], score2[neg]

    def V10_V01(sp, sn):
        np_ = len(sp); nn_ = len(sn)
        V10 = np.array([(np.sum(sp[i] > sn) + 0.5 * np.sum(sp[i] == sn)) / nn_
                        for i in range(np_)])
        V01 = np.array([(np.sum(sp > sn[j]) + 0.5 * np.sum(sp == sn[j])) / np_
                        for j in range(nn_)])
        return V10, V01

    V10_1, V01_1 = V10_V01(sp1, sn1)
    V10_2, V01_2 = V10_V01(sp2, sn2)

    var1   = np.var(V10_1, ddof=1) / n_p + np.var(V01_1, ddof=1) / n_n
    var2   = np.var(V10_2, ddof=1) / n_p + np.var(V01_2, ddof=1) / n_n
    cov12  = (np.cov(V10_1, V10_2, ddof=1)[0, 1] / n_p +
              np.cov(V01_1, V01_2, ddof=1)[0, 1] / n_n)

    var_diff = var1 + var2 - 2 * cov12
    if var_diff <= 0:
        z = 0.0; p = 1.0
    else:
        z = (auc1 - auc2) / np.sqrt(var_diff)
        p = float(2 * (1 - stats.norm.cdf(abs(z))))

    print(f"  AUC BRF  = {auc1:.4f}")
    print(f"  AUC LGBM = {auc2:.4f}")
    print(f"  Var(AUC1) = {var1:.6f}   Var(AUC2) = {var2:.6f}")
    print(f"  Cov12     = {cov12:.6f}")
    print(f"  Var(diff) = {var_diff:.6f}")
    print(f"  z-stat    = {z:.4f}")
    print(f"  p-value   = {p:.4f}  (two-sided)")
    return z, p, auc1, auc2


z, p_delong, auc_brf_d, auc_lgbm_d = delong_paired_test(y, oof_brf, oof_lgbm)
delong_result = {
    'auc_brf':        auc_brf_d,
    'auc_lgbm':       auc_lgbm_oof,
    'z_statistic':    z,
    'p_value':        p_delong,
    'variance_brf':   float(fast_delong_variance(y, oof_brf)[0]),
    'variance_lgbm':  float(fast_delong_variance(y, oof_lgbm)[0]),
    'interpretation': (
        f'z={z:.2f}, p={p_delong:.3f}: '
        + ('BalancedRF significantly > LightGBM' if p_delong < 0.05
           else 'No significant difference (p>0.05)')
    )
}
with open(MODELS_DIR / 'delong_test.json', 'w', encoding='utf-8') as f:
    json.dump(delong_result, f, indent=2)
print(f"  Saved: models/delong_test.json")

# ─────────────────────────────────────────────────────────────────────────────
# 6. UPDATED MANN-WHITNEY (add non-GLOF medians)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[+] Updating Mann-Whitney table with non-GLOF medians...")

feat_name_map = {
    'dist_glacier_m':          'Distance to glacier (m)',
    'volume_m3':               'Volume (m\u00b3)',
    'area_depth_ratio':        'Area/depth ratio',
    'depth_ensemble_mean':     'Ensemble depth mean (m)',
    'depth_std_m':             'Depth temporal std (m)',
    'depth_cook_quincey_2015': 'Depth \u2013 Cook & Quincey (m)',
    'depth_ensemble_std':      'Depth ensemble std (m)',
    'elongation':              'Elongation',
}

with open(MODELS_DIR / 'statistical_tests.json', encoding='utf-8') as f:
    st_raw = json.load(f)

# Re-read raw data for non-GLOF medians
df_raw = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
st_updated = []
for item in st_raw:
    feat = item['feature']
    item_new = dict(item)
    item_new['label'] = feat_name_map.get(feat, feat)
    if feat in df_raw.columns:
        item_new['median_glof']    = float(df_raw.loc[df_raw['had_glof']==1, feat].median())
        item_new['median_nonglof'] = float(df_raw.loc[df_raw['had_glof']==0, feat].median())
    st_updated.append(item_new)

with open(MODELS_DIR / 'statistical_tests.json', 'w', encoding='utf-8') as f:
    json.dump(st_updated, f, indent=2, ensure_ascii=False)
print("  Saved: models/statistical_tests.json (updated)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE UPDATED ROBUSTNESS RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[+] Saving updated robustness_results.json...")

# Load existing precision_at_k
with open(MODELS_DIR / 'robustness_results.json', encoding='utf-8') as f:
    rob_old = json.load(f)

rob_new = {
    'roc_auc_oof':          auc_brf_oof,
    'ci_low':               ci_low,
    'ci_high':              ci_high,
    'n_boot':               len(bootstrap_aucs),
    'bootstrap_aucs':       bootstrap_aucs.tolist(),
    'p_value':              p_value,
    'p_value_reported':     p_report,
    'n_perm':               n_perm,
    'permutation_aucs':     permutation_aucs.tolist(),
    'permutation_mean':     float(permutation_aucs.mean()),
    'permutation_std':      float(permutation_aucs.std()),
    'precision_at_k':       rob_old.get('precision_at_k', []),
    'n_features':           len(feat_avail),
    'risk_score_removed':   True,
    'glof_rank_mean_top_pct':   rob_old.get('glof_rank_mean_top_pct', 18.0),
    'glof_rank_median_top_pct': rob_old.get('glof_rank_median_top_pct', 16.1),
    'glof_rank_max_top_pct':    rob_old.get('glof_rank_max_top_pct', 46.8),
    'glof_rank_min_top_pct':    rob_old.get('glof_rank_min_top_pct', 0.3),
}

with open(MODELS_DIR / 'robustness_results.json', 'w', encoding='utf-8') as f:
    json.dump(rob_new, f, indent=2)
print("  Saved: models/robustness_results.json (with real bootstrap/permutation arrays)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"OOF ROC-AUC (BalancedRF) : {auc_brf_oof:.4f}")
print(f"OOF ROC-AUC (LightGBM)   : {auc_lgbm_oof:.4f}")
print(f"Bootstrap 95% CI         : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Permutation p-value      : {p_value} → p < {p_report:.4f}")
print(f"DeLong z                 : {z:.4f}")
print(f"DeLong p                 : {p_delong:.4f}")
sig_str = 'SIGNIFICANT' if p_delong < 0.05 else 'not significant'
print(f"DeLong result            : {sig_str} (α=0.05)")
print()
print("Files saved:")
print("  models/oof_predictions.json    — y_true, y_score_brf, y_score_lgbm")
print("  models/robustness_results.json — bootstrap/permutation arrays")
print("  models/statistical_tests.json  — updated with all medians")
print("  models/delong_test.json        — DeLong z, p, variances")
print("=" * 60)
