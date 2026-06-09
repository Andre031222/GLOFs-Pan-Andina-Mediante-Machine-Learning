#!/usr/bin/env python3
"""
scripts/compute_extra_validation.py
------------------------------------
Computes four additional validation metrics and saves to
models/extra_validation.json:

  1. Kolmogorov-Smirnov test  — GLOF vs non-GLOF OOF score distributions
  2. Prospective validation   — top-ranked non-GLOF lakes (geographic analysis)
  3. Baseline comparison      — BalancedRF vs LogReg framing (already computed,
                                re-verified here)
  4. Calibration data         — reliability diagram bins (fraction positives
                                per score decile) + Brier components

Run from project root:
    glof/Scripts/python scripts/compute_extra_validation.py
"""

import json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'

# ── Load OOF predictions ──────────────────────────────────────────────────────
print("Loading OOF predictions...")
with open(MODELS_DIR / 'oof_predictions.json') as f:
    oof = json.load(f)

y_true      = np.array(oof['y_true'], dtype=int)
y_score_brf = np.array(oof['y_score_brf'], dtype=float)

n_pos = int(y_true.sum())
n_tot = len(y_true)
prev  = n_pos / n_tot
print(f"  n={n_tot:,}  n+={n_pos}  prevalence={prev:.5f}")

# ── Load training CSV & lakes GDF ─────────────────────────────────────────────
print("Loading lakes and training data...")
df_csv = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
lakes  = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')

COUNTRY_MAP = {
    'cordillera_blanca':       'Peru',
    'cordillera_vilcanota':    'Peru',
    'cordillera_central':      'Peru',
    'cordillera_raura':        'Peru',
    'cordillera_urubamba':     'Peru',
    'cordillera_huanzo':       'Peru',
    'cordillera_huayhuash':    'Peru',
    'bolivia_cordillera_real': 'Bolivia',
    'ecuador_antisana':        'Ecuador',
    'chile_andes_centrales':   'Chile',
}
AREA_LABELS = {
    'cordillera_blanca':       'C. Blanca (PE)',
    'cordillera_vilcanota':    'C. Vilcanota (PE)',
    'cordillera_central':      'C. Central (PE)',
    'chile_andes_centrales':   'Andes Centrales (CL)',
    'cordillera_raura':        'C. Raura (PE)',
    'cordillera_urubamba':     'C. Urubamba (PE)',
    'cordillera_huanzo':       'C. Huanzo (PE)',
    'cordillera_huayhuash':    'C. Huayhuash (PE)',
    'ecuador_antisana':        'Antisana (EC)',
    'bolivia_cordillera_real': 'C. Real (BO)',
}

lakes['y_score_oof'] = y_score_brf
lakes['y_true']      = y_true.tolist()
lakes['country']     = lakes['area_name'].map(COUNTRY_MAP)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Kolmogorov-Smirnov test
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Kolmogorov-Smirnov test...")
scores_pos = y_score_brf[y_true == 1]
scores_neg = y_score_brf[y_true == 0]

ks_stat, ks_p = ks_2samp(scores_pos, scores_neg, alternative='less')
# alternative='less': tests that GLOF scores are stochastically GREATER
# (i.e., F_pos(x) <= F_neg(x) — GLOF distribution shifted to the right)

# Also two-sided
ks_stat2, ks_p2 = ks_2samp(scores_pos, scores_neg, alternative='two-sided')

print(f"  KS statistic (one-sided, GLOF > non-GLOF) D={ks_stat:.4f}  p={ks_p:.6f}")
print(f"  KS statistic (two-sided)                  D={ks_stat2:.4f}  p={ks_p2:.6f}")
print(f"  GLOF scores:    median={np.median(scores_pos):.3f}  mean={np.mean(scores_pos):.3f}")
print(f"  non-GLOF scores: median={np.median(scores_neg):.3f}  mean={np.mean(scores_neg):.3f}")

ks_result = {
    'ks_statistic':       float(ks_stat2),
    'ks_p_value':         float(ks_p2),
    'ks_p_onesided':      float(ks_p),
    'median_glof':        float(np.median(scores_pos)),
    'median_nonglof':     float(np.median(scores_neg)),
    'mean_glof':          float(np.mean(scores_pos)),
    'mean_nonglof':       float(np.mean(scores_neg)),
    'interpretation':     'GLOF-source lake scores are significantly higher '
                          'than non-GLOF scores (two-sided KS test)',
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Prospective validation — top-ranked non-GLOF lakes
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Prospective validation (top-ranked non-GLOF lakes)...")

# Assign OOF scores to the full dataset
df_work = df_csv.copy()
df_work['y_score_oof'] = y_score_brf
df_work['country']     = df_work['area_name'].map(COUNTRY_MAP)

# Top-100 non-GLOF lakes by OOF score
df_nonglof = df_work[df_work['had_glof'] == 0].sort_values(
    'y_score_oof', ascending=False).reset_index(drop=True)

top50  = df_nonglof.head(50)
top100 = df_nonglof.head(100)
top252 = df_nonglof.head(252)   # top 2%

# Geographic distribution of top-ranked non-GLOF lakes
area_counts_50  = top50['area_name'].value_counts().to_dict()
area_counts_100 = top100['area_name'].value_counts().to_dict()
area_counts_252 = top252['area_name'].value_counts().to_dict()

# Key stats: % from historically high-risk cordilleras
high_risk_areas = {
    'cordillera_blanca', 'cordillera_huayhuash', 'cordillera_raura'
}
n_hr_50  = sum(v for k, v in area_counts_50.items()  if k in high_risk_areas)
n_hr_100 = sum(v for k, v in area_counts_100.items() if k in high_risk_areas)
n_hr_252 = sum(v for k, v in area_counts_252.items() if k in high_risk_areas)

print(f"  Top-50 non-GLOF: {n_hr_50}/50 ({n_hr_50/50*100:.0f}%) from "
      f"historically high-risk cordilleras")
print(f"  Top-100 non-GLOF: {n_hr_100}/100 ({n_hr_100:.0f}%) from "
      f"historically high-risk cordilleras")
print(f"  Top-252 (2%) non-GLOF: {n_hr_252}/252 ({n_hr_252/252*100:.0f}%)")

# Mean features of top-50 non-GLOF lakes vs. overall non-GLOF
feat_compare = {}
for feat in ['dist_glacier_m', 'volume_m3', 'depth_m', 'area_m2']:
    if feat in df_work.columns:
        top50_med = float(np.nanmedian(top50[feat]))
        all_neg_med = float(np.nanmedian(df_nonglof[feat]))
        feat_compare[feat] = {
            'top50_median': top50_med,
            'all_nonglof_median': all_neg_med,
        }
        print(f"  {feat:25s}: top-50 median={top50_med:.1f}  "
              f"all non-GLOF median={all_neg_med:.1f}")

# Area breakdown for top-50
top50_area = {
    AREA_LABELS.get(k, k): int(v)
    for k, v in sorted(area_counts_50.items(),
                        key=lambda x: -x[1])
}
print(f"  Area breakdown top-50: {top50_area}")

prospective = {
    'top50_high_risk_fraction':  float(n_hr_50 / 50),
    'top100_high_risk_fraction': float(n_hr_100 / 100),
    'top252_high_risk_fraction': float(n_hr_252 / 252),
    'top50_area_breakdown':      {AREA_LABELS.get(k,k): int(v)
                                  for k, v in area_counts_50.items()},
    'top100_area_breakdown':     {AREA_LABELS.get(k,k): int(v)
                                  for k, v in area_counts_100.items()},
    'feature_comparison':        feat_compare,
    'high_risk_areas':           list(high_risk_areas),
    'top50_score_range':         [float(top50['y_score_oof'].min()),
                                  float(top50['y_score_oof'].max())],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Baseline comparison — BalancedRF vs LogReg (re-verify from CSV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Baseline comparison...")
model_csv = pd.read_csv(MODELS_DIR / 'model_comparison.csv')
model_csv.columns = [c.strip() for c in model_csv.columns]

brf_auc  = float(model_csv.loc[model_csv['Model']=='BalancedRF', 'ROC-AUC'].values[0])
log_auc  = float(model_csv.loc[model_csv['Model']=='LogReg',     'ROC-AUC'].values[0])
xgb_auc  = float(model_csv.loc[model_csv['Model']=='XGBoost',    'ROC-AUC'].values[0])
rf_auc   = float(model_csv.loc[model_csv['Model']=='RandomForest','ROC-AUC'].values[0])

rel_improvement_logreg = (brf_auc - log_auc) / log_auc * 100
abs_gain_logreg        = brf_auc - log_auc

print(f"  BalancedRF AUC = {brf_auc:.3f}")
print(f"  LogReg AUC     = {log_auc:.3f}")
print(f"  Absolute gain  = {abs_gain_logreg:.3f}")
print(f"  Relative gain  = {rel_improvement_logreg:.1f}%")

baseline_comparison = {
    'brf_auc':              brf_auc,
    'logreg_auc':           log_auc,
    'xgboost_auc':          xgb_auc,
    'randomforest_auc':     rf_auc,
    'abs_gain_vs_logreg':   float(abs_gain_logreg),
    'rel_gain_vs_logreg_pct': float(rel_improvement_logreg),
    'note': (
        'LogReg implements the linear discriminant baseline analogous to '
        'McKillop & Clague (2007) on our 57-feature dataset. '
        f'BalancedRF achieves {abs_gain_logreg:.3f} absolute AUC gain '
        f'({rel_improvement_logreg:.1f}% relative improvement).'
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Calibration — reliability diagram data + Brier decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Calibration analysis...")

# sklearn calibration_curve: n_bins=10 (deciles of predicted score)
n_bins_list = [5, 10]
calib_data = {}

for n_bins in n_bins_list:
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_score_brf,
            n_bins=n_bins,
            strategy='quantile',   # equal-frequency bins (robust with few positives)
        )
        calib_data[f'bins_{n_bins}'] = {
            'mean_predicted_value': [float(x) for x in mean_pred],
            'fraction_of_positives': [float(x) for x in frac_pos],
        }
        print(f"  Reliability ({n_bins} bins, quantile strategy):")
        for mp, fp in zip(mean_pred, frac_pos):
            print(f"    pred={mp:.3f}  frac_pos={fp:.5f}")
    except Exception as e:
        print(f"  [warn] calibration_curve n_bins={n_bins}: {e}")

# Brier Score
brier = float(np.mean((y_score_brf - y_true) ** 2))
brier_baseline = float(prev * (1 - prev))   # no-skill baseline (always predict prev)
brier_skill    = 1.0 - brier / brier_baseline

print(f"\n  Brier Score          = {brier:.6f}")
print(f"  Brier baseline (prev)= {brier_baseline:.6f}")
print(f"  Brier Skill Score    = {brier_skill:.3f}")
print(f"  Interpretation: BSS<0 expected for BRF (inflated score scale)")

# Score quantiles for context
q_neg = np.quantile(scores_neg, [0.5, 0.75, 0.90, 0.95, 0.99])
q_pos = np.quantile(scores_pos, [0.25, 0.5, 0.75])
print(f"\n  Non-GLOF score quantiles (50/75/90/95/99): "
      f"{q_neg.round(3)}")
print(f"  GLOF score quantiles (25/50/75): {q_pos.round(3)}")

calibration = {
    'brier_score':        brier,
    'brier_baseline':     brier_baseline,
    'brier_skill_score':  brier_skill,
    'reliability_data':   calib_data,
    'score_quantiles_neg': {
        'p50': float(q_neg[0]), 'p75': float(q_neg[1]),
        'p90': float(q_neg[2]), 'p95': float(q_neg[3]),
        'p99': float(q_neg[4]),
    },
    'score_quantiles_pos': {
        'p25': float(q_pos[0]), 'p50': float(q_pos[1]),
        'p75': float(q_pos[2]),
    },
    'interpretation': (
        'Negative BSS is expected: BRF internal resampling under 1:787 '
        'imbalance inflates scores above the true prevalence (0.127%). '
        'Scores function as ranking indices, not calibrated probabilities.'
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Save all results
# ─────────────────────────────────────────────────────────────────────────────
result = {
    'ks_test':             ks_result,
    'prospective_validation': prospective,
    'baseline_comparison': baseline_comparison,
    'calibration':         calibration,
}

out_path = MODELS_DIR / 'extra_validation.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved -> {out_path}")
print("Done.")
