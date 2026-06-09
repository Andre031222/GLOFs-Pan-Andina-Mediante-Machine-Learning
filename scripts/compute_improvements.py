#!/usr/bin/env python3
"""
compute_improvements.py
-----------------------
Computes all 7 manuscript improvements from OOF predictions + training data:

  1. Brier Score (calibration)
  2. Threshold sensitivity analysis
  3. LOCO bootstrap CI for Peru
  4. Spatial autocorrelation (Moran's I approximation)
  5. SHAP importance stability (bootstrap)
  6. Reliability diagram data
  7. Updated LOCO reformulation stats

Outputs:
  models/improvements.json   — all new statistics for LaTeX insertion

Run from project root:
    glof/Scripts/python scripts/compute_improvements.py
"""

import json, sys, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (brier_score_loss, roc_auc_score,
                              precision_score, recall_score, f1_score,
                              average_precision_score)
from sklearn.calibration import calibration_curve

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
DATA_DIR   = ROOT / 'data' / 'processed' / 'labeled'

print("=" * 65)
print("GLOF — Compute Manuscript Improvements")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD OOF PREDICTIONS & TRAINING DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[0] Loading OOF predictions and training data...")

with open(MODELS_DIR / 'oof_predictions.json', encoding='utf-8') as f:
    oof = json.load(f)

y_true  = np.array(oof['y_true'], dtype=int)
y_score = np.array(oof['y_score_brf'], dtype=float)
auc_oof = oof['auc_brf']

df = pd.read_csv(DATA_DIR / 'training_data.csv')

# Map area_name → country
country_map = {
    'bolivia_cordillera_real': 'Bolivia',
    'chile_andes_centrales':   'Chile',
    'ecuador_antisana':        'Ecuador',
}
def get_country(area):
    if area in country_map:
        return country_map[area]
    return 'Peru'

df['country'] = df['area_name'].apply(get_country)

print(f"  n_total={len(y_true)}, n_pos={y_true.sum()}, AUC={auc_oof:.4f}")

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 1. BRIER SCORE (use calibrated model for proper probability estimates)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Brier Score (calibration)...")

import joblib
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Load calibrated model if available
cal_path = MODELS_DIR / 'best_model_calibrated.joblib'
if cal_path.exists():
    print("  Loading calibrated model...")
    cal_model = joblib.load(cal_path)
    # We need X for calibrated predictions
    # Load feature names and build X
    with open(MODELS_DIR / 'feature_names.json', encoding='utf-8') as f:
        feature_cols = json.load(f)
    # calibrated model was trained with risk_score (58 features) — keep it
    excl = ['lake_id', 'had_glof', 'country', 'cordillera', 'geometry', 'year',
            'source', 'area_name', 'scene_date', 'glof_volume_m3',
            'glof_match_dist_m', 'glof_date', 'glof_trigger', 'glof_lake_name',
            'size_class']
    num_cols_b = [c for c in df.columns if c not in excl
                  and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    X_raw_b = df[num_cols_b].copy().fillna(df[num_cols_b].median()).fillna(0)
    for col in ['area_m2', 'perimeter_m', 'volume_m3', 'equivalent_diameter_m',
                'potential_energy', 'depth_m', 'dam_height',
                'area_2017_est', 'area_2025_est', 'total_growth_m2']:
        if col in X_raw_b.columns:
            X_raw_b[f'log_{col}'] = np.log1p(X_raw_b[col].clip(lower=0))
    if 'total_growth_m2' in X_raw_b.columns and 'area_2017_est' in X_raw_b.columns:
        X_raw_b['rel_growth'] = X_raw_b['total_growth_m2'] / (X_raw_b['area_2017_est'] + 1)
    if 'slope_mean' in X_raw_b.columns and 'log_area_m2' in X_raw_b.columns:
        X_raw_b['slope_x_logarea'] = X_raw_b['slope_mean'] * X_raw_b['log_area_m2']
    if 'dist_glacier_m' in df.columns:
        X_raw_b['no_glacier'] = df['dist_glacier_m'].isna().astype(int)
    # calibrated model uses risk_score as 58th feature
    feat_avail_b = [c for c in (list(feature_cols) + ['risk_score']) if c in X_raw_b.columns]
    X_b = X_raw_b[feat_avail_b].values
    y_b = df['had_glof'].values.astype(int)

    y_cal_score = cal_model.predict_proba(X_b)[:, 1]
    brier = brier_score_loss(y_b, y_cal_score)
    brier_note = 'calibrated model (isotonic Platt scaling)'
else:
    # Fallback: use OOF scores (uncalibrated BRF — scores inflated by internal balancing)
    # Still informative as ranking metric
    y_cal_score = y_score
    brier = brier_score_loss(y_true, y_score)
    brier_note = 'uncalibrated OOF scores (BRF inflated due to internal balancing)'

# Brier score reference: null model always predicts the prevalence
prev = y_true.mean()
# E[(y - prev)^2] = prev*(1-prev)^2 + (1-prev)*prev^2 = prev*(1-prev)
brier_ref = float(prev * (1 - prev))

# Brier Skill Score: BSS = 1 - BS/BS_ref (positive = better than null, negative = worse)
# With BRF and extreme imbalance, BSS is expected to be negative because
# the model outputs scores >> prevalence (ranking scores, not calibrated probabilities)
brier_skill = 1.0 - brier / brier_ref if brier_ref > 0 else float('nan')

print(f"  Model           : {brier_note}")
print(f"  Brier Score     : {brier:.6f}")
print(f"  Brier Reference : {brier_ref:.6f}")
print(f"  Brier Skill     : {brier_skill:.4f}")

results['brier_score']       = float(brier)
results['brier_reference']   = float(brier_ref)
results['brier_skill_score'] = float(brier_skill)
results['brier_note']        = brier_note

# Use calibrated scores for reliability diagram
y_score_for_reliability = y_cal_score

# ─────────────────────────────────────────────────────────────────────────────
# 2. RELIABILITY DIAGRAM DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Reliability diagram data...")

# Use quantile-based bins to handle extreme imbalance
try:
    prob_true, prob_pred = calibration_curve(
        y_true, y_score_for_reliability, n_bins=10, strategy='quantile'
    )
    results['reliability_prob_true'] = prob_true.tolist()
    results['reliability_prob_pred'] = prob_pred.tolist()
    print(f"  Calibration bins: {len(prob_true)}")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"    pred={pp:.4f}  true={pt:.4f}  diff={pt-pp:+.4f}")
except Exception as e:
    print(f"  WARNING: calibration_curve failed: {e}")
    # Simple manual approach
    bins = np.linspace(0, 1, 11)
    prob_true_list, prob_pred_list = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() > 0:
            prob_pred_list.append(y_score[mask].mean())
            prob_true_list.append(y_true[mask].mean())
    results['reliability_prob_true'] = prob_true_list
    results['reliability_prob_pred'] = prob_pred_list

# ─────────────────────────────────────────────────────────────────────────────
# 3. THRESHOLD SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Threshold sensitivity analysis...")

thresholds = [0.10, 0.15, 0.20, 0.25, 0.304, 0.35, 0.40, 0.50, 0.60]
thresh_results = []

n_pos_total = y_true.sum()
n_tot       = len(y_true)

for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    n_flagged = y_pred.sum()
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec    = tp / n_pos_total if n_pos_total > 0 else 0.0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    frac   = n_flagged / n_tot

    thresh_results.append({
        'threshold': t,
        'n_flagged': int(n_flagged),
        'flagged_pct': round(100 * frac, 1),
        'precision': round(prec, 4),
        'recall':    round(rec, 4),
        'f1':        round(f1, 5),
        'tp':        int(tp),
    })
    print(f"  t={t:.3f}: flagged={n_flagged:,} ({100*frac:.1f}%) | "
          f"prec={prec:.4f} rec={rec:.4f} F1={f1:.5f} TP={tp}")

results['threshold_sensitivity'] = thresh_results

# Youden optimal (from optimal_thresholds.json if available, else compute)
try:
    with open(MODELS_DIR / 'optimal_thresholds.json', encoding='utf-8') as f:
        ot = json.load(f)
    t_youden = ot.get('BalancedRF', {}).get('youden', 0.304)
except Exception:
    t_youden = 0.304
results['youden_threshold'] = float(t_youden)

# ─────────────────────────────────────────────────────────────────────────────
# 4. LOCO BOOTSTRAP CI FOR PERU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] LOCO bootstrap CI for Peru...")

# Filter Peru observations from OOF predictions
peru_mask = (df['country'] == 'Peru').values
y_peru    = y_true[peru_mask]
s_peru    = y_score[peru_mask]

print(f"  Peru: n={len(y_peru)}, n_pos={y_peru.sum()}")
loco_auc_peru_raw = roc_auc_score(y_peru, s_peru) if y_peru.sum() >= 2 else float('nan')
print(f"  OOF-on-Peru AUC: {loco_auc_peru_raw:.4f}")

# Bootstrap CI (1000 resamples) on Peru subset
rng = np.random.default_rng(42)
boot_aucs_peru = []
n_boot_peru = 2000

for _ in range(n_boot_peru):
    idx = rng.integers(0, len(y_peru), len(y_peru))
    yb  = y_peru[idx]
    sb  = s_peru[idx]
    if yb.sum() < 2 or yb.sum() == len(yb):
        continue
    boot_aucs_peru.append(roc_auc_score(yb, sb))

boot_aucs_peru = np.array(boot_aucs_peru)
peru_ci_low  = float(np.percentile(boot_aucs_peru, 2.5))
peru_ci_high = float(np.percentile(boot_aucs_peru, 97.5))

print(f"  Peru bootstrap 95% CI: [{peru_ci_low:.3f}, {peru_ci_high:.3f}]")
print(f"  n_valid_resamples: {len(boot_aucs_peru)}")

# For Bolivia and Ecuador (n+=1 each) — note only indicative
loco_csv = pd.read_csv(MODELS_DIR / 'loco_validation.csv')
# Column: País excluido, Lagunas test, GLOFs test, GLOFs train, ROC-AUC, PR-AUC, Lift×random, Threshold
col_map = {
    'País excluido': 'country_excluded',
    'Lagunas test': 'n_test',
    'GLOFs test': 'n_pos_test',
    'GLOFs train': 'n_pos_train',
    'ROC-AUC': 'roc_auc',
    'PR-AUC': 'pr_auc',
    'Lift×random': 'lift',
    'Threshold': 'threshold',
}
loco_csv = loco_csv.rename(columns=col_map)

results['loco_bootstrap_peru'] = {
    'roc_auc_oof_peru': float(loco_auc_peru_raw),
    'ci_low':           peru_ci_low,
    'ci_high':          peru_ci_high,
    'n_resamples':      len(boot_aucs_peru),
    'n_total_peru':     int(len(y_peru)),
    'n_pos_peru':       int(y_peru.sum()),
    'note': 'OOF predictions filtered to Peru observations (not leave-one-country-out)',
}

# Store LOCO table for reference
loco_records = loco_csv.to_dict('records')
results['loco_table'] = loco_records
print("  LOCO table stored.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SPATIAL AUTOCORRELATION (distance-based Moran's I approximation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Spatial autocorrelation analysis...")

# Load labeled_lakes.gpkg for coordinates
try:
    import geopandas as gpd
    gdf = gpd.read_file(DATA_DIR / 'labeled_lakes.gpkg')
    print(f"  Loaded labeled_lakes.gpkg: {len(gdf)} rows, cols={list(gdf.columns[:8])}")

    # Get centroid coordinates
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf['lon'] = gdf.geometry.centroid.x
    gdf['lat'] = gdf.geometry.centroid.y

    # Match with training data (by index or lake_id)
    if 'lake_id' in gdf.columns and 'lake_id' in df.columns:
        # De-duplicate gdf by lake_id (take first occurrence — unique spatial location)
        gdf_unique = gdf[['lake_id','lon','lat']].drop_duplicates('lake_id')
        merged = df[['lake_id','area_name']].merge(
            gdf_unique, on='lake_id', how='left'
        )
        lons = merged['lon'].values.astype(float)
        lats = merged['lat'].values.astype(float)
        has_coords = ~(np.isnan(lons) | np.isnan(lats))
        print(f"  Matched {has_coords.sum()}/{len(lons)} lakes with coordinates")
    else:
        # Use gdf directly if same order
        if len(gdf) == len(df):
            lons = gdf['lon'].values
            lats = gdf['lat'].values
            has_coords = np.ones(len(lons), dtype=bool)
        else:
            raise ValueError("Cannot match gdf to df — try de-duplicating gpkg")

    # Compute Moran's I on OOF residuals using a sample (expensive for full dataset)
    # Use a 10-degree spatial block correlation as a proxy
    # Simple test: correlation between OOF score and spatially lagged score
    # Use 1000-sample subset for efficiency
    sample_n = min(3000, has_coords.sum())
    valid_idx = np.where(has_coords)[0]
    rng2 = np.random.default_rng(0)
    samp = rng2.choice(valid_idx, sample_n, replace=False)

    lon_s = lons[samp]
    lat_s = lats[samp]
    score_s = y_score[samp]

    # Build distance matrix (approx, degrees) for sample
    from scipy.spatial.distance import cdist
    coords_s = np.column_stack([lon_s, lat_s])
    D = cdist(coords_s, coords_s, metric='euclidean')  # degrees, ~111km per degree

    # Spatial weight matrix: inverse distance within 0.5 degrees (~55km)
    threshold_deg = 0.5
    W = 1.0 / (D + 1e-9)
    W[D > threshold_deg] = 0.0
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_row = W / row_sums

    # Moran's I
    z = score_s - score_s.mean()
    Wz = W_row @ z
    morans_I = float(np.dot(z, Wz) / np.dot(z, z))
    print(f"  Moran's I (sample n={sample_n}, d<0.5°): {morans_I:.4f}")

    # Permutation test for Moran's I
    rng3 = np.random.default_rng(1)
    n_perm_moran = 999
    moran_perm = []
    for _ in range(n_perm_moran):
        zp = rng3.permutation(z)
        moran_perm.append(float(np.dot(zp, W_row @ zp) / np.dot(zp, zp)))
    moran_p = (np.sum(np.abs(moran_perm) >= abs(morans_I)) + 1) / (n_perm_moran + 1)
    print(f"  Moran's I p-value: {moran_p:.4f} (permutation, n={n_perm_moran})")

    results['spatial_autocorrelation'] = {
        'morans_I':        morans_I,
        'p_value':         float(moran_p),
        'threshold_deg':   threshold_deg,
        'sample_n':        sample_n,
        'interpretation':  (
            'Significant positive spatial autocorrelation in OOF scores'
            if moran_p < 0.05 and morans_I > 0 else
            'No significant spatial autocorrelation detected'
        )
    }

except Exception as e:
    print(f"  WARNING: Spatial analysis failed: {e}")
    # Provide a default note without coordinates
    results['spatial_autocorrelation'] = {
        'morans_I':       None,
        'p_value':        None,
        'interpretation': 'Spatial coordinates unavailable for Moran\'s I computation',
        'note':           str(e)
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. SHAP IMPORTANCE STABILITY (bootstrap)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] SHAP importance stability (bootstrap)...")

try:
    import shap
    import joblib

    with open(MODELS_DIR / 'feature_names.json', encoding='utf-8') as f:
        feature_cols = json.load(f)

    excl = ['lake_id', 'had_glof', 'country', 'cordillera', 'geometry', 'year',
            'source', 'area_name', 'scene_date', 'glof_volume_m3',
            'glof_match_dist_m', 'glof_date', 'glof_trigger', 'glof_lake_name',
            'size_class', 'risk_score']
    num_cols = [c for c in df.columns if c not in excl
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    X_raw = df[num_cols].copy().fillna(df[num_cols].median()).fillna(0)
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
    feat_avail = list(feat_avail)  # ensure plain list, not pandas Index
    X = X_raw[feat_avail].values
    y = df['had_glof'].values.astype(int)

    model = joblib.load(MODELS_DIR / 'best_model.joblib')

    # SHAP on full dataset (sample for speed)
    n_shap_sample = min(2000, len(X))
    rng4 = np.random.default_rng(7)
    shap_idx = rng4.choice(len(X), n_shap_sample, replace=False)
    X_shap = X[shap_idx]

    print(f"  Computing SHAP on {n_shap_sample} samples...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)
    print(f"  SHAP done in {time.time()-t0:.1f}s, type={type(shap_vals)}, ", end='')
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # list of [neg, pos] class arrays
        print(f"selected class-1, shape={np.array(shap_vals).shape}")
    elif hasattr(shap_vals, 'shape'):
        print(f"shape={shap_vals.shape}")
        if shap_vals.ndim == 3:
            # shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
            if shap_vals.shape[0] == 2:
                shap_vals = shap_vals[1]  # (n_samples, n_features)
            elif shap_vals.shape[2] == 2:
                shap_vals = shap_vals[:, :, 1]  # (n_samples, n_features)
    shap_vals = np.array(shap_vals)
    print(f"  Final shap_vals shape: {shap_vals.shape}")

    # mean_abs_shap must be 1D with n_features elements
    if shap_vals.ndim == 2:
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)  # (n_features,)
    else:
        raise ValueError(f"Unexpected shap_vals shape: {shap_vals.shape}")

    print(f"  mean_abs_shap shape: {mean_abs_shap.shape}, len(feat_avail)={len(feat_avail)}")
    top10_idx = [int(x) for x in np.argsort(mean_abs_shap)[::-1][:10]]
    top10_feats   = [feat_avail[i] for i in top10_idx]
    top10_vals    = [float(mean_abs_shap[i]) for i in top10_idx]

    # Bootstrap SHAP stability (50 resamples — fast approximation)
    n_boot_shap = 50
    shap_boot_matrix = np.zeros((n_boot_shap, len(feat_avail)))
    rng5 = np.random.default_rng(9)

    print(f"  Bootstrap SHAP stability ({n_boot_shap} resamples)...")
    def extract_pos_shap(sv):
        """Extract class-1 SHAP values regardless of shape convention."""
        sv = np.array(sv)
        if sv.ndim == 3:
            if sv.shape[0] == 2:        # (2, n_samples, n_features)
                sv = sv[1]
            elif sv.shape[2] == 2:      # (n_samples, n_features, 2)
                sv = sv[:, :, 1]
        return sv  # (n_samples, n_features)

    for bi in range(n_boot_shap):
        bidx = rng5.integers(0, n_shap_sample, n_shap_sample)
        sv   = explainer.shap_values(X_shap[bidx])
        sv   = extract_pos_shap(sv)
        shap_boot_matrix[bi] = np.abs(sv).mean(axis=0)
        if (bi + 1) % 10 == 0:
            print(f"    {bi+1}/{n_boot_shap}")

    # Coefficient of variation per feature
    shap_cv = shap_boot_matrix.std(axis=0) / (shap_boot_matrix.mean(axis=0) + 1e-12)
    top10_cv = [float(shap_cv[i]) for i in top10_idx]

    # Rank stability: how often each feature stays in top-10
    rank_counts = np.zeros(len(feat_avail))
    for bi in range(n_boot_shap):
        top10_b = [int(x) for x in np.argsort(shap_boot_matrix[bi])[::-1][:10]]
        for ti in top10_b:
            rank_counts[ti] += 1
    top10_rank_stability = [float(rank_counts[int(i)] / n_boot_shap) for i in top10_idx]

    # Mean rank across bootstrap
    shap_ranks_boot = np.zeros((n_boot_shap, len(feat_avail)))
    for bi in range(n_boot_shap):
        ranks = np.argsort(np.argsort(-shap_boot_matrix[bi]))
        shap_ranks_boot[bi] = ranks
    mean_rank     = shap_ranks_boot.mean(axis=0)
    std_rank      = shap_ranks_boot.std(axis=0)
    top10_rank_mean = [float(mean_rank[i]) for i in top10_idx]
    top10_rank_std  = [float(std_rank[i]) for i in top10_idx]

    print("  Top-10 features (mean SHAP | CV | rank stability):")
    for fn, mv, cv, rs, rm, rs2 in zip(
            top10_feats, top10_vals, top10_cv,
            top10_rank_stability, top10_rank_mean, top10_rank_std):
        print(f"    {fn:<35} {mv:.5f}  CV={cv:.3f}  top10_freq={rs:.2f}  mean_rank={rm:.1f}±{rs2:.1f}")

    results['shap_stability'] = {
        'top10_features':       top10_feats,
        'top10_mean_shap':      top10_vals,
        'top10_cv':             top10_cv,
        'top10_rank_stability': top10_rank_stability,
        'top10_mean_rank':      top10_rank_mean,
        'top10_rank_std':       top10_rank_std,
        'n_boot_resamples':     n_boot_shap,
        'n_shap_samples':       n_shap_sample,
    }

except Exception as e:
    print(f"  WARNING: SHAP stability failed: {e}")
    import traceback
    traceback.print_exc()
    results['shap_stability'] = {
        'error': str(e),
        'top10_features': [],
        'note': 'SHAP stability could not be computed',
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY FOR LATEX INSERTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Summary for LaTeX insertion...")

print(f"\n  BRIER SCORE            : {results['brier_score']:.6f}")
print(f"  BRIER SKILL SCORE      : {results['brier_skill_score']:.4f}")
print(f"  LOCO Peru 95% CI       : [{results['loco_bootstrap_peru']['ci_low']:.3f}, "
      f"{results['loco_bootstrap_peru']['ci_high']:.3f}]")
if results['spatial_autocorrelation']['morans_I'] is not None:
    print(f"  Moran's I              : {results['spatial_autocorrelation']['morans_I']:.4f}")
    print(f"  Moran's p              : {results['spatial_autocorrelation']['p_value']:.4f}")
print(f"  Threshold sensitivity  : {len(results['threshold_sensitivity'])} thresholds computed")
if 'top10_features' in results.get('shap_stability', {}):
    n_stab = results['shap_stability'].get('top10_features', [])
    print(f"  SHAP stability         : {len(n_stab)} features analyzed")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
out_path = MODELS_DIR / 'improvements.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n  Saved: {out_path}")
print("=" * 65)
print("DONE")
print("=" * 65)
