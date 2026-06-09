"""
Compute statistical validation for manuscript:
- DeLong test: BalancedRF vs LightGBM ROC-AUC
- Mann-Whitney U tests for top SHAP features (GLOF vs non-GLOF)
- Effect sizes (rank-biserial correlation)

Run from project root:
    python scripts/compute_statistical_tests.py
"""

import json as json_mod
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data/processed/labeled/training_data.csv"
FEATURES_FILE = ROOT / "models/feature_names.json"
OOF_FILE = ROOT / "models/oof_curves.pkl"

# Top 10 SHAP features (from NB17 analysis)
TOP_SHAP_FEATURES = [
    "dist_glacier_m",
    "slope_x_logarea",
    "volume_m3",
    "area_depth_ratio",
    "depth_ensemble_mean",
    "log_area_m2",
    "depth_std_m",
    "depth_cook_quincey_2015",
    "depth_ensemble_std",
    "elongation",
]

BONFERRONI_N = 10
ALPHA_CORRECTED = 0.05 / BONFERRONI_N


def rank_biserial(x1, x2):
    """Rank-biserial correlation effect size for Mann-Whitney U."""
    n1, n2 = len(x1), len(x2)
    u_stat, _ = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    r_b = 1 - (2 * u_stat) / (n1 * n2)
    return r_b


def delong_roc_variance(ground_truth, predictions):
    """DeLong variance estimator for a single ROC-AUC."""
    n_pos = ground_truth.sum()
    n_neg = len(ground_truth) - n_pos
    pos_scores = predictions[ground_truth == 1]
    neg_scores = predictions[ground_truth == 0]

    # placement values
    v10 = np.array([np.mean(s > neg_scores) + 0.5 * np.mean(s == neg_scores)
                    for s in pos_scores])
    v01 = np.array([np.mean(p < pos_scores) + 0.5 * np.mean(p == pos_scores)
                    for p in neg_scores])
    auc = v10.mean()
    s10 = np.var(v10, ddof=1) / n_pos
    s01 = np.var(v01, ddof=1) / n_neg
    return auc, s10 + s01


def delong_test(y_true, pred1, pred2):
    """DeLong test for paired ROC-AUC comparison."""
    auc1, var1 = delong_roc_variance(y_true, pred1)
    auc2, var2 = delong_roc_variance(y_true, pred2)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    pos1 = pred1[y_true == 1]
    pos2 = pred2[y_true == 1]
    neg1 = pred1[y_true == 0]
    neg2 = pred2[y_true == 0]

    v10_1 = np.array([np.mean(s > neg1) + 0.5 * np.mean(s == neg1) for s in pos1])
    v10_2 = np.array([np.mean(s > neg2) + 0.5 * np.mean(s == neg2) for s in pos2])
    v01_1 = np.array([np.mean(p < pos1) + 0.5 * np.mean(p == pos1) for p in neg1])
    v01_2 = np.array([np.mean(p < pos2) + 0.5 * np.mean(p == pos2) for p in neg2])

    s10_cov = np.cov(v10_1, v10_2, ddof=1)[0, 1] / n_pos
    s01_cov = np.cov(v01_1, v01_2, ddof=1)[0, 1] / n_neg

    var_diff = var1 + var2 - 2 * (s10_cov + s01_cov)
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    y = df["had_glof"].values

    with open(FEATURES_FILE) as f:
        feature_names = json_mod.load(f)

    # ── Mann-Whitney U tests ──────────────────────────────────────────────
    print(f"\nMann-Whitney U tests (Bonferroni α' = {ALPHA_CORRECTED:.4f})")
    print(f"GLOF lakes: {y.sum()} | Non-GLOF: {(y==0).sum()}")
    print(f"{'Feature':<28} {'Med GLOF':>12} {'Med non-GLOF':>13} {'U':>10} {'p (corrected)':>15} {'r_b':>8} {'Sig?':>6}")
    print("-" * 100)

    results = []
    for feat in TOP_SHAP_FEATURES:
        if feat not in df.columns:
            print(f"  {feat}: NOT FOUND in dataset")
            continue
        x_pos = df.loc[y == 1, feat].dropna().values
        x_neg = df.loc[y == 0, feat].dropna().values

        u_stat, p_raw = stats.mannwhitneyu(x_pos, x_neg, alternative="two-sided")
        p_corr = min(p_raw * BONFERRONI_N, 1.0)
        rb = rank_biserial(x_pos, x_neg)
        med_pos = np.median(x_pos)
        med_neg = np.median(x_neg)
        sig = "✓" if p_corr < 0.05 else ""

        results.append({
            "feature": feat,
            "median_glof": med_pos,
            "median_nonglof": med_neg,
            "U": u_stat,
            "p_corrected": p_corr,
            "r_b": rb,
            "significant": sig,
        })
        print(f"  {feat:<28} {med_pos:>12.3g} {med_neg:>13.3g} {u_stat:>10.0f} {p_corr:>15.4f} {rb:>8.3f} {sig:>6}")

    # ── DeLong test ───────────────────────────────────────────────────────
    print("\nDeLong test: BalancedRF vs LightGBM")
    try:
        import pickle
        with open(ROOT / "models/oof_curves.pkl", "rb") as f:
            oof = pickle.load(f)

        # Expected keys: {'BalancedRF': proba_array, 'LightGBM': proba_array, ...}
        if isinstance(oof, dict) and "BalancedRF" in oof and "LightGBM" in oof:
            pred_brf = np.array(oof["BalancedRF"])
            pred_lgb = np.array(oof["LightGBM"])

            # Align length if needed
            min_len = min(len(pred_brf), len(pred_lgb), len(y))
            auc1, auc2, z, p = delong_test(y[:min_len], pred_brf[:min_len], pred_lgb[:min_len])
            print(f"  BalancedRF AUC = {auc1:.4f}")
            print(f"  LightGBM   AUC = {auc2:.4f}")
            print(f"  DeLong z = {z:.3f}, p = {p:.4f}")
        else:
            print(f"  OOF file keys: {list(oof.keys()) if isinstance(oof, dict) else type(oof)}")
            print("  Could not extract paired OOF predictions — check oof_curves.pkl structure")
    except Exception as e:
        print(f"  Could not load OOF curves: {e}")

    # ── Save results ──────────────────────────────────────────────────────
    out = ROOT / "models/statistical_tests.json"
    with open(out, "w") as f:
        json_mod.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    print("\nCOPY THESE VALUES INTO TABLE 5 (tab:mannwhitney) IN THE MANUSCRIPT.")


if __name__ == "__main__":
    main()
