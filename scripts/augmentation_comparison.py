# -*- coding: utf-8 -*-
"""
augmentation_comparison.py
──────────────────────────
Compares original (n+=17) vs temporally-augmented (n+=538) labelling
for GLOF susceptibility with BalancedRandomForestClassifier.

Outputs
-------
  figures/publication/fig8_augmentation_comparison.png
  models/augmentation_results.json
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
from pathlib import Path
BASE       = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE / "data/processed/labeled/training_data.csv"
MODEL_PATH = BASE / "models/balancedrf.joblib"
FEAT_PATH  = BASE / "models/feature_names.json"
OUT_FIG    = BASE / "figures/publication/fig8_augmentation_comparison.png"
OUT_JSON   = BASE / "models/augmentation_results.json"

SEED = 42
N_BOOTSTRAP = 2000
N_SPLITS = 5

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}  |  had_glof=1: {df['had_glof'].sum()}")

# ── 2. Engineer missing log features ───────────────────────────────────────
EPS = 1e-6

def safe_log(series):
    return np.log1p(series.clip(lower=0) + EPS)

df["log_area_m2"]               = safe_log(df["area_m2"])
df["log_perimeter_m"]           = safe_log(df["perimeter_m"])
df["log_volume_m3"]             = safe_log(df["volume_m3"])
df["log_equivalent_diameter_m"] = safe_log(df["equivalent_diameter_m"])
df["log_potential_energy"]      = safe_log(df["potential_energy"])
df["log_depth_m"]               = safe_log(df["depth_m"])
df["log_dam_height"]            = safe_log(df["dam_height"])
df["log_area_2017_est"]         = safe_log(df["area_2017_est"])
df["log_area_2025_est"]         = safe_log(df["area_2025_est"])
df["log_total_growth_m2"]       = safe_log(df["total_growth_m2"].abs())
df["rel_growth"]                = (
    (df["area_2025_est"] - df["area_2017_est"]) /
    (df["area_2017_est"].clip(lower=EPS))
).clip(-10, 100)
df["slope_x_logarea"]           = df["slope_mean"] * df["log_area_m2"]

# ── 3. Build feature matrix ─────────────────────────────────────────────────
with open(FEAT_PATH) as f:
    feature_names = json.load(f)

# Confirm all features now exist
missing = [c for c in feature_names if c not in df.columns]
if missing:
    raise ValueError(f"Still missing features: {missing}")

print(f"  Using {len(feature_names)} features.")
X = df[feature_names].values
X = np.nan_to_num(X, nan=0.0)

# ── 4. Build labels ─────────────────────────────────────────────────────────
y_orig = df["had_glof"].values.copy()

# Augmented: all rows of GLOF-source lake_ids -> positive
glof_lake_ids = df.loc[df["had_glof"] == 1, "lake_id"].unique()
print(f"  Unique GLOF source lake_ids: {len(glof_lake_ids)}")

y_aug = y_orig.copy()
aug_mask = df["lake_id"].isin(glof_lake_ids)
y_aug[aug_mask] = 1

print(f"  Original  n+={y_orig.sum():,}  n-={( y_orig==0).sum():,}")
print(f"  Augmented n+={y_aug.sum():,}  n-={(y_aug==0).sum():,}")

# ── 5. Load BRF params ──────────────────────────────────────────────────────
ref_brf = joblib.load(MODEL_PATH)
brf_params = ref_brf.get_params()
# Ensure reproducibility
brf_params["random_state"] = SEED

def make_brf():
    return BalancedRandomForestClassifier(**brf_params)

# ── 6. OOF cross-validation ─────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

print("\nRunning OOF CV — Original …")
oof_orig = cross_val_predict(
    make_brf(), X, y_orig, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

print("Running OOF CV — Augmented …")
oof_aug = cross_val_predict(
    make_brf(), X, y_aug, cv=cv, method="predict_proba", n_jobs=-1
)[:, 1]

# ── 7. Helper: bootstrap AUC CI ─────────────────────────────────────────────
def bootstrap_auc(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_boot):
        ps = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ns = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ps, ns])
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def bootstrap_prauc(y_true, y_score, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    praucs = []
    for _ in range(n_boot):
        ps = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ns = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ps, ns])
        try:
            praucs.append(average_precision_score(y_true[idx], y_score[idx]))
        except Exception:
            pass
    praucs = np.array(praucs)
    return float(np.mean(praucs)), float(np.percentile(praucs, 2.5)), float(np.percentile(praucs, 97.5))

# ── 8. Youden threshold ──────────────────────────────────────────────────────
def youden_threshold_recall(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best_idx = np.argmax(j)
    thr = thresholds[best_idx]
    y_pred = (y_score >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    recall = tp / (tp + fn + 1e-12)
    return float(thr), float(recall)

# ── 9. Jackknife leave-one-lake-out (over the 17 GLOF source lake_ids) ──────
def jackknife_auc(X_all, y_train_labels, y_eval_labels,
                  lake_ids_array, source_ids, seed=SEED):
    """
    For each source lake_id:
      - Train BRF on all rows NOT belonging to that lake_id (using y_train_labels)
      - Predict on the held-out rows of that lake_id
      - Evaluate AUC using y_eval_labels (original labels) so both conditions
        are evaluated on the same ground truth (1 = actual GLOF event year).
    This ensures fair comparison: for augmented, we train with richer labels
    but evaluate against the same single-event truth as original.
    """
    aucs = []
    for lid in source_ids:
        mask_test  = lake_ids_array == lid
        mask_train = ~mask_test

        X_tr = X_all[mask_train]
        y_tr = y_train_labels[mask_train]
        X_te = X_all[mask_test]
        y_te = y_eval_labels[mask_test]   # always original labels for evaluation

        if y_te.sum() == 0 or (y_te == 0).sum() == 0:
            # Cannot compute AUC without both classes in eval set
            aucs.append(float("nan"))
            continue

        clf = make_brf()
        clf.fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]
        try:
            auc = roc_auc_score(y_te, scores)
        except Exception:
            auc = float("nan")
        aucs.append(auc)
    return aucs

lake_ids_arr = df["lake_id"].values

print("\nRunning Jackknife — Original …")
jk_orig = jackknife_auc(X, y_orig, y_orig, lake_ids_arr, glof_lake_ids)

print("Running Jackknife — Augmented …")
# Train with augmented labels, evaluate with original labels for fair comparison
jk_aug  = jackknife_auc(X, y_aug, y_orig, lake_ids_arr, glof_lake_ids)

# ── 10. Rank percentile of GLOF source lakes ─────────────────────────────────
def rank_percentile(y_true, y_score):
    """Average percentile rank of positives among all scores."""
    n = len(y_score)
    ranks = np.argsort(np.argsort(y_score))  # 0-based rank ascending
    pos_ranks = ranks[y_true == 1]
    # percentile: rank / (n-1) * 100
    return float(np.mean(pos_ranks / (n - 1) * 100))

# ── 11. Compute all metrics ──────────────────────────────────────────────────
print("\nComputing metrics …")

def compute_metrics(y_true, y_score, label):
    auc_mean, auc_lo, auc_hi = bootstrap_auc(y_true, y_score)
    prauc_mean, prauc_lo, prauc_hi = bootstrap_prauc(y_true, y_score)
    prevalence = float(y_true.mean())
    lift = prauc_mean / prevalence if prevalence > 0 else float("nan")
    thr, recall = youden_threshold_recall(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    rp = rank_percentile(y_true, y_score)
    print(f"  [{label}] ROC-AUC={auc_mean:.4f} [{auc_lo:.4f}–{auc_hi:.4f}]  "
          f"PR-AUC={prauc_mean:.4f}  Lift={lift:.2f}  "
          f"Recall@Youden={recall:.3f}  MCC={mcc:.3f}  RankPct={rp:.1f}%")
    return dict(
        roc_auc=auc_mean, roc_auc_lo=auc_lo, roc_auc_hi=auc_hi,
        pr_auc=prauc_mean, pr_auc_lo=prauc_lo, pr_auc_hi=prauc_hi,
        prevalence=prevalence, lift=lift,
        youden_threshold=thr, recall_youden=recall,
        mcc=mcc, rank_percentile=rp,
        n_positive=int(y_true.sum()), n_total=int(len(y_true))
    )

metrics_orig = compute_metrics(y_orig, oof_orig, "Original")
metrics_aug  = compute_metrics(y_aug,  oof_aug,  "Augmented")

metrics_orig["jackknife_auc"] = [float(v) for v in jk_orig]
metrics_aug["jackknife_auc"]  = [float(v) for v in jk_aug]

# ROC / PR curves
fpr_o, tpr_o, _ = roc_curve(y_orig, oof_orig)
fpr_a, tpr_a, _ = roc_curve(y_aug,  oof_aug)
prec_o, rec_o, _ = precision_recall_curve(y_orig, oof_orig)
prec_a, rec_a, _ = precision_recall_curve(y_aug,  oof_aug)

# ── 12. Save JSON ────────────────────────────────────────────────────────────
results = {
    "original":  metrics_orig,
    "augmented": metrics_aug,
    "n_glof_source_lakes": int(len(glof_lake_ids)),
    "jackknife_orig_mean": float(np.nanmean(jk_orig)),
    "jackknife_aug_mean":  float(np.nanmean(jk_aug)),
}
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved -> {OUT_JSON}")

# ── 13. Figure ───────────────────────────────────────────────────────────────
print("Generating figure …")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

BLUE   = "#2166ac"
ORANGE = "#d6604d"
ALPHA  = 0.20
LW     = 2.2

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    "Temporal Label Augmentation: Impact on GLOF Susceptibility Model",
    fontsize=13, fontweight="bold", y=1.01
)

# ── (a) ROC curves ──────────────────────────────────────────────────────────
ax = axes[0, 0]

# Bootstrap CI bands for ROC (sampled)
def bootstrap_roc_band(y_true, y_score, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    common_fpr = np.linspace(0, 1, 200)
    tpr_mat = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
            tpr_interp = np.interp(common_fpr, fpr_b, tpr_b)
            tpr_mat.append(tpr_interp)
        except Exception:
            pass
    tpr_mat = np.array(tpr_mat)
    return common_fpr, np.percentile(tpr_mat, 2.5, axis=0), np.percentile(tpr_mat, 97.5, axis=0)

cfpr_o, tpr_lo_o, tpr_hi_o = bootstrap_roc_band(y_orig, oof_orig)
cfpr_a, tpr_lo_a, tpr_hi_a = bootstrap_roc_band(y_aug,  oof_aug)

ax.plot([0,1],[0,1], ":", color="grey", lw=1.2, label="Chance")
ax.fill_between(cfpr_o, tpr_lo_o, tpr_hi_o, alpha=ALPHA, color=BLUE)
ax.fill_between(cfpr_a, tpr_lo_a, tpr_hi_a, alpha=ALPHA, color=ORANGE)
ax.plot(fpr_o, tpr_o, color=BLUE,   lw=LW,
        label=f"Original  (AUC={metrics_orig['roc_auc']:.3f} "
              f"[{metrics_orig['roc_auc_lo']:.3f}–{metrics_orig['roc_auc_hi']:.3f}])")
ax.plot(fpr_a, tpr_a, color=ORANGE, lw=LW,
        label=f"Augmented (AUC={metrics_aug['roc_auc']:.3f} "
              f"[{metrics_aug['roc_auc_lo']:.3f}–{metrics_aug['roc_auc_hi']:.3f}])")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("(a) ROC Curves")
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(0,1); ax.set_ylim(0,1)

# ── (b) Bar chart of key metrics ────────────────────────────────────────────
ax = axes[0, 1]
metric_labels = ["ROC-AUC", "PR-AUC / 0.1\n(scaled)", "Recall @ Youden (%)"]

orig_vals = [
    metrics_orig["roc_auc"],
    metrics_orig["pr_auc"] / 0.1,
    metrics_orig["recall_youden"] * 100
]
aug_vals = [
    metrics_aug["roc_auc"],
    metrics_aug["pr_auc"] / 0.1,
    metrics_aug["recall_youden"] * 100
]
orig_errs = [
    [(metrics_orig["roc_auc"] - metrics_orig["roc_auc_lo"]),
     (metrics_orig["roc_auc_hi"] - metrics_orig["roc_auc"])],
    [(metrics_orig["pr_auc"] - metrics_orig["pr_auc_lo"]) / 0.1,
     (metrics_orig["pr_auc_hi"] - metrics_orig["pr_auc"]) / 0.1],
    [0, 0]
]
aug_errs = [
    [(metrics_aug["roc_auc"] - metrics_aug["roc_auc_lo"]),
     (metrics_aug["roc_auc_hi"] - metrics_aug["roc_auc"])],
    [(metrics_aug["pr_auc"] - metrics_aug["pr_auc_lo"]) / 0.1,
     (metrics_aug["pr_auc_hi"] - metrics_aug["pr_auc"]) / 0.1],
    [0, 0]
]

x = np.arange(len(metric_labels))
w = 0.35
bars_o = ax.bar(x - w/2, orig_vals, w, color=BLUE,   label=f"Original (n+={metrics_orig['n_positive']})",
                yerr=np.array(orig_errs).T, capsize=4, error_kw={"elinewidth": 1.5})
bars_a = ax.bar(x + w/2, aug_vals,  w, color=ORANGE, label=f"Augmented (n+={metrics_aug['n_positive']})",
                yerr=np.array(aug_errs).T, capsize=4, error_kw={"elinewidth": 1.5})

ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylabel("Value"); ax.set_title("(b) Key Metric Comparison")
ax.legend(fontsize=8.5)
ax.set_ylim(0, max(max(orig_vals), max(aug_vals)) * 1.25)

# Annotate bars
for bar in bars_o:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=7.5, color=BLUE)
for bar in bars_a:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
            ha="center", va="bottom", fontsize=7.5, color=ORANGE)

# ── (c) Jackknife AUC stability boxplot ─────────────────────────────────────
ax = axes[1, 0]

jk_df = pd.DataFrame({
    "AUC": jk_orig + jk_aug,
    "Condition": (["Original"] * len(jk_orig)) + (["Augmented"] * len(jk_aug))
})
palette = {"Original": BLUE, "Augmented": ORANGE}
sns.boxplot(data=jk_df, x="Condition", y="AUC", palette=palette,
            width=0.45, linewidth=1.5, fliersize=5, ax=ax)

# Mean horizontal lines
for i, (vals, color) in enumerate([(jk_orig, BLUE), (jk_aug, ORANGE)]):
    m = np.nanmean(vals)
    ax.hlines(m, i - 0.22, i + 0.22, colors=color, linestyles="--", lw=1.8)
    ax.text(i + 0.25, m, f"μ={m:.3f}", va="center", fontsize=8.5, color=color)

ax.set_xlabel(""); ax.set_ylabel("Leave-One-Lake-Out AUC")
ax.set_title("(c) Jackknife AUC Stability\n(17 GLOF source lakes)")
ax.set_ylim(0, 1.05)

# ── (d) Precision-Recall curves ─────────────────────────────────────────────
ax = axes[1, 1]
prev_o = float(y_orig.mean())
prev_a = float(y_aug.mean())

ax.hlines(prev_o, 0, 1, colors=BLUE,   linestyles=":", lw=1.2, alpha=0.6,
          label=f"Baseline Original (prev={prev_o:.4f})")
ax.hlines(prev_a, 0, 1, colors=ORANGE, linestyles=":", lw=1.2, alpha=0.6,
          label=f"Baseline Augmented (prev={prev_a:.4f})")
ax.plot(rec_o, prec_o, color=BLUE,   lw=LW,
        label=f"Original  (AP={metrics_orig['pr_auc']:.3f})")
ax.plot(rec_a, prec_a, color=ORANGE, lw=LW,
        label=f"Augmented (AP={metrics_aug['pr_auc']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("(d) Precision-Recall Curves")
ax.legend(fontsize=8.0, loc="upper right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved -> {OUT_FIG}")

# ── 14. Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("AUGMENTATION COMPARISON SUMMARY")
print("=" * 68)
print(f"  n_glof_source_lakes : {len(glof_lake_ids)}")
print(f"  n+ Original         : {metrics_orig['n_positive']}")
print(f"  n+ Augmented        : {metrics_aug['n_positive']}")
print()
print(f"  ROC-AUC Original    : {metrics_orig['roc_auc']:.4f}  "
      f"95%CI [{metrics_orig['roc_auc_lo']:.4f}–{metrics_orig['roc_auc_hi']:.4f}]")
print(f"  ROC-AUC Augmented   : {metrics_aug['roc_auc']:.4f}  "
      f"95%CI [{metrics_aug['roc_auc_lo']:.4f}–{metrics_aug['roc_auc_hi']:.4f}]")
print()
print(f"  PR-AUC  Original    : {metrics_orig['pr_auc']:.4f}  "
      f"Lift={metrics_orig['lift']:.2f}x")
print(f"  PR-AUC  Augmented   : {metrics_aug['pr_auc']:.4f}  "
      f"Lift={metrics_aug['lift']:.2f}x")
print()
print(f"  Recall@Youden Orig  : {metrics_orig['recall_youden']:.3f}")
print(f"  Recall@Youden Aug   : {metrics_aug['recall_youden']:.3f}")
print()
print(f"  MCC  Original       : {metrics_orig['mcc']:.3f}")
print(f"  MCC  Augmented      : {metrics_aug['mcc']:.3f}")
print()
print(f"  JK-AUC mean Orig    : {np.nanmean(jk_orig):.4f}  "
      f"std={np.nanstd(jk_orig):.4f}")
print(f"  JK-AUC mean Aug     : {np.nanmean(jk_aug):.4f}  "
      f"std={np.nanstd(jk_aug):.4f}")
print()
print(f"  Rank Percentile Orig: {metrics_orig['rank_percentile']:.1f}%")
print(f"  Rank Percentile Aug : {metrics_aug['rank_percentile']:.1f}%")
print("=" * 68)
