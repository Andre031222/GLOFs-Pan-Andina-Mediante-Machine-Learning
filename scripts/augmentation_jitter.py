# -*- coding: utf-8 -*-
"""
augmentation_jitter.py
──────────────────────
Adds a 'jitter' condition to the augmentation comparison:
  - Gaussian feature jitter applied WITHIN each OOF fold (training set only)
  - Evaluation always on the 17 original positive labels
  - Sigma = 0.05 * std per feature (computed on fold training set)
  - N_JITTER = 10 synthetic copies per positive
  - After jitter: clip to [min_train, max_train] per feature

Updates:
  models/augmentation_results.json  (adds 'jitter' key)
  figures/publication/fig8_augmentation_comparison.png  (3 conditions, 2x2)
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
from pathlib import Path
BASE      = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data/processed/labeled/training_data.csv"
MODEL_PATH= BASE / "models/balancedrf.joblib"
FEAT_PATH = BASE / "models/feature_names.json"
OUT_FIG   = BASE / "figures/publication/fig8_augmentation_comparison.png"
OUT_JSON  = BASE / "models/augmentation_results.json"

SEED      = 42
N_BOOTSTRAP = 2000
N_SPLITS  = 5
N_JITTER  = 10
SIGMA_FRAC = 0.05   # sigma = SIGMA_FRAC * std(feature in training fold)

# ── 1. Load data ────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}  |  had_glof=1: {df['had_glof'].sum()}")

# ── 2. Engineer log/derived features ────────────────────────────────────────
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
df["rel_growth"] = (
    (df["area_2025_est"] - df["area_2017_est"]) /
    (df["area_2017_est"].clip(lower=EPS))
).clip(-10, 100)
df["slope_x_logarea"] = df["slope_mean"] * df["log_area_m2"]

# ── 3. Feature matrix & labels ──────────────────────────────────────────────
with open(FEAT_PATH) as fh:
    feature_names = json.load(fh)

missing = [c for c in feature_names if c not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

print(f"  Using {len(feature_names)} features.")
X_df = df[feature_names].copy()
y_orig = df["had_glof"].values.copy()

lake_ids_arr = df["lake_id"].values
glof_lake_ids = df.loc[df["had_glof"] == 1, "lake_id"].unique()
print(f"  Unique GLOF source lake_ids: {len(glof_lake_ids)}")

# ── 4. BRF factory ──────────────────────────────────────────────────────────
ref_brf = joblib.load(MODEL_PATH)
brf_params = ref_brf.get_params()
brf_params["random_state"] = SEED

def make_brf():
    return BalancedRandomForestClassifier(**brf_params)

# ── 5. OOF with Gaussian Jitter (training set only) ─────────────────────────
print("\nRunning OOF CV -- Gaussian Jitter ...")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_jitter = np.zeros(len(y_orig))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_df, y_orig)):
    rng = np.random.default_rng(SEED + fold_idx * 100)

    X_tr_raw = X_df.iloc[train_idx].copy()
    y_tr = y_orig[train_idx]
    X_val = X_df.iloc[val_idx].copy()
    y_val = y_orig[val_idx]

    # Fill NaN with median of training fold
    train_medians = X_tr_raw.median()
    X_tr_raw = X_tr_raw.fillna(train_medians)
    X_val = X_val.fillna(train_medians)

    # Compute per-feature stats on training fold for sigma and clamping
    train_std  = X_tr_raw.std(ddof=1).values       # shape (n_features,)
    train_min  = X_tr_raw.min().values
    train_max  = X_tr_raw.max().values
    sigma      = SIGMA_FRAC * train_std             # per-feature sigma

    # Identify positives in training fold
    pos_mask_tr = (y_tr == 1)
    X_pos = X_tr_raw.values[pos_mask_tr]            # (k, n_features)
    n_pos = X_pos.shape[0]

    # Generate N_JITTER synthetic copies per positive
    noise = rng.normal(
        loc=0.0,
        scale=sigma,                                # broadcast over rows
        size=(n_pos * N_JITTER, len(feature_names))
    )
    X_synth = np.tile(X_pos, (N_JITTER, 1)) + noise
    # Clip to [min_train, max_train] per feature
    X_synth = np.clip(X_synth, train_min, train_max)
    y_synth = np.ones(n_pos * N_JITTER, dtype=y_tr.dtype)

    # Augmented training set
    X_tr_aug = np.vstack([X_tr_raw.values, X_synth])
    y_tr_aug = np.concatenate([y_tr, y_synth])

    clf = make_brf()
    clf.fit(X_tr_aug, y_tr_aug)
    oof_jitter[val_idx] = clf.predict_proba(X_val.values)[:, 1]

    n_pos_tr = pos_mask_tr.sum()
    print(f"  Fold {fold_idx+1}: train n+={n_pos_tr} (+{n_pos_tr*N_JITTER} synth) "
          f"-> total n+={n_pos_tr + n_pos_tr*N_JITTER}  val n+={y_val.sum()}")

# ── 6. Helper functions ──────────────────────────────────────────────────────
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

def rank_percentile(y_true, y_score):
    n = len(y_score)
    ranks = np.argsort(np.argsort(y_score))
    pos_ranks = ranks[y_true == 1]
    return float(np.mean(pos_ranks / (n - 1) * 100))

def compute_metrics(y_true, y_score, label):
    auc_mean, auc_lo, auc_hi = bootstrap_auc(y_true, y_score)
    prauc_mean, prauc_lo, prauc_hi = bootstrap_prauc(y_true, y_score)
    prevalence = float(y_true.mean())
    lift = prauc_mean / prevalence if prevalence > 0 else float("nan")
    thr, recall = youden_threshold_recall(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    mcc = float(matthews_corrcoef(y_true, y_pred))
    rp = rank_percentile(y_true, y_score)
    print(f"  [{label}] ROC-AUC={auc_mean:.4f} [{auc_lo:.4f}-{auc_hi:.4f}]  "
          f"PR-AUC={prauc_mean:.4f}  Lift={lift:.2f}  "
          f"Recall@Youden={recall:.3f}  RankPct={rp:.1f}%")
    return dict(
        roc_auc=auc_mean, roc_auc_lo=auc_lo, roc_auc_hi=auc_hi,
        pr_auc=prauc_mean, pr_auc_lo=prauc_lo, pr_auc_hi=prauc_hi,
        prevalence=prevalence, lift=lift,
        youden_threshold=float(thr), recall_youden=float(recall),
        mcc=float(mcc), rank_percentile=float(rp),
        n_positive=int(y_true.sum()), n_total=int(len(y_true))
    )

# ── 7. Jackknife leave-one-lake-out with jitter ─────────────────────────────
def jackknife_auc_jitter(y_true, lake_ids_array, source_ids, seed=SEED):
    """
    Leave-one-source-lake-out jackknife.
    Train uses jitter augmentation on the reduced training set.
    Evaluate on the held-out lake rows with original labels.
    """
    X_arr = X_df.copy()
    aucs = []
    for lid in source_ids:
        mask_test  = lake_ids_array == lid
        mask_train = ~mask_test

        X_tr_raw = X_arr[mask_train].copy()
        y_tr = y_true[mask_train]
        X_te = X_arr[mask_test].copy()
        y_te = y_true[mask_test]

        if y_te.sum() == 0 or (y_te == 0).sum() == 0:
            aucs.append(float("nan"))
            continue

        rng = np.random.default_rng(seed + hash(lid) % (2**31))

        train_medians = X_tr_raw.median()
        X_tr_raw = X_tr_raw.fillna(train_medians)
        X_te = X_te.fillna(train_medians)

        train_std = X_tr_raw.std(ddof=1).values
        train_min = X_tr_raw.min().values
        train_max = X_tr_raw.max().values
        sigma_jk  = SIGMA_FRAC * train_std

        pos_mask = (y_tr == 1)
        X_pos = X_tr_raw.values[pos_mask]
        n_pos = X_pos.shape[0]

        if n_pos > 0:
            noise = rng.normal(
                loc=0.0, scale=sigma_jk,
                size=(n_pos * N_JITTER, len(feature_names))
            )
            X_synth = np.tile(X_pos, (N_JITTER, 1)) + noise
            X_synth = np.clip(X_synth, train_min, train_max)
            y_synth = np.ones(n_pos * N_JITTER, dtype=y_tr.dtype)
            X_tr_aug = np.vstack([X_tr_raw.values, X_synth])
            y_tr_aug = np.concatenate([y_tr, y_synth])
        else:
            X_tr_aug = X_tr_raw.values
            y_tr_aug = y_tr

        clf = make_brf()
        clf.fit(X_tr_aug, y_tr_aug)
        scores = clf.predict_proba(X_te.values)[:, 1]
        try:
            auc = roc_auc_score(y_te, scores)
        except Exception:
            auc = float("nan")
        aucs.append(auc)
    return aucs

print("\nRunning Jackknife -- Jitter ...")
jk_jitter = jackknife_auc_jitter(y_orig, lake_ids_arr, glof_lake_ids)

# ── 8. Compute jitter metrics (evaluated on original 17 positives) ───────────
print("\nComputing Jitter metrics ...")
metrics_jitter = compute_metrics(y_orig, oof_jitter, "Jitter")
metrics_jitter["jackknife_auc"] = [float(v) for v in jk_jitter]

# ROC / PR curves for jitter
fpr_j, tpr_j, _ = roc_curve(y_orig, oof_jitter)
prec_j, rec_j, _ = precision_recall_curve(y_orig, oof_jitter)

# ── 9. Load existing results and add jitter ──────────────────────────────────
with open(OUT_JSON) as fh:
    results = json.load(fh)

results["jitter"] = metrics_jitter
results["jackknife_jitter_mean"] = float(np.nanmean(jk_jitter))
results["jackknife_jitter_std"]  = float(np.nanstd(jk_jitter))

with open(OUT_JSON, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"\nResults saved -> {OUT_JSON}")

# ── 10. Rebuild figure with 3 conditions ────────────────────────────────────
print("Generating figure ...")

# Pull existing condition data from saved results
metrics_orig = results["original"]
metrics_aug  = results["augmented"]   # temporal augmentation

# Re-derive OOF scores for original and temporal from saved ROC curves
# We need curves -- re-run cross_val_predict for original and temporal
# (they are fast, already done in the old script; re-run here to get curves)
from sklearn.model_selection import cross_val_predict as cvp

y_aug = y_orig.copy()
aug_mask_df = df["lake_id"].isin(glof_lake_ids)
y_aug[aug_mask_df.values] = 1

X_full = X_df.fillna(X_df.median()).values

print("  Re-running OOF for Original and Temporal (for ROC curves) ...")
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_orig  = cvp(make_brf(), X_full, y_orig, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
oof_aug_t = cvp(make_brf(), X_full, y_aug,  cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

fpr_o, tpr_o, _ = roc_curve(y_orig, oof_orig)
fpr_a, tpr_a, _ = roc_curve(y_aug,  oof_aug_t)
prec_o, rec_o, _ = precision_recall_curve(y_orig, oof_orig)

# Bootstrap ROC band helper
def bootstrap_roc_band(y_true, y_score, n_boot=300, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    common_fpr = np.linspace(0, 1, 200)
    tpr_mat = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
            tpr_mat.append(np.interp(common_fpr, fpr_b, tpr_b))
        except Exception:
            pass
    tpr_mat = np.array(tpr_mat)
    return common_fpr, np.percentile(tpr_mat, 2.5, axis=0), np.percentile(tpr_mat, 97.5, axis=0)

cfpr_o, tlo_o, thi_o = bootstrap_roc_band(y_orig, oof_orig)
cfpr_j, tlo_j, thi_j = bootstrap_roc_band(y_orig, oof_jitter)

# ── Figure layout ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

BLUE   = "#2166ac"
GREEN  = "#1a9641"
ORANGE = "#d6604d"
ALPHA  = 0.18
LW     = 2.2

fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
fig.suptitle(
    "GLOF Susceptibility Model: Original vs. Training-Time Augmentation Strategies",
    fontsize=12, fontweight="bold", y=1.005
)
fig.text(
    0.5, 0.998,
    "Gaussian jitter applied within each CV fold (training set only); "
    "evaluation on original 17 positive labels",
    ha="center", va="top", fontsize=8, color="#555555", style="italic"
)

# ── (a) ROC curves ──────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot([0, 1], [0, 1], ":", color="grey", lw=1.2, label="Chance")

ax.fill_between(cfpr_o, tlo_o, thi_o, alpha=ALPHA, color=BLUE)
ax.fill_between(cfpr_j, tlo_j, thi_j, alpha=ALPHA, color=GREEN)

ax.plot(fpr_o, tpr_o, color=BLUE,   lw=LW,
        label=f"Original  AUC={metrics_orig['roc_auc']:.3f} "
              f"[{metrics_orig['roc_auc_lo']:.3f}-{metrics_orig['roc_auc_hi']:.3f}]")
ax.plot(fpr_j, tpr_j, color=GREEN,  lw=LW,
        label=f"Jitter    AUC={metrics_jitter['roc_auc']:.3f} "
              f"[{metrics_jitter['roc_auc_lo']:.3f}-{metrics_jitter['roc_auc_hi']:.3f}]")
ax.plot(fpr_a, tpr_a, color=ORANGE, lw=LW, linestyle="--",
        label=f"Temporal  AUC={metrics_aug['roc_auc']:.3f} "
              f"[{metrics_aug['roc_auc_lo']:.3f}-{metrics_aug['roc_auc_hi']:.3f}]")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("(a) ROC Curves")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ── (b) Grouped bar chart ────────────────────────────────────────────────────
ax = axes[0, 1]

conditions  = ["Original", "Jitter", "Temporal"]
colors_bar  = [BLUE, GREEN, ORANGE]
auc_vals    = [metrics_orig["roc_auc"], metrics_jitter["roc_auc"], metrics_aug["roc_auc"]]
recall_vals = [metrics_orig["recall_youden"]*100,
               metrics_jitter["recall_youden"]*100,
               metrics_aug["recall_youden"]*100]
auc_lo_err  = [metrics_orig["roc_auc"] - metrics_orig["roc_auc_lo"],
               metrics_jitter["roc_auc"] - metrics_jitter["roc_auc_lo"],
               metrics_aug["roc_auc"] - metrics_aug["roc_auc_lo"]]
auc_hi_err  = [metrics_orig["roc_auc_hi"] - metrics_orig["roc_auc"],
               metrics_jitter["roc_auc_hi"] - metrics_jitter["roc_auc"],
               metrics_aug["roc_auc_hi"] - metrics_aug["roc_auc"]]

x = np.arange(len(conditions))
w = 0.30

# ROC-AUC bars
bars_auc = ax.bar(x - w/2, auc_vals, w,
                  color=colors_bar, label="ROC-AUC",
                  yerr=[auc_lo_err, auc_hi_err],
                  capsize=4, error_kw={"elinewidth": 1.5}, alpha=0.85)
# Recall bars (hatched)
bars_rec = ax.bar(x + w/2, [v/100 for v in recall_vals], w,
                  color=colors_bar, hatch="///", label="Recall@Youden",
                  alpha=0.55)

# Annotate
for bar, val in zip(bars_auc, auc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
for bar, val in zip(bars_rec, recall_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("(b) ROC-AUC and Recall@Youden")
ax.set_ylim(0, 1.25)

solid_patch = mpatches.Patch(facecolor="grey", alpha=0.8, label="ROC-AUC (solid)")
hatch_patch = mpatches.Patch(facecolor="grey", hatch="///", alpha=0.5, label="Recall@Youden (hatched)")
ax.legend(handles=[solid_patch, hatch_patch], fontsize=8, loc="upper right")

# ── (c) Jackknife AUC boxplot ────────────────────────────────────────────────
ax = axes[1, 0]

jk_orig_vals = [v for v in results["original"]["jackknife_auc"] if not np.isnan(v)]
jk_jit_vals  = [v for v in jk_jitter if not np.isnan(v)]
jk_aug_vals  = [v for v in results["augmented"]["jackknife_auc"] if not np.isnan(v)]

jk_data = pd.DataFrame({
    "AUC": jk_orig_vals + jk_jit_vals + jk_aug_vals,
    "Condition": (["Original"] * len(jk_orig_vals) +
                  ["Jitter"]   * len(jk_jit_vals)  +
                  ["Temporal"] * len(jk_aug_vals))
})

palette_jk = {"Original": BLUE, "Jitter": GREEN, "Temporal": "#cccccc"}
order_jk   = ["Original", "Jitter", "Temporal"]
sns.boxplot(data=jk_data, x="Condition", y="AUC", palette=palette_jk,
            order=order_jk, width=0.45, linewidth=1.5, fliersize=5, ax=ax)

for i, (vals, color) in enumerate([(jk_orig_vals, BLUE),
                                    (jk_jit_vals,  GREEN),
                                    (jk_aug_vals,  ORANGE)]):
    m = np.nanmean(vals)
    ax.hlines(m, i - 0.22, i + 0.22, colors=color, linestyles="--", lw=1.8)
    ax.text(i + 0.26, m, f"mu={m:.3f}", va="center", fontsize=8, color=color)

ax.set_xlabel("")
ax.set_ylabel("Leave-One-Lake-Out AUC")
ax.set_title("(c) Jackknife AUC Stability\n(17 GLOF source lakes)")
ax.set_ylim(-0.05, 1.1)
ax.text(2, 0.05, "(grey = temporal,\nfor context)",
        ha="center", va="bottom", fontsize=7, color="#888888", style="italic")

# ── (d) Numeric comparison table ────────────────────────────────────────────
ax = axes[1, 1]
ax.axis("off")
ax.set_facecolor("white")

prev_orig = 17 / 12700

table_data = [
    ["Metric",            "Original",           "Jitter",              "Temporal"],
    ["n+",                "17",                  "17 (+170/fold)",      "538"],
    ["ROC-AUC",
     f"{metrics_orig['roc_auc']:.3f}",
     f"{metrics_jitter['roc_auc']:.3f}",
     f"{metrics_aug['roc_auc']:.3f}"],
    ["95% CI",
     f"[{metrics_orig['roc_auc_lo']:.3f}-{metrics_orig['roc_auc_hi']:.3f}]",
     f"[{metrics_jitter['roc_auc_lo']:.3f}-{metrics_jitter['roc_auc_hi']:.3f}]",
     f"[{metrics_aug['roc_auc_lo']:.3f}-{metrics_aug['roc_auc_hi']:.3f}]"],
    ["Lift",
     f"{metrics_orig['lift']:.2f}x",
     f"{metrics_jitter['lift']:.2f}x",
     f"{metrics_aug['lift']:.2f}x"],
    ["Recall@Youden",
     f"{metrics_orig['recall_youden']*100:.1f}%",
     f"{metrics_jitter['recall_youden']*100:.1f}%",
     f"{metrics_aug['recall_youden']*100:.1f}%"],
    ["JK mean+/-std",
     f"{np.nanmean(jk_orig_vals):.3f}+/-{np.nanstd(jk_orig_vals):.3f}",
     f"{np.nanmean(jk_jit_vals):.3f}+/-{np.nanstd(jk_jit_vals):.3f}",
     f"{np.nanmean(jk_aug_vals):.3f}+/-{np.nanstd(jk_aug_vals):.3f}"],
]

tbl = ax.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.05, 1.0, 0.90]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

# Style header row
for j in range(4):
    cell = tbl[0, j]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")

# Color-code condition columns
col_colors = {1: BLUE + "33", 2: GREEN + "33", 3: ORANGE + "33"}
for i in range(1, len(table_data)):
    for j, c in col_colors.items():
        tbl[i, j].set_facecolor(c)
    tbl[i, 0].set_facecolor("#f5f5f5")
    tbl[i, 0].set_text_props(fontweight="bold")

ax.set_title("(d) Numerical Comparison", fontsize=10, pad=8)

fig.tight_layout(rect=[0, 0, 1, 0.998])
fig.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved -> {OUT_FIG}")

# ── 11. Summary ──────────────────────────────────────────────────────────────
jk_o_mean = np.nanmean(jk_orig_vals);  jk_o_std = np.nanstd(jk_orig_vals)
jk_j_mean = np.nanmean(jk_jit_vals);   jk_j_std = np.nanstd(jk_jit_vals)
jk_a_mean = np.nanmean(jk_aug_vals);   jk_a_std = np.nanstd(jk_aug_vals)

print("\n" + "=" * 68)
print("AUGMENTATION COMPARISON SUMMARY  (3 conditions)")
print("=" * 68)
print(f"  ROC-AUC Original  : {metrics_orig['roc_auc']:.4f}  "
      f"95%CI [{metrics_orig['roc_auc_lo']:.4f}-{metrics_orig['roc_auc_hi']:.4f}]")
print(f"  ROC-AUC Jitter    : {metrics_jitter['roc_auc']:.4f}  "
      f"95%CI [{metrics_jitter['roc_auc_lo']:.4f}-{metrics_jitter['roc_auc_hi']:.4f}]")
print(f"  ROC-AUC Temporal  : {metrics_aug['roc_auc']:.4f}  "
      f"95%CI [{metrics_aug['roc_auc_lo']:.4f}-{metrics_aug['roc_auc_hi']:.4f}]")
print()
print(f"  PR-AUC  Original  : {metrics_orig['pr_auc']:.4f}  Lift={metrics_orig['lift']:.2f}x")
print(f"  PR-AUC  Jitter    : {metrics_jitter['pr_auc']:.4f}  Lift={metrics_jitter['lift']:.2f}x")
print(f"  PR-AUC  Temporal  : {metrics_aug['pr_auc']:.4f}  Lift={metrics_aug['lift']:.2f}x")
print()
print(f"  Recall@Youden Orig: {metrics_orig['recall_youden']*100:.1f}%")
print(f"  Recall@Youden Jit : {metrics_jitter['recall_youden']*100:.1f}%")
print(f"  Recall@Youden Temp: {metrics_aug['recall_youden']*100:.1f}%")
print()
print(f"  JK-AUC mean Orig  : {jk_o_mean:.4f} +/- {jk_o_std:.4f}")
print(f"  JK-AUC mean Jitter: {jk_j_mean:.4f} +/- {jk_j_std:.4f}")
print(f"  JK-AUC mean Temp  : {jk_a_mean:.4f} +/- {jk_a_std:.4f}")
print("=" * 68)
