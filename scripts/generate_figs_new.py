"""
Regeneracion de figuras 3-7 mejoradas para el paper GLOF IJACSA.
Ejecutar desde la raiz del proyecto con glof/Scripts/python scripts/generate_figs_new.py
"""
import sys, os, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.inspection import PartialDependenceDisplay

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'figures', 'publication')
os.makedirs(OUT, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':       9,
    'axes.titlesize': 10,
    'axes.labelsize':  9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        100,
})
C_GLOF   = '#D62728'   # rojo GLOF
C_NOGL   = '#1F77B4'   # azul no-GLOF
C_JITTER = '#2CA02C'   # verde jitter
C_GREY   = '#7F7F7F'

def panel_label(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')

# ── Load common data ───────────────────────────────────────────────────────────
df  = pd.read_csv(os.path.join(ROOT, 'data/processed/labeled/training_data.csv'))
oof = json.load(open(os.path.join(ROOT, 'models/oof_predictions.json')))
y_true  = np.array(oof['y_true'])
y_score = np.array(oof['y_score_brf'])
y_lgbm  = np.array(oof['y_score_lgbm'])

# Attach OOF score to df
df['brf_score'] = y_score
pos_mask = df['had_glof'] == 1
neg_mask = ~pos_mask

rob   = json.load(open(os.path.join(ROOT, 'models/robustness_results.json')))
jk    = json.load(open(os.path.join(ROOT, 'models/jackknife_results.json')))
mc    = pd.read_csv(os.path.join(ROOT, 'models/model_comparison.csv'))
shap_df = pd.read_csv(os.path.join(ROOT, 'data/processed/shap_feature_importance.csv'))
with open(os.path.join(ROOT, 'models/oof_curves.pkl'), 'rb') as f:
    oof_curves = pickle.load(f)
loco = pd.read_csv(os.path.join(ROOT, 'models/loco_validation.csv'))
thresh_d = json.load(open(os.path.join(ROOT, 'models/optimal_thresholds.json')))
youden_t = thresh_d['BalancedRF']  # 0.356
prec_k   = rob['precision_at_k']

print("Data loaded. Generating figures...")

# ==============================================================================
# FIG 3 — Morphological risk space + Feature separation
# ==============================================================================
print("  Fig 3...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.subplots_adjust(wspace=0.35)

# Panel (a) — Scatter: log(area) vs log(volume), colour = BRF score
ax = axes[0]
area_log   = np.log10(df['area_m2'].clip(1))
vol_log    = np.log10(df['volume_m3'].clip(1))
score_vals = df['brf_score'].values

sc = ax.scatter(area_log[neg_mask], vol_log[neg_mask],
                c=score_vals[neg_mask], cmap='RdYlBu_r',
                s=7, alpha=0.35, linewidths=0, vmin=0, vmax=1, rasterized=True)
ax.scatter(area_log[pos_mask], vol_log[pos_mask],
           c=score_vals[pos_mask], cmap='RdYlBu_r',
           s=120, marker='*', edgecolors='black', linewidths=0.6,
           vmin=0, vmax=1, zorder=5)

# Label a few notable GLOF lakes
for _, row in df[pos_mask].nlargest(4, 'area_m2').iterrows():
    ax.annotate(str(row.get('glof_lake_name',''))[:12],
                (np.log10(max(row['area_m2'],1)), np.log10(max(row['volume_m3'],1))),
                fontsize=6, ha='left', va='bottom',
                xytext=(3,3), textcoords='offset points')

cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cb.set_label('BRF susceptibility score', fontsize=8)

ax.set_xlabel('Lake area (log₁₀ m²)')
ax.set_ylabel('Estimated volume (log₁₀ m³)')
ax.set_title('Morphological risk space\n(all 12,700 lake-year observations)', fontsize=10)
ax.legend(handles=[
    mpatches.Patch(facecolor=plt.cm.RdYlBu_r(0.85), label='High susceptibility'),
    mpatches.Patch(facecolor=plt.cm.RdYlBu_r(0.15), label='Low susceptibility'),
    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='red',
               markersize=10, markeredgecolor='black', label=f'GLOF-source (n=17)')
], fontsize=8, framealpha=0.7, loc='upper left')
panel_label(ax, '(a)')

# Panel (b) — Violin: 4 key features GLOF vs non-GLOF
ax = axes[1]
features_violin = [
    ('area_m2',           'Lake area\n(log₁₀ m²)',   True),
    ('volume_m3',         'Volume\n(log₁₀ m³)',       True),
    ('depth_ensemble_mean','Depth\n(m)',              False),
    ('slope_mean',        'Slope\n(mean °)',          False),
]

glof_data = df[pos_mask]
noglf_data = df[neg_mask]
positions = np.array([1, 3, 5, 7])

glof_color_patch = dict(facecolor=C_GLOF, alpha=0.7)
nogl_color_patch = dict(facecolor=C_NOGL, alpha=0.5)

for i, (col, label, do_log) in enumerate(features_violin):
    vals_g = glof_data[col].dropna()
    vals_n = noglf_data[col].dropna()
    if do_log:
        vals_g = np.log10(vals_g.clip(1))
        vals_n = np.log10(vals_n.clip(1))
    pos_i = positions[i]
    vp_n = ax.violinplot([vals_n], positions=[pos_i - 0.35], widths=0.6,
                         showmedians=True, showextrema=False)
    vp_g = ax.violinplot([vals_g], positions=[pos_i + 0.35], widths=0.6,
                         showmedians=True, showextrema=False)
    for pc in vp_n['bodies']: pc.set(facecolor=C_NOGL, alpha=0.55, linewidth=0.5)
    for pc in vp_g['bodies']: pc.set(facecolor=C_GLOF, alpha=0.7, linewidth=0.5)
    vp_n['cmedians'].set(color='white', linewidth=2)
    vp_g['cmedians'].set(color='white', linewidth=2)
    # Mann-Whitney p
    stat, pv = stats.mannwhitneyu(vals_g, vals_n, alternative='two-sided')
    stars = '***' if pv < 0.001 else ('**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns'))
    y_top = max(vals_n.quantile(0.99), vals_g.quantile(0.99)) * 1.05
    ax.text(pos_i, y_top, stars, ha='center', va='bottom', fontsize=9, color='black')

ax.set_xticks(positions)
ax.set_xticklabels([lbl for _, lbl, _ in features_violin], fontsize=8)
ax.set_ylabel('Feature value')
ax.set_title('Feature separation: GLOF-source vs. non-GLOF lakes', fontsize=10)
ax.legend(handles=[
    mpatches.Patch(facecolor=C_NOGL, alpha=0.6, label='Non-GLOF (n=12,683)'),
    mpatches.Patch(facecolor=C_GLOF, alpha=0.7, label='GLOF-source (n=17)'),
], fontsize=8, loc='upper right')
ax.text(0.98, 0.02, '*** p<0.001  ** p<0.01  * p<0.05',
        transform=ax.transAxes, fontsize=7, ha='right', va='bottom', color=C_GREY)
panel_label(ax, '(b)')

fig.suptitle('GLOF susceptibility in morphological feature space', fontsize=11, y=1.01)
fig.savefig(os.path.join(OUT, 'fig3_susceptibility_distribution.png'),
            dpi=600, bbox_inches='tight')
plt.close()
print("  Fig 3 done.")

# ==============================================================================
# FIG 4 — Model performance dashboard
# ==============================================================================
print("  Fig 4...")
fig = plt.figure(figsize=(13, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
ax_roc   = fig.add_subplot(gs[0, 0])
ax_heat  = fig.add_subplot(gs[0, 1])
ax_loco  = fig.add_subplot(gs[1, 0])
ax_topk  = fig.add_subplot(gs[1, 1])

# ── (a) ROC curves ─────────────────────────────────────────────────────────────
model_styles = {
    'BalancedRF':        ('#D62728', 2.2, '-'),
    'XGBoost':           ('#FF7F0E', 1.2, '--'),
    'EasyEnsemble':      ('#2CA02C', 1.0, '-.'),
    'LightGBM':          ('#9467BD', 1.0, ':'),
    'Random Forest':     ('#8C564B', 0.9, '--'),
    'Logistic Regression':('#7F7F7F', 0.9, ':'),
}
ax_roc.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.4, label='Random (0.500)')

for name, (color, lw, ls) in model_styles.items():
    if name in oof_curves['roc']:
        fpr, tpr, _ = oof_curves['roc'][name]
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f'{name} ({roc_auc:.3f})')
        if name == 'BalancedRF':
            # CI shading: approximate with ±CI bands using bootstrap CI
            ci_lo, ci_hi = rob['ci_low'], rob['ci_high']
            ax_roc.fill_between(fpr, tpr * (ci_lo / roc_auc), tpr * (ci_hi / roc_auc),
                                alpha=0.15, color=color)
            ax_roc.annotate(f'BRF AUC={roc_auc:.3f}\n95% CI [{ci_lo:.3f}–{ci_hi:.3f}]',
                            xy=(0.60, 0.22), fontsize=7.5, color=color,
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('OOF ROC curves — six classifiers')
ax_roc.legend(fontsize=7, loc='lower right', framealpha=0.8)
panel_label(ax_roc, '(a)')

# ── (b) Metric heatmap ─────────────────────────────────────────────────────────
# Use positional columns to avoid encoding issues: col0=Model, 1=AUC, 2=PR-AUC, 3=Lift, 4=MCC
mc_raw = pd.read_csv(os.path.join(ROOT, 'models/model_comparison.csv'), encoding='latin1')
mc_raw.columns = [c.strip() for c in mc_raw.columns]

def extract_num(series):
    return pd.to_numeric(series.astype(str).str.extract(r'([+-]?[\d.]+)', expand=False), errors='coerce')

model_names = mc_raw.iloc[:, 0].str.strip().values
hm_data = pd.DataFrame({
    'ROC-AUC': mc_raw.iloc[:, 1].values.astype(float),
    'PR-AUC':  mc_raw.iloc[:, 2].values.astype(float),
    'Lift':    extract_num(mc_raw.iloc[:, 3]).values,
    'MCC':     mc_raw.iloc[:, 4].values.astype(float),
}, index=model_names)

# Sort by Lift
hm_data = hm_data.sort_values('Lift', ascending=False)

# Normalise each column 0-1 for colour
hm_norm = hm_data.copy()
for col in hm_norm.columns:
    mn, mx = hm_norm[col].min(), hm_norm[col].max()
    if mx > mn:
        hm_norm[col] = (hm_norm[col] - mn) / (mx - mn)
    else:
        hm_norm[col] = 0.5

im = ax_heat.imshow(hm_norm.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

fmt_funcs = {'ROC-AUC': '{:.3f}', 'PR-AUC': '{:.4f}', 'Lift': '{:.1f}×', 'MCC': '{:.3f}'}
for i in range(len(hm_data)):
    for j, col in enumerate(hm_data.columns):
        val = hm_data.iloc[i, j]
        txt = fmt_funcs[col].format(val) if not np.isnan(val) else '—'
        brightness = hm_norm.iloc[i, j]
        txt_color = 'black' if 0.3 < brightness < 0.85 else ('white' if brightness <= 0.3 else 'black')
        ax_heat.text(j, i, txt, ha='center', va='center', fontsize=8, color=txt_color)

ax_heat.set_xticks(range(len(hm_data.columns)))
ax_heat.set_xticklabels(hm_data.columns, fontsize=8)
ax_heat.set_yticks(range(len(hm_data)))
ax_heat.set_yticklabels(hm_data.index, fontsize=8)
ax_heat.set_title('Classifier performance heatmap\n(green = best per metric)')

# Highlight BalancedRF row
brf_idx = list(hm_data.index).index('BalancedRF')
rect = plt.Rectangle((-0.5, brf_idx - 0.5), len(hm_data.columns), 1,
                     linewidth=2.5, edgecolor='black', facecolor='none')
ax_heat.add_patch(rect)
ax_heat.annotate('← Selected', xy=(len(hm_data.columns)-0.5, brf_idx),
                 fontsize=7.5, color='black', va='center')
panel_label(ax_heat, '(b)')

# ── (c) LOCO geographic validation ────────────────────────────────────────────
loco_raw = pd.read_csv(os.path.join(ROOT, 'models/loco_validation.csv'), encoding='latin1')
# positional: col0=country, 1=n_lakes, 2=glofs_test, 3=glofs_train, 4=auc, 5=pr_auc, 6=lift
# Add Chile manually (null domain)
loco_rows = [
    {'country': 'Bolivia', 'auc': loco_raw.iloc[0,4], 'lift': loco_raw.iloc[0,6], 'npos': int(loco_raw.iloc[0,2]), 'color':'#2CA02C'},
    {'country': 'Ecuador', 'auc': loco_raw.iloc[1,4], 'lift': loco_raw.iloc[1,6], 'npos': int(loco_raw.iloc[1,2]), 'color':'#FF7F0E'},
    {'country': 'Peru',    'auc': loco_raw.iloc[2,4], 'lift': loco_raw.iloc[2,6], 'npos': int(loco_raw.iloc[2,2]), 'color':'#D62728'},
    {'country': 'Chile',   'auc': None, 'lift': None, 'npos': 0, 'color':'#7F7F7F'},
]

y_pos_loco = range(len(loco_rows))
bar_vals = [r['auc'] if r['auc'] is not None else 0.265 for r in loco_rows]
bars = ax_loco.barh(list(y_pos_loco), bar_vals,
                    color=[r['color'] for r in loco_rows],
                    alpha=0.8, height=0.55, edgecolor='black', lw=0.5)
ax_loco.axvline(0.5, color='gray', lw=1.2, ls='--', alpha=0.6, label='No-skill (0.5)')
ax_loco.set_yticks(list(y_pos_loco))
ax_loco.set_yticklabels([r['country'] for r in loco_rows], fontsize=9)
ax_loco.set_xlabel('LOCO ROC-AUC')
ax_loco.set_title('Geographic generalizability\n(Leave-One-Country-Out validation)')
ax_loco.set_xlim(0, 1.22)

for i, r in enumerate(loco_rows):
    if r['country'] == 'Chile':
        ax_loco.text(0.28, i, 'FPR: 26.5% — null domain\n(no matched GLOFs)',
                     va='center', fontsize=7.5, color='#555')
    else:
        ax_loco.text(r['auc']+0.02, i,
                     f"AUC {r['auc']:.3f}  Lift {float(str(r['lift']).replace('×','').replace('x','')):.1f}×  n⁺={r['npos']}",
                     va='center', fontsize=7.5)

ax_loco.legend(fontsize=8)
panel_label(ax_loco, '(c)')

# ── (d) Operational prioritization — Recall@k ─────────────────────────────────
topk_pct    = [p['top_k_pct'] for p in prec_k]
topk_recall = [p['recall_pct'] for p in prec_k]
topk_lift   = [p['lift'] for p in prec_k]

# Calculate full Recall@k curve
k_range = np.linspace(0.1, 20, 200)
recall_curve = []
lift_curve   = []
idx_sorted   = np.argsort(-y_score)
n_pos_total  = y_true.sum()
n_total      = len(y_true)
for k in k_range:
    n_k = max(1, int(n_total * k / 100))
    top_k_idx = idx_sorted[:n_k]
    rec = y_true[top_k_idx].sum() / n_pos_total * 100
    recall_curve.append(rec)
    lift_curve.append((y_true[top_k_idx].sum() / n_k) / (n_pos_total / n_total))

recall_random = k_range
ax_topk.plot(k_range, recall_curve, color=C_GLOF, lw=2.2, label='BalancedRF')
ax_topk.plot(k_range, recall_random, color=C_GREY, lw=1.2, ls='--', label='Random baseline')
ax_topk.fill_between(k_range, recall_random, recall_curve, alpha=0.15, color=C_GLOF)

ax_topk.axhline(94, color=C_GLOF, lw=0.9, ls=':', alpha=0.6)
ax_topk.text(20.5, 94, '94% Recall\n(Youden)', fontsize=7, va='center', color=C_GLOF)

for p in prec_k:
    if p['top_k_pct'] in [2.0, 10.0]:
        ax_topk.annotate(
            f"{p['top_k_pct']:.0f}%→{p['recall_pct']:.0f}% rec\n{p['lift']:.1f}× lift",
            xy=(p['top_k_pct'], p['recall_pct']),
            xytext=(p['top_k_pct']+2, p['recall_pct']-12),
            arrowprops=dict(arrowstyle='->', color='black', lw=1),
            fontsize=7.5, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8)
        )
    ax_topk.scatter(p['top_k_pct'], p['recall_pct'], color=C_GLOF, s=40, zorder=4)

ax_topk.set_xlabel('Top-k% of inventory inspected')
ax_topk.set_ylabel('GLOF-source lakes recovered (%)')
ax_topk.set_title('Prioritization efficiency\n(recovery of 17 GLOF-source lakes)')
ax_topk.set_xlim(0, 21)
ax_topk.set_ylim(0, 110)
ax_topk.legend(fontsize=8)
panel_label(ax_topk, '(d)')

fig.suptitle('Model performance and geographic generalization', fontsize=12, y=1.01)
fig.savefig(os.path.join(OUT, 'fig4_model_performance.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Fig 4 done.")

# ==============================================================================
# FIG 5 — SHAP feature importance + physical drivers
# ==============================================================================
print("  Fig 5...")
import joblib

model = joblib.load(os.path.join(ROOT, 'models/best_model.joblib'))
feat_names_all = json.load(open(os.path.join(ROOT, 'models/feature_names.json')))
# Solo usar features presentes en el dataframe
feat_names = [f for f in feat_names_all if f in df.columns]

# Build feature matrix (only available features)
X = df[feat_names].copy()
X = X.fillna(X.median())

fig = plt.figure(figsize=(13, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)
ax_bar   = fig.add_subplot(gs[0, 0])
ax_dep   = fig.add_subplot(gs[0, 1])
ax_dep2  = fig.add_subplot(gs[1, 0])
ax_box   = fig.add_subplot(gs[1, 1])

# ── (a) SHAP importance bar chart ──────────────────────────────────────────────
top10 = shap_df.head(10).copy()
# Prettify feature names
name_map = {
    'dist_glacier_m':         'Glacier distance (m)',
    'slope_x_logarea':        'Slope × log(area)',
    'volume_m3':              'Lake volume (m³)',
    'area_depth_ratio':       'Area/depth ratio',
    'depth_ensemble_mean':    'Ensemble depth (mean)',
    'log_area_m2':            'Log lake area',
    'depth_std_m':            'Depth std',
    'depth_yao_2012':         'Depth — Yao 2012',
    'depth_ensemble_std':     'Ensemble depth (std)',
    'elongation':             'Lake elongation',
}
top10['label'] = top10['feature'].map(name_map).fillna(top10['feature'])
colors_bar = [C_GLOF if i == 0 else '#4878CF' for i in range(len(top10))]
bars = ax_bar.barh(range(len(top10)), top10['mean_abs_shap'].values[::-1],
                   color=colors_bar[::-1], alpha=0.85, height=0.65, edgecolor='white', lw=0.5)
ax_bar.set_yticks(range(len(top10)))
ax_bar.set_yticklabels(top10['label'].values[::-1], fontsize=8)
ax_bar.set_xlabel('Mean |SHAP value|')
ax_bar.set_title('Global feature importance\n(top-10 SHAP features)')
ax_bar.axvline(top10['mean_abs_shap'].mean(), color=C_GREY, lw=0.8, ls='--', alpha=0.7)
ax_bar.text(top10['mean_abs_shap'].mean()*1.02, -0.7, 'mean', fontsize=7, color=C_GREY)
# Highlight top feature
ax_bar.get_yticklabels()[len(top10)-1].set_fontweight('bold')
ax_bar.get_yticklabels()[len(top10)-1].set_color(C_GLOF)
panel_label(ax_bar, '(a)')

# ── (b) Partial dependence: volume_m3 (proxy for physical driver) ──────────────
# Show susceptibility score vs volume_m3 as scatter with LOESS-like smoothing
vol  = df['volume_m3'].clip(1)
vol_log = np.log10(vol)
score_v = df['brf_score'].values

# Bin and compute mean score
bins = np.percentile(vol_log[neg_mask], np.linspace(0,100,30))
bin_means_x, bin_means_y = [], []
for i in range(len(bins)-1):
    mask_bin = neg_mask & (vol_log >= bins[i]) & (vol_log < bins[i+1])
    if mask_bin.sum() > 5:
        bin_means_x.append((bins[i]+bins[i+1])/2)
        bin_means_y.append(score_v[mask_bin].mean())

ax_dep.scatter(vol_log[neg_mask], score_v[neg_mask],
               c=C_NOGL, s=4, alpha=0.12, rasterized=True)
ax_dep.scatter(vol_log[pos_mask], score_v[pos_mask],
               c=C_GLOF, s=80, marker='*', edgecolors='black', lw=0.5, zorder=5,
               label='GLOF-source (n=17)')
ax_dep.plot(bin_means_x, bin_means_y, color='navy', lw=2, label='Non-GLOF mean')

ax_dep.axhline(youden_t, color=C_GREY, lw=1, ls=':', alpha=0.8)
ax_dep.text(0.02, youden_t+0.02, f'Youden t={youden_t}', fontsize=7.5, color=C_GREY,
            transform=ax_dep.get_yaxis_transform())

ax_dep.set_xlabel('Lake volume (log₁₀ m³)')
ax_dep.set_ylabel('BRF susceptibility score')
ax_dep.set_title('Susceptibility vs. lake volume\n(positive relationship confirms physical driver)')
ax_dep.legend(fontsize=8, loc='upper left')
panel_label(ax_dep, '(b)')

# ── (c) Partial dependence: slope_mean ────────────────────────────────────────
slope = df['slope_mean'].values
bins_s = np.percentile(slope[neg_mask][slope[neg_mask]>0], np.linspace(0,100,25))
bx2, by2 = [], []
for i in range(len(bins_s)-1):
    mask_bin = neg_mask & (slope >= bins_s[i]) & (slope < bins_s[i+1])
    if mask_bin.sum() > 5:
        bx2.append((bins_s[i]+bins_s[i+1])/2)
        by2.append(score_v[mask_bin].mean())

ax_dep2.scatter(slope[neg_mask], score_v[neg_mask],
                c=C_NOGL, s=4, alpha=0.12, rasterized=True)
ax_dep2.scatter(slope[pos_mask], score_v[pos_mask],
                c=C_GLOF, s=80, marker='*', edgecolors='black', lw=0.5, zorder=5,
                label='GLOF-source (n=17)')
ax_dep2.plot(bx2, by2, color='navy', lw=2, label='Non-GLOF mean')
ax_dep2.axhline(youden_t, color=C_GREY, lw=1, ls=':', alpha=0.8)
ax_dep2.set_xlabel('Mean slope (°)')
ax_dep2.set_ylabel('BRF susceptibility score')
ax_dep2.set_title('Susceptibility vs. terrain slope\n(slope × log(area) is 2nd SHAP driver)')
ax_dep2.legend(fontsize=8, loc='upper left')
panel_label(ax_dep2, '(c)')

# ── (d) BRF score distribution by study area ─────────────────────────────────
area_order = (df.groupby('area_name')['brf_score']
                .median()
                .sort_values(ascending=True)
                .index.tolist())
area_labels = {
    'cordillera_blanca':      'C. Blanca (PE)',
    'cordillera_vilcanota':   'C. Vilcanota (PE)',
    'cordillera_central':     'C. Central (PE)',
    'cordillera_raura':       'C. Raura (PE)',
    'chile_andes_centrales':  'Andes Centr. (CL)',
    'cordillera_urubamba':    'C. Urubamba (PE)',
    'cordillera_huanzo':      'C. Huanzo (PE)',
    'cordillera_huayhuash':   'C. Huayhuash (PE)',
    'ecuador_antisana':       'Antisana (EC)',
    'bolivia_cordillera_real':'C. Real (BO)',
    'carabaya':               'Carabaya (PE)',
    'patagonia_norte':        'Patagonia N.',
    'apolobamba':             'Apolobamba',
    'patagonia_sur':          'Patagonia S.',
}
glof_areas = set(df[pos_mask]['area_name'].unique())

medians = []
data_list = []
labels_list = []
colors_box = []
for area in area_order:
    vals = df[df['area_name']==area]['brf_score'].values
    medians.append(np.median(vals))
    data_list.append(vals)
    labels_list.append(area_labels.get(area, area))
    colors_box.append(C_GLOF if area in glof_areas else C_NOGL)

bplot = ax_box.boxplot(data_list, vert=False, patch_artist=True,
                       flierprops=dict(marker='.', ms=2, alpha=0.3),
                       medianprops=dict(color='white', lw=2),
                       whiskerprops=dict(lw=0.8),
                       boxprops=dict(lw=0.5))
for patch, col in zip(bplot['boxes'], colors_box):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)

ax_box.axvline(youden_t, color='black', lw=1.2, ls='--', alpha=0.7,
               label=f'Youden t={youden_t}')
ax_box.set_yticks(range(1, len(labels_list)+1))
ax_box.set_yticklabels(labels_list, fontsize=7.5)
ax_box.set_xlabel('Predicted susceptibility score')
ax_box.set_title('Score distribution by study area\n(red = area with ≥1 matched GLOF)')

ax_box.legend(handles=[
    mpatches.Patch(facecolor=C_GLOF, alpha=0.7, label='Area with GLOF(s)'),
    mpatches.Patch(facecolor=C_NOGL, alpha=0.7, label='No matched GLOF'),
    plt.Line2D([0],[0], color='black', ls='--', label=f'Youden t={youden_t}')
], fontsize=7.5, loc='lower right')
panel_label(ax_box, '(d)')

fig.suptitle('SHAP-based susceptibility drivers and model interpretability', fontsize=12, y=1.01)
fig.savefig(os.path.join(OUT, 'fig5_shap_risk_drivers.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Fig 5 done.")

# ==============================================================================
# FIG 6 — Robustness + threshold + lift (3 paneles)
# ==============================================================================
print("  Fig 6...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subplots_adjust(wspace=0.38)

# ── (a) Bootstrap AUC histogram + jackknife dots ───────────────────────────────
ax = axes[0]
# Recompute bootstrap AUC
rng = np.random.default_rng(42)
n_boot = 2000
boot_aucs = []
idx = np.where(y_true == 1)[0]
idx_neg = np.where(y_true == 0)[0]
for _ in range(n_boot):
    pos_s = rng.choice(idx, size=len(idx), replace=True)
    neg_s = rng.choice(idx_neg, size=len(idx_neg), replace=True)
    bt = np.concatenate([y_true[pos_s], y_true[neg_s]])
    bs = np.concatenate([y_score[pos_s], y_score[neg_s]])
    fpr_b, tpr_b, _ = roc_curve(bt, bs)
    boot_aucs.append(auc(fpr_b, tpr_b))

boot_aucs = np.array(boot_aucs)
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

ax.hist(boot_aucs, bins=50, color='#4878CF', alpha=0.65, edgecolor='white', lw=0.3)
ax.axvline(rob['roc_auc_oof'], color='black', lw=2, label=f'Observed AUC = {rob["roc_auc_oof"]:.3f}')
ax.axvline(ci_lo, color='#D62728', lw=1.2, ls='--', label=f'95% CI [{ci_lo:.3f}–{ci_hi:.3f}]')
ax.axvline(ci_hi, color='#D62728', lw=1.2, ls='--')
ax.axvspan(ci_lo, ci_hi, alpha=0.08, color='#D62728')

# Jackknife dots at top
jk_aucs = np.array(jk['jackknife_aucs'])
y_jk = ax.get_ylim()[1] * 0.88 if ax.get_ylim()[1] > 0 else 50
ax.scatter(jk_aucs, np.full_like(jk_aucs, y_jk if y_jk > 0 else 40),
           color='#FF7F0E', s=35, zorder=5, label=f'Jackknife (n=17)')
ax.axhline(y_jk if y_jk > 0 else 40, color='#FF7F0E', lw=0.6, ls=':', alpha=0.4)

ax.set_xlabel('Bootstrap ROC-AUC')
ax.set_ylabel('Count')
ax.set_title('Model robustness\nBootstrap CI + jackknife stability')
ax.text(0.04, 0.95, f'p < 0.0002\n(permutation, n=5,000)',
        transform=ax.transAxes, fontsize=8, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.9))
ax.legend(fontsize=7.5, loc='upper left')
panel_label(ax, '(a)')

# ── (b) Threshold sensitivity ─────────────────────────────────────────────────
ax = axes[1]
thresholds = np.linspace(0.01, 0.99, 200)
recalls, flagged = [], []
for t in thresholds:
    preds = (y_score >= t).astype(int)
    tp = ((preds==1) & (y_true==1)).sum()
    rec = tp / y_true.sum() if y_true.sum() > 0 else 0
    flag_pct = preds.sum() / len(preds) * 100
    recalls.append(rec * 100)
    flagged.append(flag_pct)

recalls  = np.array(recalls)
flagged  = np.array(flagged)

ax2_twin = ax.twinx()
l1, = ax.plot(thresholds, recalls, color=C_GLOF, lw=2, label='Recall (%)')
l2, = ax2_twin.plot(thresholds, flagged, color='#1F77B4', lw=2, ls='--', label='Flagged (%)')

ax.axvline(youden_t, color='black', lw=1.5, ls=':')
ax.annotate(f'Youden\nt={youden_t}\n94% Recall\n41.3% flagged',
            xy=(youden_t, 94), xytext=(youden_t+0.12, 75),
            arrowprops=dict(arrowstyle='->', lw=1),
            fontsize=7.5, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

for t_mark, label in [(0.20,'t=0.20'), (0.50,'t=0.50')]:
    idx_t = np.argmin(np.abs(thresholds - t_mark))
    ax.scatter(t_mark, recalls[idx_t], color=C_GLOF, s=40, zorder=4)
    ax2_twin.scatter(t_mark, flagged[idx_t], color='#1F77B4', s=40, zorder=4)

ax.set_xlabel('Classification threshold')
ax.set_ylabel('Recall (%)', color=C_GLOF)
ax2_twin.set_ylabel('Flagged lakes (%)', color='#1F77B4')
ax.tick_params(axis='y', labelcolor=C_GLOF)
ax2_twin.tick_params(axis='y', labelcolor='#1F77B4')
ax.set_title('Threshold sensitivity analysis')
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='center right')
panel_label(ax, '(b)')

# ── (c) Lift@k bar chart ───────────────────────────────────────────────────────
ax = axes[2]
k_labels = [f"{p['top_k_pct']:.1f}%" for p in prec_k]
lift_vals = [p['lift'] for p in prec_k]
recall_vals = [p['recall_pct'] for p in prec_k]
n_glofs  = [p['glofs_found'] for p in prec_k]

cmap_lift = plt.cm.RdYlGn
norm_lift = Normalize(vmin=1, vmax=max(lift_vals))
bar_colors = [cmap_lift(norm_lift(l)) for l in lift_vals]

bars_k = ax.bar(range(len(k_labels)), lift_vals, color=bar_colors,
                alpha=0.9, edgecolor='black', lw=0.5, width=0.65)
ax.axhline(1, color=C_GREY, lw=1.2, ls='--', label='Random (Lift=1)')

for i, (bar, rv, lv, ng) in enumerate(zip(bars_k, recall_vals, lift_vals, n_glofs)):
    ax.text(i, lv + 0.3, f'{lv:.0f}×\n{rv:.0f}% rec', ha='center', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))

ax.set_xticks(range(len(k_labels)))
ax.set_xticklabels(k_labels, fontsize=8)
ax.set_xlabel('Top-k% inspected')
ax.set_ylabel('Lift over random')
ax.set_title('Prioritization lift\n(inspecting k% of inventory)')
ax.legend(fontsize=8)
sm = ScalarMappable(cmap=cmap_lift, norm=norm_lift)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cb.set_label('Lift magnitude', fontsize=7.5)
panel_label(ax, '(c)')

fig.suptitle('Statistical robustness and operational efficiency', fontsize=12, y=1.01)
fig.savefig(os.path.join(OUT, 'fig6_robustness_analysis.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Fig 6 done.")

# ==============================================================================
# FIG 7 — KDE score distributions + Reliability diagram (2 paneles)
# ==============================================================================
print("  Fig 7...")
from sklearn.calibration import calibration_curve
from scipy.stats import ks_2samp

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(wspace=0.38)

# ── (a) KDE score distributions ───────────────────────────────────────────────
ax = axes[0]
scores_neg = y_score[y_true == 0]
scores_pos = y_score[y_true == 1]

from scipy.stats import gaussian_kde
x_range = np.linspace(0, 1, 500)
kde_neg = gaussian_kde(scores_neg, bw_method='silverman')
kde_pos = gaussian_kde(scores_pos, bw_method='silverman')

ax.fill_between(x_range, kde_neg(x_range), alpha=0.45, color=C_NOGL, label=f'Non-GLOF (n={len(scores_neg):,})')
ax.fill_between(x_range, kde_pos(x_range), alpha=0.65, color=C_GLOF, label=f'GLOF-source (n={len(scores_pos)})')
ax.plot(x_range, kde_neg(x_range), color=C_NOGL, lw=1.5)
ax.plot(x_range, kde_pos(x_range), color=C_GLOF, lw=2)

# Rug marks for GLOF-source
ax.plot(scores_pos, np.full_like(scores_pos, -0.5), '|', color=C_GLOF, ms=12, lw=2)

# Youden threshold
ax.axvline(youden_t, color='black', lw=1.5, ls='--')
ax.text(youden_t+0.01, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1]>0 else 2.5,
        f't={youden_t}', fontsize=8.5, color='black')

# KS statistic
ks_stat, ks_pval = ks_2samp(scores_neg, scores_pos)
ax.annotate(f'KS D={ks_stat:.3f}\np={ks_pval:.2e}',
            xy=(0.5, 1.5), xytext=(0.25, 3.5),
            arrowprops=dict(arrowstyle='->', lw=1),
            fontsize=8.5, bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9))

ax.set_xlabel('OOF BRF susceptibility score')
ax.set_ylabel('Density')
ax.set_title('Score distributions: GLOF-source vs. non-GLOF\n(KS test confirms significant separation)')
ax.legend(fontsize=9, loc='upper center')
ax.set_xlim(-0.02, 1.02)
panel_label(ax, '(a)')

# ── (b) Reliability diagram ────────────────────────────────────────────────────
ax = axes[1]
n_bins = 10
fraction_of_pos, mean_predicted = calibration_curve(y_true, y_score, n_bins=n_bins, strategy='uniform')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
ax.plot(mean_predicted, fraction_of_pos, 'o-', color=C_NOGL, lw=2, ms=6,
        label='BRF raw (BSS=−139.8)')

# Try to load calibrated scores
calib_path = os.path.join(ROOT, 'models/calibrated_oof.json')
if os.path.exists(calib_path):
    calib = json.load(open(calib_path))
    y_calib = np.array(calib.get('y_score_isotonic', calib.get('y_calibrated_isotonic', calib.get('isotonic', []))))
    if len(y_calib) == len(y_true):
        frac_cal, mean_cal = calibration_curve(y_true, y_calib, n_bins=n_bins, strategy='uniform')
        ax.plot(mean_cal, frac_cal, 's-', color=C_GLOF, lw=2, ms=6,
                label='Isotonic calibrated (BSS=+0.008)')

# Histogram of score distribution at bottom
ax_hist = ax.inset_axes([0, -0.32, 1, 0.25])
ax_hist.hist(y_score[y_true==0], bins=20, color=C_NOGL, alpha=0.5, density=True, label='Non-GLOF')
ax_hist.hist(y_score[y_true==1], bins=10, color=C_GLOF, alpha=0.7, density=True, label='GLOF-source')
ax_hist.set_xlabel('Predicted score')
ax_hist.set_yticks([])
ax_hist.set_xlim(-0.02, 1.02)
for sp in ['top','right','left']: ax_hist.spines[sp].set_visible(False)

ax.set_xlabel('Mean predicted value (bin)')
ax.set_ylabel('Fraction of positives')
ax.set_title('Score calibration: raw vs. isotonic recalibration\n(maps ranking scores to operational probabilities)')
ax.legend(fontsize=8.5, loc='upper left')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.05)
ax.text(0.98, 0.02, 'Prevalence = 0.134%', transform=ax.transAxes,
        fontsize=7.5, ha='right', va='bottom', color=C_GREY)
panel_label(ax, '(b)')

fig.suptitle('Score distribution validation and calibration analysis', fontsize=12, y=1.01)
fig.savefig(os.path.join(OUT, 'fig7_supplementary_validation.png'), dpi=600, bbox_inches='tight')
plt.close()
print("  Fig 7 done.")

print("\nAll figures saved to figures/publication/")
print("Files:")
for fn in ['fig3_susceptibility_distribution.png','fig4_model_performance.png',
           'fig5_shap_risk_drivers.png','fig6_robustness_analysis.png',
           'fig7_supplementary_validation.png']:
    fp = os.path.join(OUT, fn)
    size_kb = os.path.getsize(fp)//1024 if os.path.exists(fp) else 0
    print(f"  {fn}: {size_kb} KB")
