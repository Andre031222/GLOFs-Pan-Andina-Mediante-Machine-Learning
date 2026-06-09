#!/usr/bin/env python3
"""
scripts/generate_fig05.py
-------------------------
Figure 5 — SHAP-Based Interpretation of GLOF Susceptibility Drivers.

Four-panel figure:
  (a) Beeswarm plot — top 14 SHAP features
  (b) SHAP dependence for glacier distance (non-linear peak 2-15 km)
  (c) Mean SHAP fingerprint: GLOF vs non-GLOF lakes
  (d) Susceptibility distribution by study area (violin/box)

Outputs:
  figures/publication/fig5_shap_risk_drivers.pdf / .png (600 DPI)
  figures/panels/fig5a_shap_beeswarm.png
  figures/panels/fig5b_glacier_distance_dependence.png
  figures/panels/fig5c_glof_vs_nonglof_shap.png
  figures/panels/fig5d_susceptibility_by_area.png

Run from project root:
    glof/Scripts/python scripts/generate_fig05.py
"""

import json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import joblib
from pathlib import Path

try:
    import shap as shap_lib
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("[warn] shap not installed — install with: pip install shap")

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
FIG_PUB    = ROOT / 'figures' / 'publication'
FIG_PAN    = ROOT / 'figures' / 'panels'
FIG_PUB.mkdir(parents=True, exist_ok=True)
FIG_PAN.mkdir(parents=True, exist_ok=True)

DPI_PNG   = 600
IEEE_FULL = 7.16

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
AREA_COLORS = {
    'cordillera_blanca':       '#1f77b4',
    'cordillera_vilcanota':    '#ff7f0e',
    'cordillera_central':      '#2ca02c',
    'chile_andes_centrales':   '#d62728',
    'cordillera_raura':        '#9467bd',
    'cordillera_urubamba':     '#8c564b',
    'cordillera_huanzo':       '#e377c2',
    'cordillera_huayhuash':    '#7f7f7f',
    'ecuador_antisana':        '#bcbd22',
    'bolivia_cordillera_real': '#17becf',
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
C_MAIN = '#0077BB'
C_GLOF = '#EE3377'

LABEL_MAP = {
    'dist_glacier_m':          'Distance to glacier (m)',
    'perimeter_m':             'Lake perimeter (m)',
    'depth_ensemble_std':      'Depth uncertainty (\u03c3)',
    'depth_m':                 'Mean depth (m)',
    'elev_max':                'Max elevation (m)',
    'depth_ensemble_mean':     'Ensemble depth (m)',
    'depth_yao_2012':          'Depth \u2013 Yao (m)',
    'log_volume_m3':           'log(Volume m\u00b3)',
    'tri_mean':                'Terrain ruggedness',
    'area_depth_ratio':        'Area / depth ratio',
    'slope_x_logarea':         'Slope \u00d7 log(area)',
    'log_area_m2':             'log(Area m\u00b2)',
    'slope_mean':              'Mean slope (\u00b0)',
    'elongation':              'Lake elongation',
    'no_glacier':              'No glacier nearby',
    'depth_cook_quincey_2015': 'Cook depth (m)',
    'depth_std_m':             'Depth std (m)',
}

IJACSA_STYLE = {
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'axes.linewidth':     0.8,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'savefig.facecolor':  'white',
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
}
plt.rcParams.update(IJACSA_STYLE)


# ── helpers ──────────────────────────────────────────────────────────────────

def save_panel(fig, ax, path, dpi=DPI_PNG, pad=0.12):
    try:
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            return
        bbox_in = bbox.transformed(fig.dpi_scale_trans.inverted())
        x0, y0, x1, y1 = bbox_in.extents
        padded = Bbox([[x0 - pad, y0 - pad], [x1 + pad, y1 + pad]])
        fig.savefig(path, bbox_inches=padded, dpi=dpi,
                    facecolor='white', edgecolor='none')
        print(f"    panel -> {Path(path).name}")
    except Exception as e:
        print(f"    [warn] panel save failed for {Path(path).name}: {e}")


def _clean_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#555')
    ax.tick_params(direction='out', length=3, width=0.6, colors='#333')


def _grid_ax(ax, axis='y'):
    kw = dict(color='#eeeeee', lw=0.5, zorder=0)
    if axis in ('y', 'both'):
        ax.yaxis.grid(True, **kw)
    if axis in ('x', 'both'):
        ax.xaxis.grid(True, **kw)
    ax.set_axisbelow(True)


def _panel_label(ax, label, x=0.03, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', color='#111')


# ── data loading ──────────────────────────────────────────────────────────────

def load_lakes_predictions():
    lakes = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')
    lakes['country'] = lakes['area_name'].map(COUNTRY_MAP)

    df_csv = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
    with open(MODELS_DIR / 'feature_names.json') as f:
        feature_cols = json.load(f)

    excl = {'lake_id', 'had_glof', 'glof_label', 'country', 'cordillera',
            'geometry', 'year', 'source', 'area_name', 'scene_date',
            'glof_volume_m3', 'glof_match_dist_m', 'glof_date',
            'glof_trigger', 'glof_lake_name'}
    base_cols = [c for c in df_csv.columns
                 if c not in excl and df_csv[c].dtype in
                    ['float64', 'float32', 'int64', 'int32']]
    X = df_csv[base_cols].fillna(df_csv[base_cols].median()).fillna(0)

    for col in ['area_m2', 'perimeter_m', 'volume_m3', 'equivalent_diameter_m',
                'potential_energy', 'depth_m', 'dam_height',
                'area_2017_est', 'area_2025_est', 'total_growth_m2']:
        if col in X.columns:
            X[f'log_{col}'] = np.log1p(X[col].clip(lower=0))
    if 'total_growth_m2' in X.columns and 'area_2017_est' in X.columns:
        X['rel_growth'] = X['total_growth_m2'] / (X['area_2017_est'] + 1)
    if 'slope_mean' in X.columns and 'log_area_m2' in X.columns:
        X['slope_x_logarea'] = X['slope_mean'] * X['log_area_m2']
    if 'dist_glacier_m' in df_csv.columns:
        X['no_glacier'] = df_csv['dist_glacier_m'].isna().astype(int)

    feat_avail = [c for c in feature_cols if c in X.columns]
    model = joblib.load(MODELS_DIR / 'best_model.joblib')
    lakes['y_prob'] = model.predict_proba(X[feat_avail])[:, 1]

    print(f"  Lakes: {len(lakes):,} | GLOF+: {int(lakes['had_glof'].sum())}")
    return lakes, X[feat_avail]


# ── main figure ───────────────────────────────────────────────────────────────

def generate():
    if not SHAP_OK:
        print("[skip] Figure 5 requires shap. Install: pip install shap")
        return

    print("Figure 5: SHAP-Based Interpretation of GLOF Susceptibility Drivers")
    lakes, X_pred = load_lakes_predictions()

    # Compute SHAP values
    model_shap = joblib.load(MODELS_DIR / 'best_model.joblib')
    print("  Computing SHAP values...")
    explainer = shap_lib.TreeExplainer(model_shap)
    sv_raw    = explainer.shap_values(X_pred)
    # Handle 3-D array (n_samples, n_features, n_classes) or list
    if isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv = sv_raw[:, :, 1]
    elif isinstance(sv_raw, list):
        sv = sv_raw[1]
    else:
        sv = sv_raw
    print(f"  SHAP array shape: {sv.shape}")

    feat_names = list(X_pred.columns)
    _mean_abs  = np.abs(sv).mean(axis=0)
    # Indices sorted by mean |SHAP|, descending — top 10 (reduced for readability)
    _top10_idx = np.argsort(_mean_abs)[-10:][::-1]
    # Reversed for beeswarm (lowest importance at bottom = rank 0)
    _bee_order = _top10_idx[::-1]

    fig, _axes = plt.subplots(2, 2, figsize=(IEEE_FULL, 10.0), facecolor='white',
                               gridspec_kw=dict(hspace=0.58, wspace=0.62))
    ax_BEE, ax_DEP, ax_FPR, ax_VIO = _axes.flat

    # ── (a) Beeswarm — top 10 features ───────────────────────────────────────
    _rng_bee = np.random.default_rng(42)
    _glof_m  = lakes['had_glof'].values == 1
    _n_samp  = sv.shape[0]

    # Subsample non-GLOF points for clarity (keep all GLOFs)
    _max_pts = 4000
    _idx_all = np.arange(_n_samp)
    _non_glof_idx = _idx_all[~_glof_m]
    if len(_non_glof_idx) > _max_pts:
        _rng_sub = np.random.default_rng(123)
        _plot_idx = np.concatenate([
            _idx_all[_glof_m],
            _rng_sub.choice(_non_glof_idx, _max_pts, replace=False)
        ])
    else:
        _plot_idx = _idx_all

    for _rank, _fi in enumerate(_bee_order):
        _sv_f = sv[_plot_idx, _fi]
        _fv   = X_pred.iloc[_plot_idx, _fi].values
        _p1, _p99 = np.nanpercentile(_fv, [1, 99])
        _fv_n = np.clip((_fv - _p1) / (_p99 - _p1 + 1e-9), 0, 1)
        # Tighter jitter since fewer points
        _jit  = _rng_bee.uniform(-0.30, 0.30, size=len(_plot_idx))

        ax_BEE.scatter(_sv_f, np.full(len(_plot_idx), _rank) + _jit,
                       c=_fv_n, cmap='RdBu_r', vmin=0, vmax=1,
                       s=4.5, alpha=0.45, linewidths=0, rasterized=True,
                       zorder=2)
        # GLOF-source lakes: black diamonds (always plotted, full set)
        _sv_g = sv[_glof_m, _fi]
        if len(_sv_g) > 0:
            ax_BEE.scatter(_sv_g,
                           np.full(len(_sv_g), _rank),
                           s=65, marker='D', edgecolors='white',
                           linewidths=0.7, color='black', zorder=9)

    _ylabels = [LABEL_MAP.get(feat_names[i], feat_names[i])
                for i in _bee_order]
    ax_BEE.set_yticks(range(len(_bee_order)))
    ax_BEE.set_yticklabels(_ylabels, fontsize=7.5)
    ax_BEE.axvline(0, color='#777', lw=0.9, ls='--')
    ax_BEE.set_xlabel('SHAP value  (impact on model output)', fontsize=9)
    ax_BEE.set_title('Global feature importance (top 10)', fontsize=10)

    # Colorbar for feature value
    _sm_bee = ScalarMappable(cmap='RdBu_r', norm=mcolors.Normalize(0, 1))
    _sm_bee.set_array([])
    _cb_bee = plt.colorbar(_sm_bee, ax=ax_BEE, fraction=0.028, pad=0.04)
    _cb_bee.set_label('Feature value\n(low \u2192 high)', fontsize=8)
    _cb_bee.set_ticks([0, 0.5, 1])
    _cb_bee.ax.set_yticklabels(['Low', '', 'High'], fontsize=7.5)
    _cb_bee.outline.set_linewidth(0.5)

    # Legend proxy for GLOF diamond
    _glof_proxy = Line2D([], [], marker='D', color='w', markerfacecolor='black',
                         markeredgecolor='white', markersize=7,
                         label='GLOF-source lake')
    ax_BEE.legend(handles=[_glof_proxy], loc='lower right', fontsize=8,
                  framealpha=0.9, edgecolor='#ddd')

    for sp in ['top', 'right']:
        ax_BEE.spines[sp].set_visible(False)
    _panel_label(ax_BEE, '(a)', x=0.02, y=0.99)

    # ── (b) Glacier distance dependence ───────────────────────────────────────
    _dist_col = 'dist_glacier_m'
    if _dist_col in feat_names:
        _di   = feat_names.index(_dist_col)
        _dv   = X_pred[_dist_col].values / 1000.0   # convert to km
        _ds   = sv[:, _di]
        _mask = (_dv < 50) & ~np.isnan(_dv) & ~np.isnan(_ds)

        sc_d = ax_DEP.scatter(_dv[_mask], _ds[_mask],
                              c=_dv[_mask], cmap='plasma_r', vmin=0, vmax=50,
                              s=4, alpha=0.25, linewidths=0, rasterized=True,
                              zorder=2)

        # LOWESS-style smoothing via rolling mean on sorted data
        _df_dep = (pd.DataFrame({'x': _dv[_mask], 'y': _ds[_mask]})
                   .sort_values('x').reset_index(drop=True))
        _win = max(5, len(_df_dep) // 40)
        _roll = _df_dep.rolling(window=_win, center=True, min_periods=5).mean()
        _roll_clean = _roll.dropna()
        if len(_roll_clean) > 1:
            ax_DEP.plot(_roll_clean['x'], _roll_clean['y'],
                        color=C_GLOF, lw=2.5, zorder=6,
                        label='Smoothed trend')

        ax_DEP.axhline(0, color='#888', lw=0.8, ls='--')
        ax_DEP.axvspan(2, 15, alpha=0.10, color='#c0392b', zorder=1,
                       label='Peak risk zone (2\u201315 km)')

        # Annotate the peak zone
        _ylim_dep = ax_DEP.get_ylim()
        _ann_y    = _ylim_dep[0] + (_ylim_dep[1] - _ylim_dep[0]) * 0.70
        ax_DEP.text(8.5, _ann_y,
                    '\u2190 peak\nrisk\nzone',
                    fontsize=7.5, color='#c0392b', ha='center', va='center')

        ax_DEP.set_xlabel('Distance to nearest glacier (km)', fontsize=9)
        ax_DEP.set_ylabel('SHAP value', fontsize=9)
        ax_DEP.set_title('Non-linear susceptibility peak at 2\u201315 km',
                          fontsize=10)

        _cb_d = plt.colorbar(sc_d, ax=ax_DEP, fraction=0.04, pad=0.03)
        _cb_d.set_label('Distance (km)', fontsize=7.5)
        _cb_d.ax.tick_params(labelsize=7)
        _cb_d.outline.set_linewidth(0.5)

        ax_DEP.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ddd')
    else:
        ax_DEP.text(0.5, 0.5, 'dist_glacier_m not in features',
                    ha='center', va='center', transform=ax_DEP.transAxes,
                    fontsize=8, color='#888')

    _clean_ax(ax_DEP)
    _grid_ax(ax_DEP, 'y')
    _panel_label(ax_DEP, '(b)')

    # ── (c) Dumbbell chart: GLOF vs non-GLOF mean SHAP ──────────────────────
    _glof_mask = lakes['had_glof'].values == 1
    _mean_shap_glof = sv[_glof_mask].mean(axis=0)
    _mean_shap_neg  = sv[~_glof_mask].mean(axis=0)

    # Sort by (GLOF - nonGLOF) diff ascending → most impactful at top
    _diff     = _mean_shap_glof[_top10_idx] - _mean_shap_neg[_top10_idx]
    _sort_ord = np.argsort(_diff)
    _sorted_idx = _top10_idx[_sort_ord]

    _labels10 = [LABEL_MAP.get(feat_names[i], feat_names[i])
                 for i in _sorted_idx]
    _pos = np.arange(len(_sorted_idx))

    _g_vals = _mean_shap_glof[_sorted_idx]
    _n_vals = _mean_shap_neg[_sorted_idx]

    # Light grid first
    _grid_ax(ax_FPR, 'x')

    # Connecting segment between the two dots (gray, behind)
    for i, (gv, nv) in enumerate(zip(_g_vals, _n_vals)):
        ax_FPR.plot([nv, gv], [i, i],
                    color='#cccccc', lw=2.2, zorder=2,
                    solid_capstyle='round')

    # Non-GLOF dots (blue, square)
    ax_FPR.scatter(_n_vals, _pos, s=52, color=C_MAIN, zorder=5,
                   label='Non-GLOF', edgecolors='white', linewidths=0.5,
                   marker='s')
    # GLOF-source dots (pink, circle)
    ax_FPR.scatter(_g_vals, _pos, s=58, color=C_GLOF, zorder=6,
                   label='GLOF-source', edgecolors='white', linewidths=0.5)

    # Zero reference line
    ax_FPR.axvline(0, color='#888', lw=0.9, ls='--', zorder=1)

    # Difference arrow annotation for the top feature (largest gap)
    _top_i = len(_pos) - 1
    _gv_top, _nv_top = _g_vals[_top_i], _n_vals[_top_i]
    _diff_top = _gv_top - _nv_top
    ax_FPR.annotate('', xy=(_gv_top, _top_i), xytext=(_nv_top, _top_i),
                    arrowprops=dict(arrowstyle='->', color='#c0392b',
                                   lw=1.0, shrinkA=4, shrinkB=4))
    # diff label: place BELOW the top row (inside plot) to avoid going off-axis
    ax_FPR.text((_gv_top + _nv_top) / 2, _top_i - 0.38,
                f'+{_diff_top:.4f}',
                ha='center', va='top', fontsize=6.5, color='#c0392b')

    ax_FPR.set_yticks(_pos)
    ax_FPR.set_yticklabels(_labels10, fontsize=7.5)
    ax_FPR.set_xlabel('Mean SHAP contribution', fontsize=9)
    ax_FPR.set_title('SHAP fingerprint: GLOF vs. non-GLOF', fontsize=10)
    ax_FPR.legend(loc='lower right', fontsize=8.5, framealpha=0.95,
                  edgecolor='#ddd')
    _clean_ax(ax_FPR)
    _panel_label(ax_FPR, '(c)', x=0.72, y=0.97)

    # ── (d) Susceptibility by study area (violin) ─────────────────────────────
    _areas_sorted = (lakes.groupby('area_name')['y_prob']
                     .median().sort_values(ascending=False).index.tolist())
    _viol_data = [lakes.loc[lakes['area_name'] == a, 'y_prob'].values
                  for a in _areas_sorted]
    # Filter out areas with no data
    _areas_ok   = [a for a, d in zip(_areas_sorted, _viol_data) if len(d) > 1]
    _viol_data  = [d for d in _viol_data if len(d) > 1]

    vp = ax_VIO.violinplot(_viol_data, vert=False, showmedians=True,
                            widths=0.70)
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(AREA_COLORS.get(_areas_ok[i], '#888'))
        pc.set_alpha(0.78)
        pc.set_edgecolor('#333')
        pc.set_linewidth(0.5)
    vp['cmedians'].set_color('#333')
    vp['cmedians'].set_linewidth(1.5)
    for v_part in ['cbars', 'cmaxes', 'cmins']:
        if v_part in vp:
            vp[v_part].set_color('#555')
            vp[v_part].set_linewidth(0.7)

    # Overlay GLOF-source lake scores as red stars
    for _a, y_pos in zip(_areas_ok, range(1, len(_areas_ok) + 1)):
        _g_probs = lakes.loc[(lakes['area_name'] == _a) &
                             (lakes['had_glof'] == 1), 'y_prob'].values
        if len(_g_probs) > 0:
            ax_VIO.scatter(_g_probs, np.full(len(_g_probs), y_pos),
                           s=70, marker='*', c='#c0392b',
                           zorder=9, edgecolors='k', linewidths=0.4)

    ax_VIO.axvline(0.304, color='#c0392b', lw=1.5, ls='--', alpha=0.80,
                   zorder=4)
    # Youden label: blended transform → x in data coords, y in axes fraction
    # so it stays at the top of the panel regardless of ylim
    from matplotlib.transforms import blended_transform_factory as _btf
    _tx = _btf(ax_VIO.transData, ax_VIO.transAxes)
    ax_VIO.text(0.304 + 0.012, 0.97,
                'Youden\n0.304',
                transform=_tx, fontsize=6.5, color='#c0392b',
                ha='left', va='top', clip_on=False,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#c0392b', alpha=0.88, linewidth=0.5))
    ax_VIO.set_yticks(range(1, len(_areas_ok) + 1))
    ax_VIO.set_yticklabels([AREA_LABELS.get(a, a) for a in _areas_ok],
                            fontsize=7.5)
    ax_VIO.set_xlabel('Predicted GLOF susceptibility score', fontsize=9)
    # Legend only for GLOF-source star marker
    _star_prx = Line2D([], [], marker='*', color='#c0392b', linestyle='None',
                       markersize=8, label='GLOF-source lake',
                       markeredgecolor='k', markeredgewidth=0.4)
    ax_VIO.legend(handles=[_star_prx], loc='upper left', fontsize=8,
                  framealpha=0.9, edgecolor='#ddd')
    _clean_ax(ax_VIO)
    _panel_label(ax_VIO, '(d)')

    # ── Export ────────────────────────────────────────────────────────────────
    out_pdf = FIG_PUB / 'fig5_shap_risk_drivers.pdf'
    out_png = FIG_PUB / 'fig5_shap_risk_drivers.png'
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png, dpi=DPI_PNG, bbox_inches='tight', facecolor='white')

    for ax, name in [(ax_BEE, 'fig5a_shap_beeswarm'),
                     (ax_DEP, 'fig5b_glacier_distance_dependence'),
                     (ax_FPR, 'fig5c_glof_vs_nonglof_shap'),
                     (ax_VIO, 'fig5d_susceptibility_by_area')]:
        save_panel(fig, ax, FIG_PAN / f'{name}.png')

    plt.close(fig)
    for p in [out_pdf, out_png]:
        if p.exists():
            print(f"  {p.name:50s}  {p.stat().st_size / 1024:>8.1f} KB")
    print("  Figure 5 complete.")


if __name__ == '__main__':
    generate()
