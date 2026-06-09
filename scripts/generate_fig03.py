#!/usr/bin/env python3
"""
generate_fig03.py
-----------------
Figure 3 — Susceptibility Score Distribution.

Two-panel figure:
  (a) Lake morphology scatter (area vs. depth) colored by susceptibility score
  (b) KDE of predicted GLOF susceptibility for GLOF vs. non-GLOF lakes

Generates:
  figures/publication/fig3_susceptibility_distribution.pdf
  figures/publication/fig3_susceptibility_distribution.png  (600 DPI)

Run from project root:
    glof/Scripts/python scripts/generate_fig03.py
"""

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
FIG_PUB    = ROOT / 'figures' / 'publication'
FIG_PAN    = ROOT / 'figures' / 'panels'
FIG_PUB.mkdir(parents=True, exist_ok=True)
FIG_PAN.mkdir(parents=True, exist_ok=True)

DPI_PNG   = 600
IEEE_FULL = 7.16

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    'cordillera_blanca':        'Peru',
    'cordillera_vilcanota':     'Peru',
    'cordillera_central':       'Peru',
    'cordillera_raura':         'Peru',
    'cordillera_urubamba':      'Peru',
    'cordillera_huanzo':        'Peru',
    'cordillera_huayhuash':     'Peru',
    'bolivia_cordillera_real':  'Bolivia',
    'ecuador_antisana':         'Ecuador',
    'chile_andes_centrales':    'Chile',
}
AREA_COLORS = {
    'cordillera_blanca':        '#1f77b4',
    'cordillera_vilcanota':     '#ff7f0e',
    'cordillera_central':       '#2ca02c',
    'chile_andes_centrales':    '#d62728',
    'cordillera_raura':         '#9467bd',
    'cordillera_urubamba':      '#8c564b',
    'cordillera_huanzo':        '#e377c2',
    'cordillera_huayhuash':     '#7f7f7f',
    'ecuador_antisana':         '#bcbd22',
    'bolivia_cordillera_real':  '#17becf',
}
AREA_LABELS = {
    'cordillera_blanca':        'C. Blanca (PE)',
    'cordillera_vilcanota':     'C. Vilcanota (PE)',
    'cordillera_central':       'C. Central (PE)',
    'chile_andes_centrales':    'Andes Centrales (CL)',
    'cordillera_raura':         'C. Raura (PE)',
    'cordillera_urubamba':      'C. Urubamba (PE)',
    'cordillera_huanzo':        'C. Huanzo (PE)',
    'cordillera_huayhuash':     'C. Huayhuash (PE)',
    'ecuador_antisana':         'Antisana (EC)',
    'bolivia_cordillera_real':  'C. Real (BO)',
}
COUNTRY_COLORS = {
    'Peru':    '#c0392b',
    'Ecuador': '#27ae60',
    'Bolivia': '#2980b9',
    'Chile':   '#e67e22',
}
TRIG_COLORS = {
    'Moraine dam':  '#c0392b',
    'Ice dam':      '#2980b9',
    'Bedrock dam':  '#27ae60',
    'Supraglacial': '#e67e22',
    'Unknown':      '#95a5a6',
}
C_MAIN = '#0077BB'
C_GLOF = '#EE3377'

IJACSA_STYLE = {
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.8,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
}

# Custom colormap: blue -> white -> red (susceptibility)
SUSC_CMAP = LinearSegmentedColormap.from_list(
    'glof_susc',
    ['#2166ac', '#f7f7f7', '#d6604d', '#b2182b'],
    N=256,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#555')
    ax.tick_params(direction='out', length=3, width=0.6, colors='#333')
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax, label, x=0.03, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', color='#111')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    lakes = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')
    glofs = gpd.read_file(DATA_DIR / 'labeled' / 'historical_glofs.gpkg')
    lakes['country'] = lakes['area_name'].map(COUNTRY_MAP)

    df_csv = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')

    with open(MODELS_DIR / 'feature_names.json') as f:
        feature_cols = json.load(f)

    excl = {
        'lake_id', 'had_glof', 'glof_label', 'country', 'cordillera',
        'geometry', 'year', 'source', 'area_name', 'scene_date',
        'glof_volume_m3', 'glof_match_dist_m', 'glof_date',
        'glof_trigger', 'glof_lake_name',
    }
    base_cols = [
        c for c in df_csv.columns
        if c not in excl and df_csv[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
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

    glofs['year_int'] = pd.to_datetime(glofs['date'], errors='coerce').dt.year

    return lakes, glofs


# ---------------------------------------------------------------------------
# Panel (a): Lake morphology scatter — area vs. depth, colored by score
# ---------------------------------------------------------------------------
def plot_panel_a(ax, lakes):
    # Determine column names with fallbacks
    area_col  = 'area_m2'   if 'area_m2'  in lakes.columns else None
    depth_col = 'depth_m'   if 'depth_m'  in lakes.columns else None
    glof_col  = 'had_glof'  if 'had_glof' in lakes.columns else 'glof_label'

    if area_col is None or depth_col is None:
        ax.text(0.5, 0.5, 'Required columns (area_m2 / depth_m) not found',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        _panel_label(ax, '(a)')
        return

    # Filter to rows with valid depth
    valid = lakes.dropna(subset=[depth_col])
    valid = valid[valid[depth_col] > 0]
    valid = valid[valid[area_col] > 0]

    is_glof     = valid[glof_col].astype(int) == 1 if glof_col in valid.columns else pd.Series(False, index=valid.index)
    non_glof_df = valid[~is_glof]
    glof_df     = valid[is_glof]

    # Sub-sample non-GLOF to keep plot manageable
    n_sample = min(3000, len(non_glof_df))
    if n_sample > 0:
        non_glof_sample = non_glof_df.sample(n_sample, random_state=42)
    else:
        non_glof_sample = non_glof_df

    x_ng = non_glof_sample[area_col].values / 1e4
    y_ng = non_glof_sample[depth_col].values
    c_ng = non_glof_sample['y_prob'].values

    sc = ax.scatter(
        x_ng, y_ng,
        c=c_ng, cmap=SUSC_CMAP, vmin=0, vmax=1,
        s=8, alpha=0.30, linewidths=0, rasterized=True, zorder=3,
    )

    # GLOF lakes on top
    if len(glof_df) > 0:
        x_gl = glof_df[area_col].values / 1e4
        y_gl = glof_df[depth_col].values
        c_gl = glof_df['y_prob'].values

        ax.scatter(
            x_gl, y_gl,
            c=c_gl, cmap=SUSC_CMAP, vmin=0, vmax=1,
            s=110, alpha=1.0, marker='D',
            edgecolors='#111', linewidths=0.9, zorder=8,
        )

        # Label top 4 GLOF lakes by area — staggered offsets to avoid overlap
        top4 = glof_df.nlargest(4, area_col)
        name_col = None
        for nc in ['lake_name', 'glof_lake_name', 'name']:
            if nc in top4.columns:
                name_col = nc
                break
        # Sort by depth descending so topmost point gets top offset
        top4_sorted = top4.sort_values(depth_col, ascending=False)
        # Only label 3 lakes (4th tends to land on the colorbar tick labels)
        # Offset pairs (dx, dy) in points — staggered vertically to the right
        label_offsets = [(14, 12), (14, 0), (14, -13)]
        for (_, row), (dx, dy) in zip(
                list(top4_sorted.iterrows())[:3], label_offsets):
            lbl = row[name_col][:14] if name_col and pd.notna(row.get(name_col)) else ''
            if lbl:
                ax.annotate(
                    lbl,
                    xy=(row[area_col] / 1e4, row[depth_col]),
                    xytext=(dx, dy),
                    textcoords='offset points',
                    fontsize=6.5, fontstyle='italic', color='#222',
                    va='center', ha='left', zorder=9,
                    arrowprops=dict(
                        arrowstyle='-', lw=0.5, color='#666', alpha=0.65,
                        shrinkA=3, shrinkB=2,
                    ),
                )

    # Log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    _clean_ax(ax)

    ax.set_xlabel(r'Lake area ($\times 10^4$ m$^2$)')
    ax.set_ylabel('Mean depth (m)')
    ax.set_title('Lake morphology and susceptibility score', fontsize=10, pad=4)

    # Colorbar — extra pad so nearby lake labels don't overlap tick text
    cbar = plt.colorbar(sc, ax=ax, fraction=0.033, pad=0.07)
    cbar.set_label('GLOF susceptibility score', fontsize=8.5)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=7.5)

    # Legend for GLOF marker
    from matplotlib.lines import Line2D
    diamond = Line2D(
        [], [], linestyle='None', marker='D',
        markerfacecolor='#888888', markeredgecolor='#111',
        markeredgewidth=0.9, markersize=7,
        label='GLOF-source lake',
    )
    ax.legend(handles=[diamond], fontsize=8, loc='lower right',
              framealpha=0.85, edgecolor='#ccc')

    _panel_label(ax, '(a)')


# ---------------------------------------------------------------------------
# Panel (b): KDE — susceptibility score distribution
#
# The GLOF KDE (n=16) produces a sharp spike near score=1 that dwarfs
# the non-GLOF KDE.  We cap the y-axis at 1.3× the non-GLOF peak so
# both distributions are readable, and annotate the truncated GLOF peak.
# ---------------------------------------------------------------------------
def plot_panel_b(ax, lakes):
    glof_col = 'had_glof' if 'had_glof' in lakes.columns else 'glof_label'

    p_all  = lakes['y_prob'].values
    labels = (lakes[glof_col].astype(int).values
              if glof_col in lakes.columns
              else np.zeros(len(lakes), dtype=int))

    p_glof    = p_all[labels == 1]
    p_nonglof = p_all[labels == 0]

    x = np.linspace(0, 1, 500)

    # Non-GLOF KDE
    y_ng = np.zeros_like(x)
    if len(p_nonglof) > 1:
        kde_ng = gaussian_kde(p_nonglof, bw_method=0.12)
        y_ng   = kde_ng(x)
        ax.fill_between(x, y_ng, alpha=0.30, color=C_MAIN, zorder=3)
        ax.plot(x, y_ng, color=C_MAIN, lw=2.0, zorder=4,
                label=f'Non-GLOF (n\u202f=\u202f{len(p_nonglof):,})')

    # GLOF KDE — wider bandwidth reduces artefact spike from small n
    y_g = np.zeros_like(x)
    if len(p_glof) > 1:
        kde_g = gaussian_kde(p_glof, bw_method=0.28)
        y_g   = kde_g(x)
        ax.fill_between(x, y_g, alpha=0.50, color=C_GLOF, zorder=5)
        ax.plot(x, y_g, color=C_GLOF, lw=2.5, zorder=6,
                label=f'GLOF-source (n\u202f=\u202f{len(p_glof):,})')

    # Cap y-axis at 1.3× the non-GLOF peak so both KDEs are visible
    ng_peak   = float(y_ng.max()) if y_ng.max() > 0 else 1.0
    y_cap     = ng_peak * 1.35
    y_cap     = max(y_cap, float(y_g.max()) * 0.25, 1.5)  # show at least 25% of GLOF peak

    # Rug plot for individual GLOF scores — just above the x-axis baseline
    rug_y = -0.025 * y_cap
    if len(p_glof) > 0:
        ax.scatter(
            p_glof, np.full(len(p_glof), rug_y),
            marker='|', s=90, color=C_GLOF,
            zorder=7, linewidths=1.8, clip_on=False,
        )

    # Youden threshold line (t=0.356, from optimal_thresholds.json)
    youden_t = 0.356
    ax.axvline(youden_t, color='#555555', lw=1.5, ls='--', zorder=6)

    # Annotation placed LEFT of threshold where density is low
    ax.annotate(
        f't\u202f=\u202f{youden_t:.3f}\n(Youden opt.)',
        xy=(youden_t, y_cap * 0.52),
        xytext=(youden_t - 0.20, y_cap * 0.60),
        fontsize=7.5, color='#444', ha='center', va='center',
        arrowprops=dict(arrowstyle='->', lw=0.8, color='#666'),
    )

    # If GLOF KDE exceeds the cap, annotate the clipped peak
    glof_peak = float(y_g.max())
    if glof_peak > y_cap * 0.92:
        peak_x = float(x[np.argmax(y_g)])
        ax.annotate(
            f'GLOF peak\n\u2248\u2009{glof_peak:.0f}',
            xy=(min(peak_x, 0.97), y_cap),
            xytext=(max(peak_x - 0.18, 0.55), y_cap * 0.82),
            fontsize=7, color=C_GLOF, ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.7, color=C_GLOF),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(rug_y * 1.5, y_cap)
    ax.set_xlabel('Predicted GLOF susceptibility score')
    ax.set_ylabel('Density')
    ax.set_title('Predicted susceptibility distribution', fontsize=10, pad=4)

    _clean_ax(ax)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.85, edgecolor='#ccc')
    _panel_label(ax, '(b)')


# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------
def generate():
    plt.rcParams.update(IJACSA_STYLE)
    print("Loading data...")
    lakes, glofs = load_data()
    print(f"  Lakes: {len(lakes):,}  |  GLOFs: {len(glofs):,}")

    # Single-column layout: panels stacked vertically (2 rows × 1 col)
    # Width = ~3.5" (IEEEtran single column); height = 7.0" for two stacked panels
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(3.5, 7.0),
        gridspec_kw=dict(hspace=0.52),
    )

    print("  Plotting panel (a)...")
    plot_panel_a(ax_a, lakes)

    print("  Plotting panel (b)...")
    plot_panel_b(ax_b, lakes)

    for ext, kw in [('.pdf', {}), ('.png', {'dpi': DPI_PNG})]:
        out = FIG_PUB / f'fig3_susceptibility_distribution{ext}'
        fig.savefig(out, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f"  {out.name}  {out.stat().st_size / 1024:.1f} KB")

    plt.close(fig)
    print("  Figure 3 complete.")


if __name__ == '__main__':
    generate()
