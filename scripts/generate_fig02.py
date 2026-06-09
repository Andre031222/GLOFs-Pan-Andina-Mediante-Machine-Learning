#!/usr/bin/env python3
"""
generate_fig02.py
-----------------
Figure 2 -- Glacial Lake Inventory and GLOF Historical Context.

Four-panel figure:
  (a) Total glacial lake area per country 2017-2025
  (b) Lake observations per cordillera (horizontal bar + GLOF rate twin axis)
  (c) GLOF trigger mechanisms (donut chart)
  (d) Historical GLOF timeline 1932-2021 (stem-style scatter)

Generates:
  figures/publication/fig2_inventory_glof_context.pdf
  figures/publication/fig2_inventory_glof_context.png  (600 DPI)

Run from project root:
    glof/Scripts/python scripts/generate_fig02.py
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
from matplotlib.ticker import MultipleLocator
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

# GLOF trigger label mapping
TRIG_LABEL_MAP = {
    'ice_avalanche':   'Ice avalanche',
    'rockfall':        'Rockfall',
    'moraine_failure': 'Moraine dam',
    'overtopping':     'Overtopping',
    'piping':          'Piping',
    'earthquake':      'Earthquake',
    'Moraine dam':     'Moraine dam',
    'Ice dam':         'Ice dam',
    'Bedrock dam':     'Bedrock dam',
    'Supraglacial':    'Supraglacial',
    'Unknown':         'Unknown',
}

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


def _spread_labels(ys, min_gap):
    """
    Spread a sorted list of y-positions so consecutive values differ by at
    least min_gap. Uses bidirectional adjustment (push both neighbors) for
    minimal total displacement.
    """
    n = len(ys)
    if n <= 1:
        return list(ys)
    adjusted = list(ys)
    for _ in range(80):
        moved = False
        for i in range(1, n):
            gap = adjusted[i] - adjusted[i - 1]
            if gap < min_gap:
                push = (min_gap - gap) / 2.0
                adjusted[i - 1] -= push
                adjusted[i]     += push
                moved = True
        if not moved:
            break
    return adjusted


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    lakes  = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')
    glofs  = gpd.read_file(DATA_DIR / 'labeled' / 'historical_glofs.gpkg')
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

    return lakes, glofs, df_csv


# ---------------------------------------------------------------------------
# Panel (a): Total lake area 2017-2025 by country
# End-of-line labels are spread with _spread_labels() so they never overlap.
# ---------------------------------------------------------------------------
def plot_panel_a(ax, lakes, df_csv):
    yr_cols = sorted(
        [c for c in df_csv.columns
         if c.startswith('area_20') and len(c) == 9 and not c.endswith('est')],
    )
    if not yr_cols:
        ax.text(0.5, 0.5, 'No annual area columns found',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        _panel_label(ax, '(a)')
        return

    years = [int(c.split('_')[1]) for c in yr_cols]

    plot_df = df_csv.copy()
    if 'area_name' not in plot_df.columns and 'area_name' in lakes.columns:
        plot_df['area_name'] = lakes['area_name'].values
    if 'country' not in plot_df.columns:
        plot_df['country'] = plot_df['area_name'].map(COUNTRY_MAP)

    # Replace zeros with NaN (no observation that year)
    area_df = plot_df[yr_cols].replace(0, np.nan)
    plot_df_nz = plot_df.copy()
    plot_df_nz[yr_cols] = area_df

    # Cap per-lake area at 99th percentile to suppress outliers
    p99 = area_df.stack().quantile(0.99)
    for c in yr_cols:
        plot_df_nz[c] = plot_df_nz[c].clip(upper=p99)

    # Country totals, m2 -> km2
    country_groups = plot_df_nz.groupby('country')[yr_cols].sum() / 1e6

    # Per-cordillera faint dashed lines (visual texture)
    cord_groups = plot_df_nz.groupby('area_name')[yr_cols].sum() / 1e6
    for cord, row in cord_groups.iterrows():
        vals       = row[yr_cols].values.astype(float)
        valid_mask = ~np.isnan(vals) & (vals > 0)
        if valid_mask.sum() < 2:
            continue
        color  = COUNTRY_COLORS.get(COUNTRY_MAP.get(cord, 'Peru'), '#888')
        yrs_v  = [years[i] for i in range(len(years)) if valid_mask[i]]
        vals_v = vals[valid_mask]
        ax.plot(yrs_v, vals_v, lw=0.6, alpha=0.22, ls='--', color=color, zorder=2)

    # Country bold lines -- legend instead of end-of-line labels to avoid overlap
    for country, row in country_groups.iterrows():
        vals       = row[yr_cols].values.astype(float)
        valid_mask = ~np.isnan(vals) & (vals > 0)
        if valid_mask.sum() < 2:
            continue
        color  = COUNTRY_COLORS.get(country, '#888')
        yrs_v  = [years[i] for i in range(len(years)) if valid_mask[i]]
        vals_v = vals[valid_mask]
        ax.plot(yrs_v, vals_v, lw=2.5, color=color, zorder=5, label=country)
        ax.scatter(yrs_v, vals_v, s=40, color=color, zorder=5, marker='o')

    _clean_ax(ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total lake area (km²)')
    ax.set_title('Total glacial lake area 2017–2025', fontsize=10, pad=4)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.8)

    ax.legend(
        loc='upper left',
        fontsize=7.5,
        framealpha=0.90,
        edgecolor='#ccc',
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.5,
    )

    # Place "(a)" at upper-right so it is not hidden behind the legend
    _panel_label(ax, '(a)', x=0.72, y=0.97)


# ---------------------------------------------------------------------------
# Panel (b): Lake observations per cordillera (bar + GLOF rate twin axis)
# xlim is set explicitly so the count labels printed after each bar never
# overflow the axes boundaries.
# ---------------------------------------------------------------------------
def plot_panel_b(ax, lakes):
    agg = (
        lakes.groupby('area_name')
        .agg(n_lakes=('area_m2', 'count'), n_glofs=('had_glof', 'sum'))
        .reset_index()
    )
    agg['glof_rate'] = agg['n_glofs'] / agg['n_lakes'] * 100
    agg = agg.sort_values('n_lakes', ascending=True).reset_index(drop=True)

    y_pos  = np.arange(len(agg))
    colors = [AREA_COLORS.get(a, '#888888') for a in agg['area_name']]
    max_n  = int(agg['n_lakes'].max())

    # Pre-set xlim so count labels have guaranteed space
    ax.set_xlim(0, max_n * 1.22)

    bars = ax.barh(y_pos, agg['n_lakes'], height=0.65,
                   color=colors, alpha=0.85, edgecolor='white', zorder=3)

    for bar, n in zip(bars, agg['n_lakes']):
        ax.text(
            bar.get_width() + max_n * 0.013,
            bar.get_y() + bar.get_height() / 2,
            str(int(n)),
            va='center', ha='left', fontsize=7.5, color='#333',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [AREA_LABELS.get(a, a) for a in agg['area_name']], fontsize=7.5
    )
    ax.set_xlabel('Number of lake-year observations')
    ax.set_title('Lake observations per cordillera', fontsize=10, pad=4)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#555')
    ax.tick_params(direction='out', length=3, width=0.6, colors='#333')
    ax.xaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Twin x-axis: GLOF occurrence rate (%)
    ax2      = ax.twiny()
    rate_max = max(agg['glof_rate'].max() * 1.20, 2.0)
    ax2.set_xlim(0, rate_max)
    ax2.scatter(
        agg['glof_rate'].clip(upper=rate_max), y_pos,
        marker='D', s=55, color='#c0392b', zorder=6, clip_on=True,
    )
    ax2.set_xlabel('GLOF occurrence rate (%)', color='#c0392b', fontsize=8)
    ax2.tick_params(axis='x', colors='#c0392b', labelsize=7.5)
    ax2.spines['top'].set_color('#c0392b')
    ax2.spines['top'].set_linewidth(0.8)
    for sp in ['right', 'left', 'bottom']:
        ax2.spines[sp].set_visible(False)

    _panel_label(ax, '(b)')


# ---------------------------------------------------------------------------
# Panel (c): GLOF trigger mechanism donut
# Percentage labels are shown inside the donut only for slices >= 5%;
# smaller slices get their percentage in the legend instead, eliminating
# text overlap on narrow wedges.
# ---------------------------------------------------------------------------
def plot_panel_c(ax, glofs):
    trig_col = None
    for candidate in ['trigger', 'glof_trigger', 'dam_type']:
        if candidate in glofs.columns:
            trig_col = candidate
            break

    if trig_col is None:
        ax.text(0.5, 0.5, 'Trigger column not found',
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        _panel_label(ax, '(c)', x=-0.05, y=1.05)
        return

    raw_counts = glofs[trig_col].value_counts()
    display_counts = {}
    for raw_key, cnt in raw_counts.items():
        lbl = TRIG_LABEL_MAP.get(raw_key, str(raw_key).replace('_', ' ').title())
        display_counts[lbl] = display_counts.get(lbl, 0) + cnt

    # Sort largest first so the donut reads clockwise from dominant trigger
    items  = sorted(display_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [it[0] for it in items]
    sizes  = [it[1] for it in items]
    total  = sum(sizes)

    tab10  = plt.cm.get_cmap('tab10')
    colors = [tab10(i % 10) for i in range(len(labels))]

    # Show percent inside wedge only for slices >= 5%; others: blank
    def _autopct(pct):
        return f'{pct:.0f}%' if pct >= 5.0 else ''

    wedges, _texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=_autopct,
        pctdistance=0.75,
        startangle=90,
        wedgeprops=dict(width=0.50, edgecolor='white', linewidth=2),
    )
    for i, at in enumerate(autotexts):
        if at.get_text():
            at.set_fontsize(8)
            at.set_fontweight('bold')
            # Choose contrasting text colour based on wedge luminance
            r, g, b, _ = colors[i]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            at.set_color('white' if lum < 0.55 else '#222')

    # Center total label
    ax.text(0, 0, f'{total}\nGLOFs', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#333')

    ax.set_aspect('equal')
    ax.set_title('GLOF trigger mechanisms', fontsize=10, pad=6)

    # Legend: always show percentage here (critical for small slices)
    legend_handles = [
        mpatches.Patch(
            color=colors[i],
            label=f'{labels[i]}  ({sizes[i] / total * 100:.0f}%)',
        )
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.5, -0.12),
        loc='upper center',
        fontsize=7.5,
        framealpha=0.95,
        edgecolor='#ccc',
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        handletextpad=0.5,
    )
    _panel_label(ax, '(c)', x=-0.05, y=1.05)


# ---------------------------------------------------------------------------
# Panel (d): Historical GLOF timeline 1932-2021
# The 1941 Palcacocha annotation position is computed from actual data so
# it never overlaps adjacent points or goes out of bounds.
# ---------------------------------------------------------------------------
def plot_panel_d(ax, glofs):
    df = glofs.copy()
    df = df.dropna(subset=['year_int'])
    df['year_int'] = df['year_int'].astype(int)

    vol_col = None
    for candidate in ['volume_released_m3', 'volume_m3', 'glof_volume_m3',
                       'estimated_volume_m3']:
        if candidate in df.columns:
            vol_col = candidate
            break

    vol    = df[vol_col].fillna(1e5).clip(lower=1).values if vol_col else \
             np.full(len(df), 1e5)
    yvals  = np.log10(vol + 1)

    def decade_color(yr):
        if yr < 1970:
            return '#2980b9'
        elif yr < 2000:
            return '#e67e22'
        return '#c0392b'

    pt_colors = [decade_color(y) for y in df['year_int'].values]
    sz_raw    = np.sqrt(np.clip(vol, 1e4, 1e9) / 1e4) * 3
    sz_plot   = np.clip(sz_raw, 20, 200)

    yr_arr = df['year_int'].values
    ax.vlines(yr_arr, 0, yvals, lw=0.7, color='#aaaaaa', alpha=0.6, zorder=2)
    ax.scatter(yr_arr, yvals, s=sz_plot, c=pt_colors,
               alpha=0.85, edgecolors='#333', linewidths=0.5, zorder=5)

    ax.set_xlim(1928, 2024)
    ax.set_xlabel('Year')
    # Use mathtext so log subscript renders correctly with any font
    ax.set_ylabel(r'$\log_{10}$(Volume m$^3$ + 1)')
    ax.set_title('Historical GLOF timeline 1932\u20132021', fontsize=10, pad=4)

    # Compute y-axis limits before placing the annotation
    ax.autoscale(axis='y', tight=False)
    y_lo_ax, y_hi_ax = ax.get_ylim()

    # Annotate 1941 Palcacocha -- place text in the lower-middle area
    # (y ≈ 35-45% of axis height) where the 1930-1960 zone is sparse,
    # with a white bbox so the label reads over any background lines.
    mask_1941 = df['year_int'] == 1941
    if mask_1941.any():
        idx_1941  = np.where(yr_arr == 1941)[0][0]
        yval_1941 = float(yvals[idx_1941])
        txt_x = 1952
        txt_y = y_lo_ax + (y_hi_ax - y_lo_ax) * 0.40   # 40% up the axis
        txt_y = max(txt_y, y_lo_ax + 0.6)
        ax.annotate(
            '1941 Palcacocha\n(~5,000 deaths)',
            xy=(1941, yval_1941),
            xytext=(txt_x, txt_y),
            fontsize=7,
            ha='center',
            color='#222',
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='white',
                edgecolor='#999',
                alpha=0.92,
                lw=0.7,
            ),
            arrowprops=dict(arrowstyle='->', lw=0.9, color='#555'),
        )

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#555')
    ax.tick_params(direction='out', length=3, width=0.6, colors='#333')
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(axis='x', rotation=45)

    decade_patches = [
        mpatches.Patch(color='#2980b9', label='1930s\u20131960s'),
        mpatches.Patch(color='#e67e22', label='1970s\u20131990s'),
        mpatches.Patch(color='#c0392b', label='2000s\u20132021'),
    ]
    ax.legend(handles=decade_patches, fontsize=7.5, loc='upper left',
              framealpha=0.85, edgecolor='#ccc', handlelength=1.2)

    _panel_label(ax, '(d)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate():
    plt.rcParams.update(IJACSA_STYLE)
    print("Loading data...")
    lakes, glofs, df_csv = load_data()
    print(f"  Lakes: {len(lakes):,}  |  GLOFs: {len(glofs):,}")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(IEEE_FULL, 10.5),
        gridspec_kw=dict(hspace=0.62, wspace=0.46),
    )
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    print("  Plotting panel (a)...")
    plot_panel_a(ax_a, lakes, df_csv)

    print("  Plotting panel (b)...")
    plot_panel_b(ax_b, lakes)

    print("  Plotting panel (c)...")
    plot_panel_c(ax_c, glofs)

    print("  Plotting panel (d)...")
    plot_panel_d(ax_d, glofs)

    for ext, kw in [('.pdf', {}), ('.png', {'dpi': DPI_PNG})]:
        out = FIG_PUB / f'fig2_inventory_glof_context{ext}'
        fig.savefig(out, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f"  Saved: {out.name}  ({out.stat().st_size / 1024:.1f} KB)")

    plt.close(fig)
    print("  Figure 2 complete.")


if __name__ == '__main__':
    generate()
