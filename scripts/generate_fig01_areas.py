#!/usr/bin/env python3
"""
scripts/generate_fig01_areas.py
--------------------------------
Individual zoom maps — one figure per study area (10 total).

Each figure:
  • CartoDB Voyager basemap (real terrain, city names, roads)
  • Lake scatter coloured by susceptibility — larger, visible markers
  • Historical GLOF events: gold stars labelled with lake name + year
  • Colorbar placed OUTSIDE the map (right side, proper axes)
  • Tight zoom centred on the actual lake/GLOF cluster

Outputs in  figures/publication/areas/:
  fig_area_ecuador.pdf / .png
  fig_area_peru_blanca.pdf / .png
  ... (10 total)

Run from project root:
    glof/Scripts/python scripts/generate_fig01_areas.py
"""

import json, warnings, sys
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
import joblib
from pathlib import Path

try:
    import contextily as ctx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False
    print("[warn] contextily not installed — using flat basemap")

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
FIG_AREAS  = ROOT / 'figures' / 'publication' / 'areas'
FIG_AREAS.mkdir(parents=True, exist_ok=True)

_NE_SHP = (ROOT / 'glof' / 'Lib' / 'site-packages' / 'pyogrio'
           / 'tests' / 'fixtures' / 'naturalearth_lowres'
           / 'naturalearth_lowres.shp')

DPI_PNG = 600

CMAP_SUSC = LinearSegmentedColormap.from_list(
    'glof_susc',
    [(0.00, '#2171b5'), (0.25, '#74c4e4'),
     (0.50, '#ffffd9'), (0.75, '#f4701e'),
     (1.00, '#c0392b')], N=256)
NORM_SUSC = mcolors.Normalize(0, 1)

TF     = Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
TF_INV = Transformer.from_crs('epsg:3857', 'epsg:4326', always_xy=True)
def ll2xy(lon, lat): return TF.transform(lon, lat)
def xy2ll(x, y):     return TF_INV.transform(x, y)

GLOF_FACE = '#FFD700'
GLOF_EDGE = '#7b0000'

# ── Study areas — (key, full title, output slug) ─────────────────────────────
AREAS = [
    ('ecuador_antisana',       'Ecuador — Antisana',              'fig_area_ecuador'),
    ('cordillera_blanca',      'Perú — Cordillera Blanca',        'fig_area_peru_blanca'),
    ('cordillera_huayhuash',   'Perú — Cordillera Huayhuash',     'fig_area_peru_huayhuash'),
    ('cordillera_raura',       'Perú — Cordillera Raura',         'fig_area_peru_raura'),
    ('cordillera_central',     'Perú — Cordillera Central',       'fig_area_peru_central'),
    ('cordillera_urubamba',    'Perú — Cordillera Urubamba',      'fig_area_peru_urubamba'),
    ('cordillera_vilcanota',   'Perú — Cordillera Vilcanota',     'fig_area_peru_vilcanota'),
    ('cordillera_huanzo',      'Perú — Cordillera Huanzo',        'fig_area_peru_huanzo'),
    ('bolivia_cordillera_real','Bolivia — Cordillera Real',       'fig_area_bolivia'),
    ('chile_andes_centrales',  'Chile — Andes Centrales',         'fig_area_chile'),
    ('carabaya',               'Perú — Cordillera Carabaya',      'fig_area_peru_carabaya'),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_world():
    if _NE_SHP.exists():
        return gpd.read_file(_NE_SHP).to_crs(epsg=3857)
    return None


def load_data():
    lakes    = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')
    glofs_raw= gpd.read_file(DATA_DIR / 'labeled' / 'historical_glofs.gpkg')

    df = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
    with open(MODELS_DIR / 'feature_names.json') as fh:
        feat_cols = json.load(fh)

    excl = {'lake_id','had_glof','glof_label','country','cordillera',
            'geometry','year','source','area_name','scene_date',
            'glof_volume_m3','glof_match_dist_m',
            'glof_date','glof_trigger','glof_lake_name'}
    base = [c for c in df.columns
            if c not in excl and df[c].dtype in
               ['float64','float32','int64','int32']]
    X = df[base].fillna(df[base].median()).fillna(0)
    for col in ['area_m2','perimeter_m','volume_m3','equivalent_diameter_m',
                'potential_energy','depth_m','dam_height',
                'area_2017_est','area_2025_est','total_growth_m2']:
        if col in X.columns:
            X[f'log_{col}'] = np.log1p(X[col].clip(lower=0))
    if 'total_growth_m2' in X.columns and 'area_2017_est' in X.columns:
        X['rel_growth'] = X['total_growth_m2'] / (X['area_2017_est'] + 1)
    if 'slope_mean' in X.columns and 'log_area_m2' in X.columns:
        X['slope_x_logarea'] = X['slope_mean'] * X['log_area_m2']
    if 'dist_glacier_m' in df.columns:
        X['no_glacier'] = df['dist_glacier_m'].isna().astype(int)

    avail = [c for c in feat_cols if c in X.columns]
    mdl   = joblib.load(MODELS_DIR / 'best_model.joblib')
    lakes['y_prob'] = mdl.predict_proba(X[avail])[:, 1]

    lakes_p = lakes.to_crs(epsg=3857)
    lakes_p['cx'] = lakes_p.geometry.centroid.x
    lakes_p['cy'] = lakes_p.geometry.centroid.y

    glofs_wm = gpd.GeoDataFrame(
        glofs_raw,
        geometry=gpd.points_from_xy(glofs_raw['longitude'], glofs_raw['latitude']),
        crs='EPSG:4326').to_crs(epsg=3857)
    glofs_wm['label']    = (glofs_raw['lake'].str.strip() + '\n('
                            + glofs_raw['year'].astype(str) + ')')
    glofs_wm['area_key'] = glofs_raw['area'].str.strip()

    print(f"  Loaded: {len(lakes_p):,} lakes | {len(glofs_wm)} hist. GLOFs")
    return lakes_p, glofs_wm


# ── Tight extent ──────────────────────────────────────────────────────────────

def _tight_extent(sub_lakes, sub_glofs, pad=0.30, min_half=22_000):
    """
    Compute tight (xlim, ylim) centred on the actual data cluster.
    pad   : fractional margin on each side of the cluster extent
    min_half : minimum half-span in metres (floor for very compact clusters)
    """
    xs = list(sub_lakes['cx']) + list(sub_glofs.geometry.x)
    ys = list(sub_lakes['cy']) + list(sub_glofs.geometry.y)

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = (x0+x1)/2, (y0+y1)/2

    hw = max((x1-x0)/2 * (1+pad), min_half)
    hh = max((y1-y0)/2 * (1+pad), min_half)

    # Keep aspect ratio within 1.4× so the map isn't too skewed
    if hh > hw * 1.4: hw = hh / 1.4
    if hw > hh * 1.4: hh = hw / 1.4

    return (cx-hw, cx+hw), (cy-hh, cy+hh)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_bar(ax, xlim, ylim):
    span = xlim[1] - xlim[0]
    bar  = 5_000  if span < 50_000  else \
           10_000 if span < 100_000 else \
           25_000 if span < 250_000 else 50_000
    lab  = f'{bar//1000} km'
    x0   = xlim[0] + span * 0.04
    y0   = ylim[0] + (ylim[1]-ylim[0]) * 0.04
    ax.plot([x0, x0+bar], [y0, y0], lw=4, color='#111',
            solid_capstyle='butt', zorder=16)
    ax.text(x0+bar/2, y0+(ylim[1]-ylim[0])*0.015, lab,
            ha='center', va='bottom', fontsize=7.5,
            color='#111', fontweight='bold', zorder=16)


def _north_arrow(ax, xlim, ylim):
    span = xlim[1] - xlim[0]
    hspan= ylim[1] - ylim[0]
    na_x = xlim[0] + span * 0.09
    na_y = ylim[0] + hspan * 0.11
    dy   = hspan * 0.05
    ax.annotate('N', xy=(na_x, na_y+dy), xytext=(na_x, na_y),
                fontsize=12, fontweight='bold', ha='center', va='bottom',
                color='#111',
                arrowprops=dict(arrowstyle='->', color='#111', lw=1.5),
                zorder=16)


def _glof_labels(ax, sub_glofs, xlim, ylim):
    if sub_glofs.empty: return
    df = sub_glofs.copy()
    df['gx'] = df.geometry.x
    df['gy'] = df.geometry.y
    df = df.sort_values('gx').reset_index(drop=True)

    offsets = [(28, 20), (-28, 20), (28, -24), (-28, -24),
               (38, 6),  (-38, 6),  (32, -12), (-32, -12),
               (22, 28), (-22, 28)]

    for i, row in df.iterrows():
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(
            row['label'],
            xy=(row['gx'], row['gy']),
            xytext=(ox, oy), textcoords='offset points',
            fontsize=6.5, color='#5a0000', fontstyle='italic',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.28', fc='#fff8f0',
                      ec='#8b1a1a', alpha=0.92, lw=0.65),
            arrowprops=dict(arrowstyle='->', color='#8b1a1a',
                            lw=0.6, mutation_scale=6.5),
            zorder=13)


# ── Per-area figure ───────────────────────────────────────────────────────────

def draw_area(area_key, title, slug, lakes_p, glofs_wm, world):
    sub_lakes = lakes_p[lakes_p['area_name'] == area_key].copy()
    sub_glofs = glofs_wm[glofs_wm['area_key'] == area_key].copy()

    if sub_lakes.empty and sub_glofs.empty:
        print(f"  [skip] {area_key} — no data");  return

    xlim, ylim = _tight_extent(sub_lakes, sub_glofs)

    # ── Figure with room for colorbar on the right ────────────────────────────
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(9, 9), facecolor='white')
    # Main map axes — leave right margin for colorbar
    ax = fig.add_axes([0.04, 0.06, 0.82, 0.86])

    ax.set_xlim(*xlim);  ax.set_ylim(*ylim)
    ax.set_facecolor('#d6eaf8')
    ax.set_aspect('equal', adjustable='box')

    for sp in ax.spines.values():
        sp.set_linewidth(0.6);  sp.set_color('#666666')
    ax.tick_params(labelsize=7.5, length=3, width=0.5)

    # ── CartoDB Voyager basemap ───────────────────────────────────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:3857',
                            source=ctx.providers.CartoDB.Voyager,
                            zoom='auto', attribution_size=5,
                            zorder=0, reset_extent=False)
        except Exception as e:
            print(f"    [warn] tile failed: {e}")
            if world is not None:
                world.plot(ax=ax, color='#eae6e0', edgecolor='#aaa',
                           linewidth=0.3, zorder=0)
    elif world is not None:
        world.plot(ax=ax, color='#eae6e0', edgecolor='#aaa',
                   linewidth=0.3, zorder=0)

    # Restore extent after tile download (tiles may shift xlim/ylim)
    ax.set_xlim(*xlim);  ax.set_ylim(*ylim)

    # ── Lake scatter — larger, more visible ───────────────────────────────────
    if len(sub_lakes) > 0:
        order = sub_lakes['y_prob'].argsort().values
        ax.scatter(
            sub_lakes['cx'].values[order],
            sub_lakes['cy'].values[order],
            c=sub_lakes['y_prob'].values[order],
            cmap=CMAP_SUSC, norm=NORM_SUSC,
            s=np.clip(
                np.log10(sub_lakes['area_m2'].fillna(1000).values[order]/300+1)*14,
                3, 90),
            alpha=0.88, linewidths=0.3, edgecolors='#444444',
            rasterized=False, zorder=5)

    # ── Historical GLOF stars + labels ───────────────────────────────────────
    if len(sub_glofs) > 0:
        ax.scatter(
            sub_glofs.geometry.x, sub_glofs.geometry.y,
            s=160, marker='*', c=GLOF_FACE,
            edgecolors=GLOF_EDGE, linewidths=0.9, zorder=11)
        _glof_labels(ax, sub_glofs, xlim, ylim)

    # ── Area centroid ring ───────────────────────────────────────────────────
    if len(sub_lakes) > 0:
        ax.plot(sub_lakes['cx'].mean(), sub_lakes['cy'].mean(),
                marker='o', ms=8, mfc='none',
                mec='#111111', mew=1.4, zorder=12, ls='none')

    # ── Lat/lon tick labels ───────────────────────────────────────────────────
    lon0, lat0 = xy2ll(xlim[0], ylim[0])
    lon1, lat1 = xy2ll(xlim[1], ylim[1])
    lons = np.linspace(lon0, lon1, 4)[1:-1]
    lats = np.linspace(lat0, lat1, 4)[1:-1]
    ax.set_xticks([ll2xy(lo, (lat0+lat1)/2)[0] for lo in lons])
    ax.set_xticklabels([f'{abs(lo):.2f}°{"W" if lo<0 else "E"}' for lo in lons],
                       fontsize=7.5)
    ax.set_yticks([ll2xy((lon0+lon1)/2, la)[1] for la in lats])
    ax.set_yticklabels([f'{abs(la):.2f}°{"S" if la<0 else "N"}' for la in lats],
                       fontsize=7.5)

    # ── Scale bar & north arrow ───────────────────────────────────────────────
    _scale_bar(ax, xlim, ylim)
    _north_arrow(ax, xlim, ylim)

    # ── SA locator inset (top-right corner, inside map) ───────────────────────
    ax_loc = inset_axes(ax, width='22%', height='26%', loc='upper right',
                        borderpad=0.5)
    ax_loc.set_facecolor('#cde8f7')
    if world is not None:
        world.plot(ax=ax_loc, color='#dedad5', edgecolor='#999',
                   linewidth=0.25, zorder=1)
        sa_names = ['Peru','Bolivia','Ecuador','Chile','Argentina',
                    'Brazil','Colombia','Venezuela','Paraguay','Uruguay']
        sa_w = world[world['name'].isin(sa_names)]
        if not sa_w.empty:
            bb = sa_w.total_bounds
            # Zoom into western SA only (exclude Atlantic coast)
            ax_loc.set_xlim(bb[0] - 300_000, bb[0] + (bb[2]-bb[0])*0.62)
            ax_loc.set_ylim(bb[1] - 200_000, bb[3] + 200_000)
        ax_loc.add_patch(mpatches.Rectangle(
            (xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
            lw=2.2, ec='#c0392b', fc='#c0392b', alpha=0.45, zorder=5))
        ax_loc.add_patch(mpatches.Rectangle(
            (xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
            lw=2.2, ec='#c0392b', fc='none', zorder=6))
    ax_loc.set_xticks([]); ax_loc.set_yticks([])
    for sp in ax_loc.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color('#555')

    # ── Legend (inside map, bottom-right) ────────────────────────────────────
    legend_h = [
        Line2D([0],[0], marker='o', color='w', markersize=8,
               markerfacecolor='#2171b5', markeredgecolor='#444',
               markeredgewidth=0.4, label='Low susc. (≤ 0.25)'),
        Line2D([0],[0], marker='o', color='w', markersize=8,
               markerfacecolor='#f4701e', markeredgecolor='#444',
               markeredgewidth=0.4, label='High susc. (≥ 0.75)'),
        Line2D([0],[0], marker='*', color='w', markersize=13,
               markerfacecolor=GLOF_FACE, markeredgecolor=GLOF_EDGE,
               label='Historical GLOF'),
        Line2D([0],[0], marker='o', color='w', markersize=8,
               markerfacecolor='none', markeredgecolor='#111',
               markeredgewidth=1.4, label='Area centroid'),
    ]
    ax.legend(handles=legend_h, loc='lower right',
              fontsize=7.5, frameon=True, framealpha=0.93,
              edgecolor='#bbbbbb', ncol=1,
              title='Legend', title_fontsize=8,
              borderpad=0.8, handlelength=1.5, handletextpad=0.5)

    # ── Colorbar — OUTSIDE map, right side ────────────────────────────────────
    # Use make_axes_locatable to attach a proper colorbar axis to the right of ax
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.14)

    sm = plt.cm.ScalarMappable(cmap=CMAP_SUSC, norm=NORM_SUSC)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('GLOF Susceptibility Score', fontsize=8.5, labelpad=6)
    cbar.ax.tick_params(labelsize=8, length=3, pad=3)
    cbar.outline.set_linewidth(0.6)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_yticklabels(
        ['0.00\n(Low)', '0.25', '0.50', '0.75', '1.00\n(High)'], fontsize=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_lakes = len(sub_lakes)
    n_glofs = len(sub_glofs)
    fig.suptitle(f'{title}   —   {n_lakes:,} glacier lakes  |  {n_glofs} hist. GLOFs',
                 fontsize=10.5, fontweight='bold', y=0.99, color='#111111')

    # ── Export ────────────────────────────────────────────────────────────────
    for ext, kw in [('.pdf', {}), ('.png', {'dpi': DPI_PNG})]:
        out = FIG_AREAS / f'{slug}{ext}'
        fig.savefig(out, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f"  {out.name:55s}  {out.stat().st_size/1024:>8.1f} KB")

    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def generate():
    print("Figure 1 areas — individual study-area zoom maps")
    world = load_world()
    lakes_p, glofs_wm = load_data()

    for area_key, title, slug in AREAS:
        print(f"\n  → {title}")
        draw_area(area_key, title, slug, lakes_p, glofs_wm, world)

    print(f"\n  All maps saved to  figures/publication/areas/")


if __name__ == '__main__':
    generate()
