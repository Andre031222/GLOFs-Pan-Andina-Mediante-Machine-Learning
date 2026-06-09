#!/usr/bin/env python3
"""
scripts/generate_fig01.py
--------------------------
Figure 1 — Pan-Andean GLOF Susceptibility Map (overview).

Features:
  • CartoDB Voyager basemap via contextily (real terrain colours + city names)
  • All 10 cordilleras: fine-arrow labels, no text collision
  • Colorbar placed OUTSIDE the map (proper axis, right side)
  • SA locator inset (top-right corner inside map)
  • Scale bar, north arrow, legend

Outputs:
  figures/publication/fig1_susceptibility_map.pdf
  figures/publication/fig1_susceptibility_map.png  (600 DPI)

Run from project root:
    glof/Scripts/python scripts/generate_fig01.py
"""

import json, warnings, sys, math
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

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
FIG_PUB    = ROOT / 'figures' / 'publication'
FIG_PUB.mkdir(parents=True, exist_ok=True)

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

TF = Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
def ll2xy(lon, lat): return TF.transform(lon, lat)

SA_FILL = {
    'Ecuador': '#d5e8d4', 'Peru': '#dae8d4',
    'Bolivia': '#d5e3e8', 'Chile': '#e8dfd5',
    'Argentina': '#ece8d5', 'Brazil': '#e8e0d5',
    'Colombia': '#dde8d8', 'Paraguay': '#e8e3d5',
    'Venezuela': '#d8e8e2', 'Uruguay': '#e8e3d5',
}

# Main-map annotation positions (label lon/lat — kept close to centroids)
ANNOTS = [
    ('ecuador_antisana',       'Antisana\n(Ecuador)',       -79.0,  -0.8, 'right'),
    ('cordillera_blanca',      'Cord. Blanca',              -79.0,  -7.5, 'right'),
    ('cordillera_huayhuash',   'Cord. Huayhuash',           -79.0,  -9.8, 'right'),
    ('cordillera_raura',       'Cord. Raura',               -79.0, -11.8, 'right'),
    ('cordillera_central',     'Cord. Central',             -79.0, -13.5, 'right'),
    ('cordillera_urubamba',    'Cord. Urubamba',            -67.5, -11.5, 'left'),
    ('cordillera_vilcanota',   'Cord. Vilcanota',           -67.5, -13.5, 'left'),
    ('cordillera_huanzo',      'Cord. Huanzo',              -67.5, -15.0, 'left'),
    ('bolivia_cordillera_real','Cord. Real\n(Bolivia)',     -67.5, -17.0, 'left'),
    ('chile_andes_centrales',  'Andes Centrales\n(Chile)',  -67.5, -33.5, 'left'),
]


def load_world():
    if _NE_SHP.exists():
        return gpd.read_file(_NE_SHP).to_crs(epsg=3857)
    return None


def load_data():
    lakes = gpd.read_file(DATA_DIR / 'labeled' / 'labeled_lakes.gpkg')
    glofs = gpd.read_file(DATA_DIR / 'labeled' / 'historical_glofs.gpkg')

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

    sa_cent = (lakes_p.groupby('area_name')
                       .agg(cx=('cx','mean'), cy=('cy','mean'))
                       .reset_index())

    glofs_wm = gpd.GeoDataFrame(
        glofs,
        geometry=gpd.points_from_xy(glofs['longitude'], glofs['latitude']),
        crs='EPSG:4326').to_crs(epsg=3857)

    print(f"  Lakes: {len(lakes):,}  GLOF+: {int(lakes['had_glof'].sum())}  "
          f"Hist. GLOFs: {len(glofs_wm)}")
    return lakes_p, glofs_wm, sa_cent


def generate():
    print("Figure 1 – Pan-Andean GLOF Susceptibility Map")
    world = load_world()
    lakes_p, glofs_wm, sa_cent = load_data()
    cent  = {r.area_name: (r.cx, r.cy) for r in sa_cent.itertuples()}

    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 9, 'axes.titlesize': 10,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    # ── Map extent ────────────────────────────────────────────────────────────
    ann_xy = {a[0]: (ll2xy(a[2], a[3]), a[1], a[4]) for a in ANNOTS}
    all_tx = [v[0][0] for v in ann_xy.values()]
    b      = lakes_p.total_bounds
    XLIM   = (min(b[0]-150_000, min(all_tx)-200_000),
              max(b[2]+150_000, max(all_tx)+300_000))
    YLIM   = (b[1]-100_000, b[3]+120_000)

    # ── Figure — main map axes + external colorbar ────────────────────────────
    fig = plt.figure(figsize=(10, 14), facecolor='white')
    # Map axes: tight left/right so colorbar goes outside
    ax = fig.add_axes([0.06, 0.05, 0.82, 0.91])

    ax.set_xlim(*XLIM);  ax.set_ylim(*YLIM)
    ax.set_facecolor('#cde8f7')
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color('#555')

    # ── CartoDB Voyager basemap (more colour, terrain detail) ─────────────────
    if _HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:3857',
                            source=ctx.providers.CartoDB.Voyager,
                            zoom=5, attribution_size=5,
                            zorder=0, reset_extent=False)
            ax.set_xlim(*XLIM);  ax.set_ylim(*YLIM)   # restore after tile
        except Exception as e:
            print(f"  [warn] tile: {e} — using flat basemap")
            _flat_basemap(ax, world)
    else:
        _flat_basemap(ax, world)

    # ── Lake scatter ──────────────────────────────────────────────────────────
    idx = lakes_p['y_prob'].argsort().values
    ax.scatter(
        lakes_p['cx'].values[idx], lakes_p['cy'].values[idx],
        c=lakes_p['y_prob'].values[idx],
        cmap=CMAP_SUSC, norm=NORM_SUSC,
        s=np.clip(np.log10(lakes_p['area_m2'].fillna(1000).values[idx]/800+1)*2.5,
                  0.4, 10),
        alpha=0.82, linewidths=0, rasterized=True, zorder=4)

    # ── Historical GLOF stars ─────────────────────────────────────────────────
    ax.scatter(glofs_wm.geometry.x, glofs_wm.geometry.y,
               s=65, marker='*', c='#FFD700',
               edgecolors='#7b0000', linewidths=0.8, zorder=8)

    # ── Centroid rings ────────────────────────────────────────────────────────
    for name, (cx, cy) in cent.items():
        ax.plot(cx, cy, marker='o', ms=3.5, mfc='white',
                mec='#111', mew=0.85, zorder=9, ls='none')

    # ── Fine-arrow annotations ────────────────────────────────────────────────
    arr_kw = dict(arrowstyle='->', color='#222222', lw=0.65, mutation_scale=7)
    for name, ((tx, ty), label, ha) in ann_xy.items():
        if name not in cent: continue
        px, py = cent[name]
        ax.annotate(
            label, xy=(px, py), xytext=(tx, ty),
            ha=ha, va='center',
            fontsize=7.5, fontstyle='italic', color='#111111',
            arrowprops=arr_kw, zorder=11,
            bbox=dict(boxstyle='round,pad=0.22', fc='white',
                      ec='none', alpha=0.90, lw=0))

    # ── Axis labels & ticks ───────────────────────────────────────────────────
    ax.set_xlabel('Longitude', fontsize=9.5, labelpad=4)
    ax.set_ylabel('Latitude',  fontsize=9.5, labelpad=4)

    xt = np.linspace(XLIM[0], XLIM[1], 6)
    ax.set_xticks(xt)
    ax.set_xticklabels(
        [f'{abs(x/111_319.49):.0f}°W' for x in xt], fontsize=8)

    import math as _m
    yt_degs = [d for d in range(0, -40, -5)
               if YLIM[0] <= _m.log(_m.tan(_m.radians(d+90)/2))*6_378_137 <= YLIM[1]]
    yt_m    = [_m.log(_m.tan(_m.radians(d+90)/2))*6_378_137 for d in yt_degs]
    ax.set_yticks(yt_m)
    ax.set_yticklabels(
        [f'{abs(d)}°S' if d < 0 else '0°' for d in yt_degs], fontsize=8)

    # ── Scale bar ─────────────────────────────────────────────────────────────
    sb_x0 = XLIM[0] + (XLIM[1]-XLIM[0])*0.03
    sb_y0 = YLIM[0] + (YLIM[1]-YLIM[0])*0.022
    ax.plot([sb_x0, sb_x0+500_000], [sb_y0, sb_y0],
            lw=4, color='#111', solid_capstyle='butt', zorder=12)
    ax.text(sb_x0+250_000, sb_y0+(YLIM[1]-YLIM[0])*0.012,
            '500 km', ha='center', va='bottom',
            fontsize=8, color='#111', fontweight='bold')

    # ── North arrow ───────────────────────────────────────────────────────────
    na_x = XLIM[0] + (XLIM[1]-XLIM[0])*0.065
    na_y = YLIM[0] + (YLIM[1]-YLIM[0])*0.075
    dy   = (YLIM[1]-YLIM[0])*0.030
    ax.annotate('N', xy=(na_x, na_y+dy), xytext=(na_x, na_y),
                fontsize=11, fontweight='bold', ha='center', va='bottom',
                color='#111',
                arrowprops=dict(arrowstyle='->', color='#111', lw=1.4),
                zorder=12)

    # ── SA locator inset (top-right, inside map) ──────────────────────────────
    ax_loc = inset_axes(ax, width='20%', height='24%', loc='upper right',
                        borderpad=0.7)
    ax_loc.set_facecolor('#cde8f7')
    if world is not None:
        world.plot(ax=ax_loc, color='#dedad5', edgecolor='#999',
                   linewidth=0.25, zorder=1)
        sa_names = ['Peru','Bolivia','Ecuador','Chile','Argentina',
                    'Brazil','Colombia','Venezuela','Paraguay','Uruguay']
        sa_w = world[world['name'].isin(sa_names)]
        if not sa_w.empty:
            # Zoom into western SA only — exclude Brazil/Atlantic coast
            bb = sa_w.total_bounds
            ax_loc.set_xlim(bb[0] - 300_000, bb[0] + (bb[2]-bb[0])*0.62)
            ax_loc.set_ylim(bb[1] - 200_000, bb[3] + 200_000)
        bx0,by0,bx1,by1 = lakes_p.total_bounds
        ax_loc.add_patch(mpatches.Rectangle(
            (bx0,by0), bx1-bx0, by1-by0,
            lw=1.8, ec='#c0392b', fc='#c0392b', alpha=0.30, zorder=5))
        ax_loc.add_patch(mpatches.Rectangle(
            (bx0,by0), bx1-bx0, by1-by0,
            lw=1.8, ec='#c0392b', fc='none', zorder=6))
    ax_loc.set_xticks([]); ax_loc.set_yticks([])
    for sp in ax_loc.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color('#555')
    ax_loc.set_title('Study region', fontsize=6.5, pad=2, color='#222')

    # ── Legend (inside map, bottom-left) ─────────────────────────────────────
    legend_h = [
        Line2D([0],[0], marker='o', color='w', markersize=5,
               markerfacecolor='#2166ac', markeredgecolor='none',
               label='Low susceptibility (≤ 0.25)'),
        Line2D([0],[0], marker='o', color='w', markersize=9,
               markerfacecolor='#f46d43', markeredgecolor='none',
               label='High susceptibility (≥ 0.75)'),
        Line2D([0],[0], marker='o', color='w', markersize=5,
               markerfacecolor='#888888', markeredgecolor='none',
               label='Point size ∝ log(lake area)'),
        Line2D([0],[0], marker='*', color='w', markersize=12,
               markerfacecolor='#FFD700', markeredgecolor='#7b0000',
               label=f'Historical GLOF  (n={len(glofs_wm)})'),
        Line2D([0],[0], marker='o', color='w', markersize=7,
               markerfacecolor='white', markeredgecolor='#111',
               label='Study area centroid'),
    ]
    ax.legend(handles=legend_h, loc='lower left',
              fontsize=8, frameon=True, framealpha=0.93,
              edgecolor='#cccccc', ncol=1,
              title='Legend', title_fontsize=8.5,
              borderpad=0.9, handlelength=1.5, handletextpad=0.6)

    # ── Colorbar — OUTSIDE map, proper right-side axes ────────────────────────
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3.5%', pad=0.16)

    sm = plt.cm.ScalarMappable(cmap=CMAP_SUSC, norm=NORM_SUSC)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('GLOF Susceptibility Score', fontsize=9, labelpad=6)
    cbar.ax.tick_params(labelsize=8.5, length=3, pad=3)
    cbar.outline.set_linewidth(0.6)
    cbar.set_ticks([0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_yticklabels(
        ['0.00\n(Low)', '0.25', '0.50', '0.75', '1.00\n(High)'], fontsize=8.5)

    # ── Export ────────────────────────────────────────────────────────────────
    for ext, kw in [('.pdf', {}), ('.png', {'dpi': DPI_PNG})]:
        out = FIG_PUB / f'fig1_susceptibility_map{ext}'
        fig.savefig(out, bbox_inches='tight', facecolor='white',
                    edgecolor='none', **kw)
        print(f"  {out.name:52s}  {out.stat().st_size/1024:>8.1f} KB")

    plt.close(fig)
    print("  Figure 1 complete.")


def _flat_basemap(ax, world):
    """Fallback: coloured NE polygons when contextily is unavailable."""
    if world is None: return
    for _, row in world.iterrows():
        nm = row.get('name', '')
        fc = SA_FILL.get(nm, '#eae6e0')
        gpd.GeoDataFrame([row], geometry='geometry', crs=world.crs).plot(
            ax=ax, color=fc, edgecolor='#aaaaaa', linewidth=0.35, zorder=1)


if __name__ == '__main__':
    generate()
