#!/usr/bin/env python3
"""
scripts/generate_fig07.py
--------------------------
Figure 7 — Supplementary Validation: Score Distributions, Calibration,
            and Prospective Analysis.

Three-panel figure:
  (a) KDE score distributions GLOF vs non-GLOF + KS annotation
  (b) Reliability diagram (calibration curve) — BRF OOF scores
  (c) Geographic distribution of top-50 highest-scoring non-GLOF lakes

Outputs:
  figures/publication/fig7_supplementary_validation.pdf / .png

Run from project root:
    glof/Scripts/python scripts/generate_fig07.py
"""

import json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / 'models'
DATA_DIR   = ROOT / 'data' / 'processed'
FIG_PUB    = ROOT / 'figures' / 'publication'
FIG_PUB.mkdir(parents=True, exist_ok=True)

DPI_PNG   = 600
IEEE_FULL = 7.16

C_GLOF = '#EE3377'
C_MAIN = '#0077BB'
C_LIFT = '#EE7733'

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


def _clean_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_linewidth(0.7)
        ax.spines[sp].set_color('#555')
    ax.tick_params(direction='out', length=3, width=0.6, colors='#333')


def _grid_y(ax):
    ax.yaxis.grid(True, color='#eeeeee', lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax, label, x=0.03, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', color='#111')


def generate():
    print("Figure 7: Supplementary Validation")

    # Load data
    with open(MODELS_DIR / 'oof_predictions.json') as f:
        oof = json.load(f)
    with open(MODELS_DIR / 'extra_validation.json') as f:
        ev = json.load(f)

    y_true      = np.array(oof['y_true'], dtype=int)
    y_score_brf = np.array(oof['y_score_brf'], dtype=float)

    df_csv = pd.read_csv(DATA_DIR / 'labeled' / 'training_data.csv')
    df_csv['y_score_oof'] = y_score_brf

    scores_pos = y_score_brf[y_true == 1]
    scores_neg = y_score_brf[y_true == 0]
    prev = y_true.mean()

    ks_d = ev['ks_test']['ks_statistic']
    ks_p = ev['ks_test']['ks_p_value']
    calib = ev['calibration']
    prosp = ev['prospective_validation']

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(IEEE_FULL, 9.4), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.60, wspace=0.50,
                            height_ratios=[1.0, 1.0])
    ax_kde  = fig.add_subplot(gs[0, 0])   # (a) KDE score distributions
    ax_cal  = fig.add_subplot(gs[0, 1])   # (b) Reliability diagram
    ax_geo  = fig.add_subplot(gs[1, :])   # (c) Geographic breakdown (full width)

    # ── (a) KDE score distributions + KS annotation ──────────────────────────
    x_grid = np.linspace(0, 1, 400)

    if len(scores_neg) >= 2:
        kde_neg = gaussian_kde(scores_neg, bw_method=0.08)
        y_neg   = kde_neg(x_grid)
        ax_kde.fill_between(x_grid, y_neg, alpha=0.25, color=C_MAIN, zorder=2)
        ax_kde.plot(x_grid, y_neg, color=C_MAIN, lw=2.0, label='Non-GLOF  (n=12,572)', zorder=3)

    if len(scores_pos) >= 2:
        kde_pos = gaussian_kde(scores_pos, bw_method=0.18)
        y_pos   = kde_pos(x_grid)
        ax_kde.fill_between(x_grid, y_pos, alpha=0.55, color=C_GLOF, zorder=4)
        ax_kde.plot(x_grid, y_pos, color=C_GLOF, lw=2.5, label='GLOF-source  (n=16)', zorder=5)

    # Rug for GLOF
    max_y = max(y_neg.max(), y_pos.max()) if len(scores_pos)>=2 else y_neg.max()
    rug_y = -0.02 * max_y
    ax_kde.scatter(scores_pos, np.full(len(scores_pos), rug_y),
                   marker='|', s=120, color=C_GLOF, linewidths=2.0,
                   clip_on=False, zorder=8)

    # Threshold line
    ax_kde.axvline(0.304, color='#444', lw=1.3, ls='--', alpha=0.8,
                   label='t = 0.304 (Youden)')

    # KS stats as a subtitle line — always visible, never overlaps curves
    p_str = f'p = {ks_p:.2e}' if ks_p < 0.001 else f'p = {ks_p:.4f}'

    ax_kde.set_xlabel('OOF susceptibility score', fontsize=9)
    ax_kde.set_ylabel('Density', fontsize=9)
    ax_kde.set_title(
        f'Score distributions (KS test)\nD = {ks_d:.3f},  {p_str}',
        fontsize=10, linespacing=1.5)
    ax_kde.set_xlim(-0.02, 1.02)
    ax_kde.legend(fontsize=7.5, framealpha=0.9, edgecolor='#ddd',
                  loc='upper left')
    _clean_ax(ax_kde)
    _grid_y(ax_kde)
    _panel_label(ax_kde, '(a)')

    # ── (b) Reliability diagram — raw vs isotonic calibrated ─────────────────
    # Load calibrated scores if available (from compute_calibration.py)
    calib_path = MODELS_DIR / 'calibrated_oof.json'
    has_calib  = calib_path.exists()
    if has_calib:
        with open(calib_path) as _f:
            calib_data = json.load(_f)
        _rel_raw = calib_data['reliability'].get('raw_10', {})
        _rel_iso = calib_data['reliability'].get('isotonic_10', {})
        bss_raw  = calib_data['metrics']['raw']['bss']
        bss_iso  = calib_data['metrics']['isotonic']['bss']
        mp_raw   = _rel_raw.get('mean_predicted_value', [])
        fp_raw   = _rel_raw.get('fraction_of_positives', [])
        mp_iso   = _rel_iso.get('mean_predicted_value', [])
        fp_iso   = _rel_iso.get('fraction_of_positives', [])
    else:
        # Fallback to extra_validation.json
        bins10   = calib['reliability_data'].get('bins_10', {})
        mp_raw   = bins10.get('mean_predicted_value', [])
        fp_raw   = bins10.get('fraction_of_positives', [])
        mp_iso   = []
        fp_iso   = []
        bss_raw  = -139.8
        bss_iso  = None

    # Perfect calibration diagonal
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=0.9, alpha=0.5,
                label='Perfect calibration', zorder=1)

    # Raw BRF OOF curve
    if mp_raw and fp_raw:
        ax_cal.plot(mp_raw, fp_raw, 'o-',
                    color=C_MAIN, lw=2.0, markersize=5,
                    label=f'BRF raw  (BSS={bss_raw:.2f})', zorder=5)
        ax_cal.fill_between(mp_raw, fp_raw, alpha=0.12, color=C_MAIN)

    # Isotonic calibrated curve
    if mp_iso and fp_iso:
        ax_cal.plot(mp_iso, fp_iso, 's-',
                    color=C_GLOF, lw=2.0, markersize=5,
                    label=f'Isotonic calibrated  (BSS={bss_iso:.2f})', zorder=6)
        ax_cal.fill_between(mp_iso, fp_iso, alpha=0.12, color=C_GLOF)

    # Prevalence baseline
    ax_cal.axhline(prev, color='#aaa', lw=1.0, ls=':',
                   label=f'Prevalence ({prev:.5f})')

    # Annotation — moved to lower-right (plot is mostly empty there)
    _ann_txt = ('Isotonic calibration\nmaps scores to\noperational '
                'probabilities\n(AUC preserved)')
    ax_cal.text(0.97, 0.52, _ann_txt,
                transform=ax_cal.transAxes,
                fontsize=6.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#f0fff0',
                          edgecolor='#27ae60', alpha=0.95, linewidth=0.7))

    ax_cal.set_xlabel('Mean predicted value (bin)', fontsize=9)
    ax_cal.set_ylabel('Fraction of positives', fontsize=9)
    ax_cal.set_title('Reliability diagram: raw vs. calibrated', fontsize=10)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(bottom=0)
    # Legend at upper-left — no longer conflicts with annotation
    ax_cal.legend(fontsize=6.5, framealpha=0.9, edgecolor='#ddd',
                  loc='upper left')
    _clean_ax(ax_cal)
    _grid_y(ax_cal)
    _panel_label(ax_cal, '(b)')

    # ── (c) Geographic distribution of top-50 non-GLOF lakes ─────────────────
    top100_area = prosp.get('top100_area_breakdown', {})
    # Sort by count descending
    sorted_areas = sorted(top100_area.items(), key=lambda x: -x[1])
    labels = [a for a, _ in sorted_areas]
    counts = [c for _, c in sorted_areas]

    # Map labels back to original area keys for colors
    label_to_key = {v: k for k, v in AREA_LABELS.items()}
    bar_colors = [AREA_COLORS.get(label_to_key.get(l, ''), '#888') for l in labels]

    bars = ax_geo.bar(range(len(labels)), counts,
                      color=bar_colors, alpha=0.87,
                      edgecolor='white', width=0.65, zorder=3)

    # Label each bar
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax_geo.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        str(cnt),
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#333')

    # Highlight high-risk areas
    high_risk_labels = {'C. Blanca (PE)', 'C. Huayhuash (PE)', 'C. Raura (PE)'}
    for i, (lbl, bar) in enumerate(zip(labels, bars)):
        if lbl in high_risk_labels:
            bar.set_edgecolor('#c0392b')
            bar.set_linewidth(1.5)

    n_hr = prosp.get('top100_high_risk_fraction', 0) * 100
    ax_geo.text(0.98, 0.95,
                f'{n_hr:.0f}% of top-100\nfrom high-risk\ncordilleras\n(red border)',
                transform=ax_geo.transAxes,
                fontsize=7.5, va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef0f0',
                          edgecolor='#c0392b', alpha=0.9))

    ax_geo.set_xticks(range(len(labels)))
    ax_geo.set_xticklabels(labels, rotation=38, ha='right', fontsize=7.5)
    ax_geo.set_ylabel('Number of lakes in top-100', fontsize=9)
    ax_geo.set_title(
        'Geographic distribution of top-100 highest-scoring non-GLOF lakes '
        '(prospective validation)',
        fontsize=10)
    _clean_ax(ax_geo)
    _grid_y(ax_geo)
    _panel_label(ax_geo, '(c)')

    # ── Export ────────────────────────────────────────────────────────────────
    out_pdf = FIG_PUB / 'fig7_supplementary_validation.pdf'
    out_png = FIG_PUB / 'fig7_supplementary_validation.png'
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png, dpi=DPI_PNG, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    for p in [out_pdf, out_png]:
        if p.exists():
            print(f"  {p.name:55s}  {p.stat().st_size/1024:>8.1f} KB")
    print("  Figure 7 complete.")


if __name__ == '__main__':
    generate()
