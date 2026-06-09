"""
scripts/rematch_glofs_graduated_buffer.py
------------------------------------------
Re-run the GLOF spatial matching with a **graduated buffer**:

    - Events dated >= 2017 (Sentinel-2 era)   : 5,000 m buffer  (precise coords)
    - Events dated <  2017 (historical)        : 10,000 m buffer (GloFLOD coords
                                                  have documented 1-10 km uncertainty;
                                                  Lützow et al. 2023, §3.2)

Scientific rationale
--------------------
Historical GLOF catalogs (GloFLOD, INAIGEM) frequently record the *downstream
impact zone* or *administrative reporting point* rather than the true source-lake
centroid (Allen et al. 2022; Lützow et al. 2023).  The GloFLOD paper explicitly
documents positional uncertainty of up to ~10 km for pre-satellite events.
Using a larger buffer for historical events is therefore epistemically justified
and does not introduce circular reasoning — the source lake is independently
detected from Sentinel-2 imagery, not inferred from GLOF proximity.

Expected additional positives (conservative estimate)
------------------------------------------------------
  Bolivia Khara_Khota   2016  distance=8,488m   → match at 10km  (+1)
  Bolivia Zongo_1       2009  distance=13,360m  → match at 14km  (not captured at 10km)
  Others (Carabaya Yanavico: 15,598m, etc.)     → not matched at 10km

Run from project root:
    glof/Scripts/python scripts/rematch_glofs_graduated_buffer.py

Outputs
-------
  data/processed/labeled/historical_glofs_rematched.gpkg  — updated match records
  data/processed/labeled/labeled_lakes_v2.gpkg            — lake labels with graduated buffer
  data/processed/labeled/training_data_v2.csv             — updated training CSV
  models/matching_report_v2.txt                            — detailed match report
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / 'data' / 'processed'
MODELS_DIR  = ROOT / 'models'

BUFFER_S2_M       = 5_000   # metres — Sentinel-2 era events (>= 2017)
BUFFER_HIST_M     = 10_000  # metres — historical events (< 2017)
S2_START_YEAR     = 2017

CRS_MATCH = {
    # UTM zone per area for metric distance computation
    'apolobamba':            'EPSG:32719',
    'bolivia_cordillera_real': 'EPSG:32719',
    'bolivia_norte':         'EPSG:32719',
    'carabaya':              'EPSG:32719',
    'chile_andes_centrales': 'EPSG:32719',
    'cordillera_blanca':     'EPSG:32718',
    'cordillera_central':    'EPSG:32718',
    'cordillera_huanzo':     'EPSG:32719',
    'cordillera_huayhuash':  'EPSG:32718',
    'cordillera_raura':      'EPSG:32718',
    'cordillera_urubamba':   'EPSG:32719',
    'cordillera_vilcanota':  'EPSG:32719',
    'ecuador_antisana':      'EPSG:32717',
    'patagonia_norte':       'EPSG:32718',
    'patagonia_sur':         'EPSG:32718',
}
DEFAULT_CRS = 'EPSG:32719'


def load_data():
    lakes_path = DATA_DIR / 'labeled' / 'labeled_lakes.gpkg'
    glofs_path = DATA_DIR / 'labeled' / 'historical_glofs.gpkg'

    if not lakes_path.exists():
        raise FileNotFoundError(f"Lake inventory not found: {lakes_path}")
    if not glofs_path.exists():
        raise FileNotFoundError(f"GLOF inventory not found: {glofs_path}")

    lakes = gpd.read_file(lakes_path)
    glofs = gpd.read_file(glofs_path)
    print(f"  Lakes loaded : {len(lakes):,}  |  GLOF events: {len(glofs)}")
    return lakes, glofs


def graduated_buffer(event_date_str):
    """Return buffer in metres based on event date."""
    try:
        year = int(str(event_date_str)[:4])
    except (ValueError, TypeError):
        return BUFFER_HIST_M
    return BUFFER_S2_M if year >= S2_START_YEAR else BUFFER_HIST_M


def rematch(lakes_gdf, glofs_gdf):
    """
    Spatial matching with graduated buffer.
    Returns updated lakes_gdf with had_glof, glof_* columns reset,
    and a match_log list.
    """
    # Reset GLOF labels
    lakes = lakes_gdf.copy()
    for col in ['had_glof', 'glof_date', 'glof_trigger',
                'glof_volume_m3', 'glof_lake_name', 'glof_match_dist_m']:
        if col in lakes.columns:
            lakes[col] = 0 if col == 'had_glof' else None

    match_log = []
    total_matched = 0
    total_events  = 0

    areas = glofs_gdf['area'].unique()

    for area in sorted(areas):
        area_glofs = glofs_gdf[glofs_gdf['area'] == area].copy()
        area_lakes = lakes[lakes['area_name'] == area].copy()
        total_events += len(area_glofs)

        if len(area_lakes) == 0:
            for _, g in area_glofs.iterrows():
                buf = graduated_buffer(g['date'])
                match_log.append({
                    'area': area, 'lake': g['lake'], 'date': str(g['date']),
                    'buffer_m': buf, 'matched': False,
                    'dist_m': None, 'reason': 'no_lakes_in_area',
                })
            continue

        utm = CRS_MATCH.get(area, DEFAULT_CRS)
        area_lakes_utm = area_lakes.to_crs(utm)
        area_glofs_utm = area_glofs.to_crs(utm)

        for _, g in area_glofs_utm.iterrows():
            buf = graduated_buffer(g['date'])
            dists = area_lakes_utm.geometry.centroid.distance(g.geometry)
            min_d = dists.min()
            closest_idx = dists.idxmin()

            if min_d <= buf:
                # Match found
                orig_idx = closest_idx
                lakes.at[orig_idx, 'had_glof']         = 1
                lakes.at[orig_idx, 'glof_date']        = str(g['date'])
                lakes.at[orig_idx, 'glof_trigger']     = g.get('trigger', None)
                lakes.at[orig_idx, 'glof_volume_m3']   = g.get('volume_released_m3', None)
                lakes.at[orig_idx, 'glof_lake_name']   = g.get('lake', None)
                lakes.at[orig_idx, 'glof_match_dist_m'] = round(min_d)
                total_matched += 1
                match_log.append({
                    'area': area, 'lake': g['lake'], 'date': str(g['date']),
                    'buffer_m': buf, 'matched': True,
                    'dist_m': round(min_d), 'reason': 'within_buffer',
                })
            else:
                match_log.append({
                    'area': area, 'lake': g['lake'], 'date': str(g['date']),
                    'buffer_m': buf, 'matched': False,
                    'dist_m': round(min_d), 'reason': f'dist={min_d:.0f}m > buffer={buf}m',
                })

    n_pos_lakes = int((lakes['had_glof'] == 1).sum())
    return lakes, match_log, total_events, total_matched, n_pos_lakes


def print_report(match_log, total_events, total_matched, n_pos_lakes):
    matched     = [m for m in match_log if m['matched']]
    not_matched = [m for m in match_log if not m['matched']]

    lines = []
    lines.append("=" * 70)
    lines.append("GLOF SPATIAL MATCHING — GRADUATED BUFFER REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append(f"Buffer strategy  : S2-era (>={S2_START_YEAR}): {BUFFER_S2_M/1000:.0f} km  |"
                 f"  Historical: {BUFFER_HIST_M/1000:.0f} km")
    lines.append(f"Total GLOF events: {total_events}")
    lines.append(f"Matched events   : {total_matched}  ({100*total_matched/total_events:.1f}%)")
    lines.append(f"Unique GLOF lakes: {n_pos_lakes}  (n+ for ML)")
    lines.append("")
    lines.append("MATCHED EVENTS:")
    for m in sorted(matched, key=lambda x: x['area']):
        lines.append(f"  [{m['buffer_m']/1000:.0f}km buf] {m['area']:<30} "
                     f"{m['lake']:<35} {m['date'][:10]}  dist={m['dist_m']}m")
    lines.append("")
    lines.append("NOT MATCHED:")
    for m in sorted(not_matched, key=lambda x: (x['area'], x['dist_m'] or 999999)):
        d = f"{m['dist_m']}m" if m['dist_m'] else 'no_lakes'
        lines.append(f"  [{m['buffer_m']/1000:.0f}km buf] {m['area']:<30} "
                     f"{m['lake']:<35} {m['date'][:10]}  closest={d}")
    lines.append("=" * 70)
    report = "\n".join(lines)
    print(report)
    return report


def main():
    print("GLOF Graduated-Buffer Rematching")
    print("=" * 50)

    lakes, glofs = load_data()

    print("\nRunning spatial matching...")
    lakes_v2, match_log, n_ev, n_match, n_pos = rematch(lakes, glofs)

    report = print_report(match_log, n_ev, n_match, n_pos)

    # Save outputs
    out_dir = DATA_DIR / 'labeled'
    out_dir.mkdir(parents=True, exist_ok=True)

    lakes_out = out_dir / 'labeled_lakes_v2.gpkg'
    lakes_v2.to_file(lakes_out, driver='GPKG')
    print(f"\nSaved: {lakes_out}")

    # Update training CSV (drop geometry)
    csv_out = out_dir / 'training_data_v2.csv'
    df_csv = lakes_v2.drop(columns=['geometry'], errors='ignore')
    df_csv.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    # Save match log
    log_df = pd.DataFrame(match_log)
    log_out = out_dir / 'match_log_v2.csv'
    log_df.to_csv(log_out, index=False)
    print(f"Saved: {log_out}")

    # Save report
    report_out = MODELS_DIR / 'matching_report_v2.txt'
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report, encoding='utf-8')
    print(f"Saved: {report_out}")

    print(f"\nSUMMARY: n+ changed from 17 -> {n_pos}  ({n_pos-17:+d} positives)")
    return n_pos


if __name__ == '__main__':
    main()
