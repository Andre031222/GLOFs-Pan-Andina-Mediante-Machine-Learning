"""
run_pipeline_v2.py
------------------
Executes the full expansion pipeline for the 4 new study areas:
  patagonia_norte, patagonia_sur, apolobamba, carabaya

Steps:
  1. Download Sentinel-2 + DEM for each new area (NB20-NB23)
  2. Process DEMs / terrain derivatives (NB11 logic inline)
  3. Detect glacial lakes with per-area elevation bounds (NB12)
  4. Extract lake features (NB13)
  5. Match GLOF events to lakes (NB14)
  6. Retrain the susceptibility model (NB15)

Run from project root with the venv activated:
    python run_pipeline_v2.py 2>&1 | tee logs/pipeline_v2.log
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f'pipeline_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

sys.path.insert(0, str(PROJECT_ROOT))

NEW_AREAS = ['patagonia_norte', 'patagonia_sur', 'apolobamba', 'carabaya']

# Areas whose config changed (bbox expanded or cloud filter relaxed) — force re-download
# even if data already exists in data/raw/.
FORCE_REDOWNLOAD = {'patagonia_norte', 'patagonia_sur'}
DOWNLOAD_NOTEBOOKS = {
    'patagonia_norte': 'notebooks/20_download_patagonia_norte.ipynb',
    'patagonia_sur':   'notebooks/21_download_patagonia_sur.ipynb',
    'apolobamba':      'notebooks/22_download_apolobamba.ipynb',
    'carabaya':        'notebooks/23_download_carabaya.ipynb',
}
PIPELINE_NOTEBOOKS = [
    ('NB11 DEM',      'notebooks/11_dem_processing.ipynb'),
    ('NB12 Lakes',    'notebooks/12_lake_detection.ipynb'),
    ('NB13 Features', 'notebooks/13_feature_extraction.ipynb'),
    ('NB14 GLOF',     'notebooks/14_historical_glofs.ipynb'),
    ('NB15 Model',    'notebooks/15_model_training.ipynb'),
]

NBCONVERT_TIMEOUT = 36000  # 10 hours per notebook


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_notebook(label, nb_path, timeout=NBCONVERT_TIMEOUT):
    nb_path = PROJECT_ROOT / nb_path
    if not nb_path.exists():
        log(f'  [SKIP] {label}: notebook not found: {nb_path}')
        return False

    log(f'  START {label}: {nb_path.name}')
    t0 = time.time()
    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--inplace',
        f'--ExecutePreprocessor.timeout={timeout}',
        '--ExecutePreprocessor.kernel_name=python3',
        str(nb_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout + 60,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f'  OK    {label}: completed in {elapsed/60:.1f} min')
        return True
    else:
        log(f'  FAIL  {label}: exit={result.returncode} in {elapsed/60:.1f} min')
        log(f'        stderr: {result.stderr[-2000:]}')
        return False


def check_area_downloaded(area):
    s2_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / area
    dem_dir = PROJECT_ROOT / 'data' / 'raw' / 'dem' / area
    s2_ok = s2_dir.exists() and any(s2_dir.iterdir())
    dem_ok = dem_dir.exists() and any(dem_dir.glob('*.tif'))
    return s2_ok, dem_ok


def main():
    log('=' * 70)
    log('GLOF ANDES v2 EXPANSION PIPELINE')
    log(f'New areas: {NEW_AREAS}')
    log(f'Log file : {LOG_FILE}')
    log('=' * 70)

    # ── Step 1: Downloads ────────────────────────────────────────────────────
    log('\n[STEP 1] Downloading Sentinel-2 + DEM for new areas')
    log(f'  Force re-download: {sorted(FORCE_REDOWNLOAD)}')
    download_ok = {}
    for area in NEW_AREAS:
        s2_ok, dem_ok = check_area_downloaded(area)
        if s2_ok and dem_ok and area not in FORCE_REDOWNLOAD:
            log(f'  [skip] {area}: already downloaded')
            download_ok[area] = True
        else:
            if area in FORCE_REDOWNLOAD:
                log(f'  [force] {area}: config changed — re-downloading')
            nb = DOWNLOAD_NOTEBOOKS[area]
            ok = run_notebook(f'Download {area}', nb, timeout=NBCONVERT_TIMEOUT)
            download_ok[area] = ok
            if not ok:
                log(f'  [WARN] Download failed for {area} — skipping area')

    successful_areas = [a for a, ok in download_ok.items() if ok]
    log(f'\n  Downloads complete: {len(successful_areas)}/{len(NEW_AREAS)} areas OK')

    if not successful_areas:
        log('  ERROR: No areas downloaded successfully. Aborting.')
        return 1

    # ── Steps 2-6: Full pipeline notebooks ──────────────────────────────────
    log('\n[STEPS 2-6] Running full pipeline notebooks')
    for label, nb_path in PIPELINE_NOTEBOOKS:
        ok = run_notebook(label, nb_path, timeout=NBCONVERT_TIMEOUT)
        if not ok:
            log(f'  [WARN] {label} failed — continuing to next step')

    # ── Final summary ────────────────────────────────────────────────────────
    log('\n' + '=' * 70)
    log('PIPELINE COMPLETE')
    log('=' * 70)

    # Check labeled dataset output
    labeled_path = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'labeled_lakes.gpkg'
    if labeled_path.exists():
        import geopandas as gpd
        try:
            gdf = gpd.read_file(labeled_path)
            n_pos = int(gdf['had_glof'].sum()) if 'had_glof' in gdf.columns else -1
            n_total = len(gdf)
            log(f'  Labeled dataset: {n_total} lakes, {n_pos} GLOF positives')
        except Exception as e:
            log(f'  Labeled dataset exists but could not read: {e}')
    else:
        log('  WARNING: labeled_lakes.gpkg not found')

    # Check model output
    model_path = PROJECT_ROOT / 'models' / 'robustness_results.json'
    if model_path.exists():
        try:
            with open(model_path) as f:
                r = json.load(f)
            auc = r.get('roc_auc_mean', r.get('roc_auc', 'N/A'))
            log(f'  Model ROC-AUC: {auc}')
        except Exception:
            log(f'  Model results file exists: {model_path}')
    else:
        log('  WARNING: robustness_results.json not found')

    log(f'\nFull log: {LOG_FILE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
