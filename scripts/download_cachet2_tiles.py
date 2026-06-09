"""
download_cachet2_tiles.py
Descarga dirigida de tiles Sentinel-2 para Lago Cachet 2 (-47.18S, -73.02W).
Busca mes a mes para evitar timeouts del API de Planetary Computer.
"""
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pystac_client
import planetary_computer as pc
from src.download import download_band_windowed

BANDS   = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
AREA    = 'patagonia_norte'
# Tight bbox around Cachet 2 and nearby lakes
BBOX    = [-73.8, -47.8, -72.5, -46.8]
MAX_CLOUD = 25
# December only — most tiles have 0% cloud in this month
MONTHS_YEARS = [(12, y) for y in range(2019, 2025)]
OUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / AREA
MAX_RETRIES = 4


def search_with_retry(catalog, bbox, date_str, max_cloud, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            search = catalog.search(
                collections=['sentinel-2-l2a'],
                bbox=bbox,
                datetime=date_str,
                query={'eo:cloud_cover': {'lt': max_cloud}},
                max_items=30,
            )
            return list(search.items())
        except Exception as e:
            wait = 15 * (2 ** attempt)
            print(f'  [retry {attempt+1}/{retries}] {e} — waiting {wait}s')
            time.sleep(wait)
    return []


def main():
    catalog = pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=pc.sign_inplace,
    )

    downloaded = 0
    for month, year in MONTHS_YEARS:
        date_str = f'{year}-{month:02d}-01/{year}-{month:02d}-28'
        print(f'\n{year}-{month:02d}  searching...', flush=True)

        items = search_with_retry(catalog, BBOX, date_str, MAX_CLOUD)
        if not items:
            print(f'  no items found')
            continue

        # Pick up to 2 unique MGRS tiles (least cloudy)
        items.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100))
        seen = set()
        for item in items:
            tile = item.properties.get('s2:mgrs_tile', item.id)
            if tile in seen or len(seen) >= 2:
                continue
            seen.add(tile)

            cloud = item.properties.get('eo:cloud_cover', 0)
            yr_dir = OUT_DIR / str(year)
            yr_dir.mkdir(parents=True, exist_ok=True)

            for band in BANDS:
                out_file = yr_dir / f'{item.id}_{band}.tif'
                if out_file.exists() and out_file.stat().st_size > 10_000:
                    continue
                for attempt in range(MAX_RETRIES):
                    try:
                        download_band_windowed(
                            item=item, band=band,
                            bbox_wgs84=BBOX, output_path=out_file,
                        )
                        downloaded += 1
                        break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(10 * (attempt + 1))
                        else:
                            print(f'  [FAIL] {band}: {e}')

            mb = sum(f.stat().st_size for f in yr_dir.glob(f'{item.id}_*.tif') if f.exists()) / 1e6
            print(f'  OK  tile={tile}  cloud={cloud:.0f}%  {mb:.0f} MB', flush=True)

    print(f'\nDone. {downloaded} band files downloaded.')

    # Summary
    import glob, os
    all_tiles = glob.glob(str(OUT_DIR / '**' / '*.tif'), recursive=True)
    total_mb = sum(os.path.getsize(t) for t in all_tiles) / 1e6
    print(f'Total patagonia_norte: {len(all_tiles)} files, {total_mb:.0f} MB')


if __name__ == '__main__':
    main()
