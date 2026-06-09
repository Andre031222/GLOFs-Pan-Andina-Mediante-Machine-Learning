"""
compute_dist_glacier.py
=======================
Calcula la distancia mínima de cada lago glaciar al glaciar más cercano
usando RGI 7.0 (regiones 16 y 17 para los Andes).

Salida: data/interim/terrain/{area}/dist_glacier.tif  (distancia en metros)

Uso: ejecutar desde la raíz del proyecto con .venv_glof/bin/python3
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

RAW_DIR     = Path('data/raw')
INTERIM_DIR = Path('data/interim')
RGI_DIR     = RAW_DIR / 'rgi7'

# ── Cargar RGI 7.0 ──────────────────────────────────────────────────────────
print('Cargando RGI 7.0...')
rgis = []
for reg, fname in [('RGI16', 'RGI16.shp'), ('RGI17', 'RGI17.shp')]:
    p = RGI_DIR / fname
    if p.exists():
        r = gpd.read_file(p)
        rgis.append(r)
        print(f'  {reg}: {len(r)} glaciares')
    else:
        print(f'  {reg}: NO ENCONTRADO ({p})')

if not rgis:
    print('ERROR: No se encontraron archivos RGI.')
    sys.exit(1)

glaciers_wgs84 = gpd.pd.concat(rgis, ignore_index=True)
glaciers_wgs84 = glaciers_wgs84[['geometry']].copy()
print(f'Total glaciares cargados: {len(glaciers_wgs84)}')
print()

# ── Procesar cada área ───────────────────────────────────────────────────────
for area, cfg in EXPANDED_STUDY_AREAS.items():
    dem_path = INTERIM_DIR / 'dem' / f'{area}_dem_utm.tif'
    out_path = INTERIM_DIR / 'terrain' / area / 'dist_glacier.tif'

    if not dem_path.exists():
        print(f'[SKIP] {area}: DEM no encontrado')
        continue

    print(f'Procesando: {area}')

    with rasterio.open(dem_path) as src:
        meta   = src.meta.copy()
        tf     = src.transform
        h, w   = src.height, src.width
        utm_crs = src.crs

    # Clip glaciares al bbox del área (con buffer de 30 km)
    from pyproj import Transformer
    bbox = cfg['bbox']  # [lon_min, lat_min, lon_max, lat_max]
    buf  = 0.3          # ~30 km en grados
    glaciers_clip = glaciers_wgs84.cx[
        bbox[0]-buf : bbox[2]+buf,
        bbox[1]-buf : bbox[3]+buf
    ].copy()

    if len(glaciers_clip) == 0:
        print(f'  Sin glaciares en bbox — dist_glacier = 99999 m')
        dist = np.full((h, w), 99999.0, dtype=np.float32)
    else:
        # Reproyectar glaciares a UTM del área
        glaciers_utm = glaciers_clip.to_crs(utm_crs)

        # Rasterizar: 1 donde hay glaciar, 0 en el resto
        glacier_mask = rasterize(
            [(geom, 1) for geom in glaciers_utm.geometry if geom is not None],
            out_shape=(h, w),
            transform=tf,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        n_glacier_px = glacier_mask.sum()
        print(f'  Glaciares UTM: {len(glaciers_utm)} polígonos | {n_glacier_px} píxeles')

        if n_glacier_px == 0:
            print(f'  Sin píxeles glaciar en el DEM — dist_glacier = 99999 m')
            dist = np.full((h, w), 99999.0, dtype=np.float32)
        else:
            # distance_transform_edt: distancia en píxeles desde cero-mask
            # no-glacier = 0 → queremos distancia desde estos píxeles al glaciar
            dist_px = distance_transform_edt(glacier_mask == 0)
            cell_size = abs(tf[0])  # metros por píxel
            dist = (dist_px * cell_size).astype(np.float32)

    # Guardar
    out_meta = meta.copy()
    out_meta.update({'dtype': 'float32', 'count': 1,
                     'nodata': -9999.0, 'compress': 'lzw'})
    with rasterio.open(out_path, 'w', **out_meta) as dst:
        dst.write(dist[np.newaxis])

    valid = dist[dist >= 0]
    print(f'  dist_glacier: min={valid.min():.0f}m  mean={valid.mean():.0f}m  max={valid.max():.0f}m')
    print(f'  Guardado: {out_path}')
    print()

print('=== dist_glacier completado para todas las áreas ===')
