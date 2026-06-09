"""
add_dist_glacier_to_features.py
================================
Añade dist_glacier_m real (desde RGI 7.0) al CSV/GPKG de features
generado por NB13. Ejecutar DESPUÉS de NB13.

Método: para cada lago (centroide en UTM), extrae el valor del
raster dist_glacier.tif del área correspondiente.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

PROCESSED_DIR = Path('data/processed')
INTERIM_DIR   = Path('data/interim')

features_gpkg = PROCESSED_DIR / 'features' / 'lake_features.gpkg'
features_csv  = PROCESSED_DIR / 'features' / 'lake_features.csv'

if not features_gpkg.exists():
    print(f'ERROR: {features_gpkg} no existe. Ejecutar NB13 primero.')
    sys.exit(1)

print('Cargando lake_features.gpkg...')
gdf = gpd.read_file(features_gpkg)
print(f'  {len(gdf)} lagos, CRS={gdf.crs}')

area_col = 'area_name' if 'area_name' in gdf.columns else 'study_area'
gdf['dist_glacier_m'] = np.nan

for area in gdf[area_col].unique():
    dist_path = INTERIM_DIR / 'terrain' / area / 'dist_glacier.tif'
    if not dist_path.exists():
        print(f'  [SKIP] {area}: dist_glacier.tif no encontrado')
        continue

    mask = gdf[area_col] == area
    area_gdf = gdf[mask].copy()

    with rasterio.open(dist_path) as src:
        utm_crs = src.crs
        # Centroides en UTM
        if str(area_gdf.crs) != str(utm_crs):
            centroids_utm = area_gdf.to_crs(utm_crs).geometry.centroid
        else:
            centroids_utm = area_gdf.geometry.centroid

        coords = [(pt.x, pt.y) for pt in centroids_utm]
        vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
        # Reemplazar nodata / 99999 con NaN
        vals[(vals < 0) | (vals >= 99000)] = np.nan

    gdf.loc[mask, 'dist_glacier_m'] = vals
    valid = vals[~np.isnan(vals)]
    print(f'  {area}: {len(valid)}/{len(vals)} lagos con dist válida | '
          f'mean={np.nanmean(vals)/1000:.1f}km min={np.nanmin(vals):.0f}m')

# Guardar
for col in gdf.select_dtypes(include='category').columns:
    gdf[col] = gdf[col].astype(str)

gdf.to_file(features_gpkg, driver='GPKG')
print(f'\nGPKG actualizado: {features_gpkg}')

gdf.drop(columns=['geometry']).to_csv(features_csv, index=False)
print(f'CSV actualizado:  {features_csv}')
print(f'\ndist_glacier_m — resumen global:')
print(f'  válidos: {gdf["dist_glacier_m"].notna().sum()}/{len(gdf)}')
print(f'  mean: {gdf["dist_glacier_m"].mean()/1000:.1f} km')
print(f'  min:  {gdf["dist_glacier_m"].min():.0f} m')
print(f'  max:  {gdf["dist_glacier_m"].max()/1000:.1f} km')
