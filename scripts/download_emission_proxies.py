#!/usr/bin/env python3
"""
Download and preprocess global emission proxy layers at 1km:
  1. VIIRS nighttime lights 2019 (500m → 1km)  — NOAA/EOG Colorado Mines
  2. WorldPop population density 2020 (1km)     — worldpop.org
  3. GRIP road density (500m → 1km)             — GLOBIO/PBL Netherlands

Output: /scratch/project_462001140/ammar/eccv/data/zarr/emission_proxies_global.zarr
        shape=(3, 18000, 36000), dtype=float32
        channels: [road_density, nighttime_lights, population]
        grid: lat 90→-90, lon -180→180, 0.01deg
"""
import numpy as np
import zarr, numcodecs
from pathlib import Path
import urllib.request, os, tempfile, zipfile, tarfile
import warnings; warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("WARNING: rasterio not found — install with: pip install rasterio")

OUT_ZARR = Path('/scratch/project_462001140/ammar/eccv/data/zarr/emission_proxies_global.zarr')
TMP_DIR  = Path('/scratch/project_462000640/ammar/tmp/emission_proxies')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Target grid: 18000×36000 at 0.01deg
NLAT, NLON = 18000, 36000
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=2)
CHUNKS     = (1, 512, 512)   # same as GHAP

def download(url, dest):
    if Path(dest).exists():
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")

def resample_to_grid(src_path, shape=(NLAT, NLON)):
    """Resample any GeoTIFF to global 0.01deg grid using rasterio."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    with rasterio.open(src_path) as src:
        data = src.read(
            1,
            out_shape=(shape[0], shape[1]),
            resampling=Resampling.average
        ).astype(np.float32)
    return data

# ══════════════════════════════════════════════════════════
# 1. VIIRS Nighttime Lights 2019 (Annual VNP46A4)
#    Source: Colorado Mines EOG
#    https://eogdata.mines.edu/nighttime_light/annual/v22/
# ══════════════════════════════════════════════════════════
print('\n=== 1. VIIRS Nighttime Lights 2019 ===')
viirs_url  = 'https://eogdata.mines.edu/nighttime_light/annual/v22/2019/VNP46A4/VNP46A4_NearNadir_Composite_Snow_Free_qflag255_20190101_20191231_global_vcmslcfg_v2.2.tif.gz'
viirs_gz   = str(TMP_DIR / 'viirs_2019.tif.gz')
viirs_tif  = str(TMP_DIR / 'viirs_2019.tif')

download(viirs_url, viirs_gz)

if not Path(viirs_tif).exists():
    import gzip, shutil
    print('  Extracting...')
    with gzip.open(viirs_gz, 'rb') as f_in:
        with open(viirs_tif, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print('  Resampling to 0.01deg...')
viirs_data = resample_to_grid(viirs_tif)
viirs_data = np.clip(viirs_data, 0, None)
# Normalize to 0-1 range (log scale, common for nightlights)
viirs_log = np.log1p(viirs_data)
viirs_norm = viirs_log / np.nanpercentile(viirs_log[viirs_log > 0], 99)
viirs_norm = np.clip(viirs_norm, 0, 1).astype(np.float32)
print(f'  VIIRS shape: {viirs_norm.shape}, non-zero: {(viirs_norm>0).sum():,}')


# ══════════════════════════════════════════════════════════
# 2. WorldPop Population Density 2020 (1km)
#    Source: worldpop.org UN-adjusted
# ══════════════════════════════════════════════════════════
print('\n=== 2. WorldPop Population Density 2020 ===')
wpop_url  = 'https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km_UNadj/2020/0_Mosaicked/ppp_2020_1km_Aggregated_UNadj.tif'
wpop_tif  = str(TMP_DIR / 'worldpop_2020_1km.tif')

download(wpop_url, wpop_tif)

print('  Resampling to 0.01deg...')
wpop_data = resample_to_grid(wpop_tif)
wpop_data = np.nan_to_num(wpop_data, nan=0.0)
wpop_data = np.clip(wpop_data, 0, None)
# Log normalize
wpop_log  = np.log1p(wpop_data)
wpop_norm = wpop_log / np.nanpercentile(wpop_log[wpop_log > 0], 99)
wpop_norm = np.clip(wpop_norm, 0, 1).astype(np.float32)
print(f'  WorldPop shape: {wpop_norm.shape}, non-zero: {(wpop_norm>0).sum():,}')


# ══════════════════════════════════════════════════════════
# 3. GRIP Global Road Density (500m)
#    Source: https://www.globio.info/download-grip-dataset
#    GRIP4 global roads — total road length per cell
# ══════════════════════════════════════════════════════════
print('\n=== 3. GRIP Road Density ===')
grip_url  = 'https://dataportaal.pbl.nl/downloads/GRIP4/GRIP4_TotalRoads_density_global.zip'
grip_zip  = str(TMP_DIR / 'grip4_roads.zip')
grip_dir  = TMP_DIR / 'grip4'
grip_tif  = str(grip_dir / 'GRIP4_TotalRoads_density_global_v1.0.tif')

download(grip_url, grip_zip)

if not grip_dir.exists() or not Path(grip_tif).exists():
    print('  Extracting ZIP...')
    with zipfile.ZipFile(grip_zip, 'r') as z:
        z.extractall(str(grip_dir))
    # Find the tif
    tifs = list(grip_dir.rglob('*.tif'))
    print(f'  Found: {tifs}')
    grip_tif = str(tifs[0]) if tifs else grip_tif

print('  Resampling to 0.01deg...')
grip_data = resample_to_grid(grip_tif)
grip_data = np.nan_to_num(grip_data, nan=0.0)
grip_data = np.clip(grip_data, 0, None)
# Log normalize
grip_log  = np.log1p(grip_data)
grip_norm = grip_log / np.nanpercentile(grip_log[grip_log > 0], 99)
grip_norm = np.clip(grip_norm, 0, 1).astype(np.float32)
print(f'  GRIP shape: {grip_norm.shape}, non-zero: {(grip_norm>0).sum():,}')


# ══════════════════════════════════════════════════════════
# 4. Save to zarr
# ══════════════════════════════════════════════════════════
print('\n=== Saving to zarr ===')
if OUT_ZARR.exists():
    import shutil; shutil.rmtree(str(OUT_ZARR))

root = zarr.open_group(str(OUT_ZARR), mode='w', zarr_format=2)

for name, arr in [('road_density', grip_norm),
                   ('nighttime_lights', viirs_norm),
                   ('population', wpop_norm)]:
    print(f'  Writing {name}...')
    root.create_array(name, data=arr, chunks=(512, 512),
                      compressor=COMPRESSOR, overwrite=True)
    print(f'    shape={arr.shape}, mean={arr.mean():.4f}, max={arr.max():.4f}')

# Save metadata
root.attrs['description'] = 'Global emission proxy layers at 0.01deg (1km)'
root.attrs['channels']    = ['road_density', 'nighttime_lights', 'population']
root.attrs['grid']        = 'lat 90 to -90, lon -180 to 180, 0.01deg'
root.attrs['sources']     = {
    'road_density':     'GRIP4 Global Roads Inventory Project v1.0',
    'nighttime_lights': 'VIIRS VNP46A4 Annual 2019, Colorado Mines EOG',
    'population':       'WorldPop 2020 1km UN-adjusted',
}

print(f'\nSaved: {OUT_ZARR}')
print('Channels: road_density, nighttime_lights, population')
print('Shape: (18000, 36000) each')
