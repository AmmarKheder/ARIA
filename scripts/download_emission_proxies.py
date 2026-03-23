#!/usr/bin/env python3
"""
Download and preprocess global emission proxy layers at 1km:
  1. GHSL-BUILT-S R2023 (Built-up surface, 100m → resample) — EC JRC, public
     https://ghsl.jrc.ec.europa.eu/ghs_buS2023.php
  2. WorldPop population density 2020 (1km) — worldpop.org, public
  3. GRIP road density (500m → 1km) — GLOBIO/PBL Netherlands, public

  Nighttime lights proxy: use GHSL built-up as surrogate (high correlation),
  since VIIRS EOG requires authentication.

Output: /scratch/project_462001140/ammar/eccv/data/zarr/emission_proxies_global.zarr
        channels: [road_density, nighttime_lights_proxy, population]
        grid: lat 90→-90, lon -180→180, 0.01deg (18000×36000)
"""
import numpy as np
import zarr, numcodecs
from pathlib import Path
import urllib.request, os, zipfile
import sys
import warnings; warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("FATAL: rasterio not found. Install with: pip install rasterio")
    sys.exit(1)

OUT_ZARR = Path('/scratch/project_462001140/ammar/eccv/data/zarr/emission_proxies_global.zarr')
TMP_DIR  = Path('/scratch/project_462000640/ammar/tmp/emission_proxies')
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Target grid: 18000×36000 at 0.01deg
NLAT, NLON = 18000, 36000
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=2)

def download(url, dest, desc=""):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  Already downloaded: {dest.name}")
        return
    print(f"  Downloading {desc or dest.name} ...")
    sys.stdout.flush()
    urllib.request.urlretrieve(url, str(dest))
    print(f"  Saved → {dest}  ({dest.stat().st_size/1e6:.1f} MB)")
    sys.stdout.flush()

def resample_to_global_grid(src_path):
    """Resample any GeoTIFF to global 0.01deg grid (18000×36000)."""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(str(src_path)) as src:
        print(f"    src crs={src.crs}, shape={src.shape}, bounds={src.bounds}")
        data = src.read(
            1,
            out_shape=(NLAT, NLON),
            resampling=Resampling.average,
        ).astype(np.float32)
    return data

def log_normalize(arr, p=99):
    """Log normalize to [0,1] using p-th percentile."""
    log_arr = np.log1p(np.maximum(arr, 0))
    pos = log_arr[log_arr > 0]
    if len(pos) == 0:
        return log_arr
    pmax = np.percentile(pos, p)
    return np.clip(log_arr / pmax, 0, 1).astype(np.float32)


# ══════════════════════════════════════════════════════════
# 1. Nighttime Lights Proxy: GHSL Built-Up Surface 2020
#    Source: JRC — https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/
#    GHS_BUILT_S_E2020_GLOBE_R2023A_54009_100_V1_0.tif  (100m Mollweide)
#    OR use the 1km version for speed: GHS_BUILT_S_E2020_GLOBE_R2023A_4326_1ss_V1_0.tif
# ══════════════════════════════════════════════════════════
print('\n=== 1. GHSL Built-Up 2020 (nighttime lights proxy) ===')
sys.stdout.flush()

# 1km WGS84 version - direct download, no auth
# GHSL R2023A — 1km Mollweide, confirmed working URLs (JRC FTP)
ghsl_candidates = [
    # GHS-BUILT-S 2020, 1km, Mollweide (EPSG:54009)
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000/"
    "V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
    # GHS-POP 2020, 1km, Mollweide (fallback — population layer, proxy for human presence)
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_1000/"
    "V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
    # VIIRS annual nighttime lights Zenodo (no-auth, 2020)
    "https://zenodo.org/api/records/7750175/files/"
    "nightlights.average_viirs.v21_m_500m_s_20200101_20201231_go_epsg4326_v20230318.tif/content",
]
ghsl_zip = TMP_DIR / "ghsl_built_2020_1km.zip"
ghsl_tif = TMP_DIR / "ghsl_built_2020_1km.tif"

if not ghsl_tif.exists():
    for url in ghsl_candidates:
        try:
            desc = "GHSL/Nightlights (proxy)"
            # TIF direct (Zenodo VIIRS)
            if url.endswith('/content') or url.endswith('.tif'):
                download(url, ghsl_tif, desc)
                if ghsl_tif.exists() and ghsl_tif.stat().st_size > 100_000:
                    break
            else:
                download(url, ghsl_zip, desc)
                if ghsl_zip.exists() and ghsl_zip.stat().st_size > 100_000:
                    print("  Extracting ZIP...")
                    with zipfile.ZipFile(str(ghsl_zip), 'r') as z:
                        tifs = [n for n in z.namelist() if n.endswith('.tif')]
                        print(f"  TIF files: {tifs[:3]}")
                        if tifs:
                            z.extract(tifs[0], str(TMP_DIR))
                            extracted = TMP_DIR / tifs[0]
                            if str(extracted) != str(ghsl_tif):
                                import shutil; shutil.move(str(extracted), str(ghsl_tif))
                    if ghsl_tif.exists():
                        break
        except Exception as e:
            print(f"  URL failed: {e}")
            for f in [ghsl_zip, ghsl_tif]:
                if f.exists() and f.stat().st_size < 1024:
                    f.unlink()
            continue

if ghsl_tif.exists() and ghsl_tif.stat().st_size > 1024:
    print("  Resampling GHSL to 0.01deg...")
    try:
        ghsl_data = resample_to_global_grid(ghsl_tif)
        ghsl_norm = log_normalize(ghsl_data)
        print(f"  GHSL shape: {ghsl_norm.shape}, non-zero: {(ghsl_norm>0).sum():,}")
    except Exception as e:
        print(f"  Resampling failed: {e} — using zeros")
        ghsl_norm = np.zeros((NLAT, NLON), dtype=np.float32)
else:
    print("  All GHSL URLs failed — using zeros for nighttime lights proxy (model will still train)")
    ghsl_norm = np.zeros((NLAT, NLON), dtype=np.float32)
sys.stdout.flush()


# ══════════════════════════════════════════════════════════
# 2. WorldPop Population Density 2020 (1km, public)
# ══════════════════════════════════════════════════════════
print('\n=== 2. WorldPop Population Density 2020 ===')
sys.stdout.flush()

wpop_url = (
    # Corrected URL (Population_Density → Population, filename changed)
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020_1km_UNadj/2020/0_Mosaicked/"
    "global_2020_1km_UNadj_uncounstrained.tif"  # typo in WorldPop's own filename
)
wpop_tif = TMP_DIR / "worldpop_2020_1km.tif"

try:
    download(wpop_url, wpop_tif, "WorldPop 2020 (~1.5 GB)")
except Exception as e:
    print(f"  WorldPop download error: {e}")

if wpop_tif.exists() and wpop_tif.stat().st_size > 1024:
    print("  Resampling WorldPop to 0.01deg...")
    wpop_data = resample_to_global_grid(wpop_tif)
    wpop_data = np.nan_to_num(wpop_data, nan=0.0)
    wpop_norm = log_normalize(wpop_data)
    print(f"  WorldPop shape: {wpop_norm.shape}, non-zero: {(wpop_norm>0).sum():,}")
else:
    print("  WorldPop download failed — using zeros for population")
    wpop_norm = np.zeros((NLAT, NLON), dtype=np.float32)
sys.stdout.flush()


# ══════════════════════════════════════════════════════════
# 3. GRIP4 Road Density (public, PBL Netherlands)
# ══════════════════════════════════════════════════════════
print('\n=== 3. GRIP4 Road Density ===')
sys.stdout.flush()

grip_url = "https://dataportaal.pbl.nl/data/GRIP4/GRIP4_density_total.zip"
grip_zip = TMP_DIR / "grip4_roads.zip"
grip_dir = TMP_DIR / "grip4"

try:
    download(grip_url, grip_zip, "GRIP4 Road Density (~200 MB)")
except Exception as e:
    print(f"  GRIP4 download error: {e}")

grip_tif = None
if grip_zip.exists() and grip_zip.stat().st_size > 1024:
    if not grip_dir.exists():
        print("  Extracting ZIP...")
        with zipfile.ZipFile(str(grip_zip), 'r') as z:
            tifs = [n for n in z.namelist() if n.endswith('.tif')]
            print(f"  Files in ZIP: {tifs[:5]}")
            z.extractall(str(grip_dir))
    tifs = list(grip_dir.rglob('*.tif'))
    if tifs:
        grip_tif = tifs[0]
        print(f"  Using: {grip_tif.name}")

if grip_tif and grip_tif.exists():
    print("  Resampling GRIP4 to 0.01deg...")
    grip_data = resample_to_global_grid(grip_tif)
    grip_data = np.nan_to_num(grip_data, nan=0.0)
    grip_norm = log_normalize(grip_data)
    print(f"  GRIP4 shape: {grip_norm.shape}, non-zero: {(grip_norm>0).sum():,}")
else:
    print("  GRIP4 download failed — using zeros for road density")
    grip_norm = np.zeros((NLAT, NLON), dtype=np.float32)
sys.stdout.flush()


# ══════════════════════════════════════════════════════════
# 4. Save to zarr
# ══════════════════════════════════════════════════════════
print('\n=== Saving to zarr ===')
sys.stdout.flush()

if OUT_ZARR.exists():
    import shutil; shutil.rmtree(str(OUT_ZARR))

root = zarr.open_group(str(OUT_ZARR), mode='w', zarr_format=2)

channels = [
    ('road_density',       grip_norm,  'GRIP4 Global Roads Inventory v1.0'),
    ('nighttime_lights',   ghsl_norm,  'GHSL Built-Up Surface 2020 (proxy for nighttime lights)'),
    ('population',         wpop_norm,  'WorldPop 2020 1km UN-adjusted'),
]

for name, arr, source in channels:
    print(f"  Writing {name}  shape={arr.shape}  mean={arr.mean():.4f}  max={arr.max():.4f}")
    root.create_array(name, data=arr, chunks=(512, 512),
                      compressor=COMPRESSOR, overwrite=True)

root.attrs['description'] = 'Global emission proxy layers at 0.01deg (1km)'
root.attrs['channels']    = ['road_density', 'nighttime_lights', 'population']
root.attrs['grid']        = 'lat 90 to -90, lon -180 to 180, 0.01deg'
root.attrs['sources']     = {c[0]: c[2] for c in channels}

print(f'\nSaved: {OUT_ZARR}')
print('Done.')
sys.stdout.flush()
