#!/usr/bin/env python3
"""
Rebuild GHAP daily zarr for a given year from Zenodo D1K zip archives.
Usage: python rebuild_ghap_year.py <year> <zenodo_record_id>
"""
import os, sys, shutil, tempfile, time, zipfile
import urllib.request
from pathlib import Path
from calendar import monthrange, isleap

import numpy as np
import zarr
import numcodecs
import xarray as xr

YEAR          = int(sys.argv[1])
ZENODO_RECORD = sys.argv[2]
RAW_DIR  = Path(f"/scratch/project_462001140/ammar/eccv/data/raw/ghap_daily_{YEAR}_zips")
OUT_DIR  = Path("/scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily")
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=2)
CHUNKS     = (1, 2048, 2048)  # large chunks to minimize inode count (18K files/year vs 933K)

RAW_DIR.mkdir(parents=True, exist_ok=True)

total_days  = 366 if isleap(YEAR) else 365
tmp_path    = OUT_DIR / f"{YEAR}_new.zarr"
output_path = OUT_DIR / f"{YEAR}.zarr"
old_path    = OUT_DIR / f"{YEAR}_old.zarr"

print(f"=== Rebuild GHAP {YEAR} (Zenodo {ZENODO_RECORD}) ===")
print(f"Total days: {total_days}")

if tmp_path.exists():
    shutil.rmtree(str(tmp_path))

zarr_store = None
t_start    = time.time()
day_idx    = 0

for month in range(1, 13):
    filename = f"GHAP_PM2.5_D1K_{YEAR}{month:02d}_V1.zip"
    url      = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{filename}?download=1"
    zip_path = RAW_DIR / filename

    if not zip_path.exists() or zip_path.stat().st_size < 1e7:
        print(f"\n[{YEAR}-{month:02d}] Downloading {filename}...", flush=True)
        def progress(b, bs, ts):
            if ts > 0 and b % 2000 == 0:
                print(f"  {100*b*bs/ts:.0f}%", end='\r', flush=True)
        try:
            urllib.request.urlretrieve(url, str(zip_path), reporthook=progress)
            print(f"  OK ({zip_path.stat().st_size/1e9:.1f} GB)", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}")
            day_idx += monthrange(YEAR, month)[1]
            continue
    else:
        print(f"\n[{YEAR}-{month:02d}] Exists ({zip_path.stat().st_size/1e9:.1f} GB)", flush=True)

    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            nc_files = sorted([m for m in zf.namelist() if m.endswith('.nc')])
            print(f"  {len(nc_files)} daily NC files", flush=True)
            with tempfile.TemporaryDirectory(dir=str(RAW_DIR)) as tmpdir:
                for nc_name in nc_files:
                    zf.extract(nc_name, tmpdir)
                    nc_path = os.path.join(tmpdir, nc_name)
                    try:
                        ds = xr.open_dataset(nc_path)
                        pm_var = next((v for v in ds.data_vars if 'pm' in v.lower() or 'PM' in v),
                                      list(ds.data_vars)[0])
                        day_data = ds[pm_var].values.astype(np.float32)
                        if day_data.ndim == 3:
                            day_data = day_data[0]
                        ds.close()
                        # NaN = ocean/missing → 0 (consistent with 2018-2020 zarrs)
                        day_data = np.nan_to_num(day_data, nan=0.0)

                        if zarr_store is None:
                            h, w = day_data.shape
                            print(f"  Grid: ({h}, {w}) → zarr shape ({total_days},{h},{w})", flush=True)
                            zarr_store = zarr.open(
                                str(tmp_path), mode='w',
                                shape=(total_days, h, w),
                                chunks=CHUNKS, dtype=np.float32,
                                compressor=COMPRESSOR, zarr_format=2
                            )
                            zarr_store.attrs.update({'year': YEAR,
                                                     'source': 'GHAP GlobalHighPM2.5 D1K',
                                                     'units': 'ug/m3'})

                        zarr_store[day_idx] = day_data
                        print(f"  Day {day_idx+1:03d}/{total_days} {nc_name[-12:]}: "
                              f"max={day_data.max():.1f}  [{time.time()-t_start:.0f}s]", flush=True)
                    except Exception as e:
                        print(f"  ERROR {nc_name}: {e}")
                        if zarr_store is not None:
                            zarr_store[day_idx] = np.zeros(zarr_store.shape[1:], dtype=np.float32)
                    day_idx += 1
                    os.unlink(nc_path)
    except Exception as e:
        print(f"  ZIP ERROR month {month}: {e}")
        day_idx += monthrange(YEAR, month)[1]
        continue

    zip_path.unlink()
    print(f"  Zip deleted. Elapsed: {time.time()-t_start:.0f}s", flush=True)

if zarr_store is None:
    print("FAILED: no data written"); sys.exit(1)

gb = sum(f.stat().st_size for f in tmp_path.rglob('*') if f.is_file()) / 1e9
print(f"\nAll {day_idx} days written — {gb:.1f} GB")

# Atomic swap (same filesystem = instant rename)
print("Swapping zarr...", flush=True)
if output_path.exists():
    os.rename(str(output_path), str(old_path))
os.rename(str(tmp_path), str(output_path))
if old_path.exists():
    shutil.rmtree(str(old_path))

print(f"DONE {YEAR} in {time.time()-t_start:.0f}s total")
