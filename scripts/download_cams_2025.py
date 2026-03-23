#!/usr/bin/env python3
"""
Download CAMS Global Analysis (NRT) for 2025.
Dataset: cams-global-atmospheric-composition-forecasts (analysis step=0)
Resolution: 0.4° → regridded to 0.75° to match training CAMS EAC4
Variables: pm2p5, no2, go3, so2, co, pm10
Output: /scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily/2025.zarr
"""
import sys
import numpy as np
import xarray as xr
import zarr
import numcodecs
import cdsapi
from pathlib import Path
from datetime import date, timedelta

YEAR = 2025
# Only download up to yesterday (NRT has ~1 day lag)
END_DATE = date(2025, 3, 22)  # update as needed
OUT_DIR  = Path("/scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily")
TMP_DIR  = Path("/scratch/project_462001140/ammar/eccv/data/raw/cams_2025_tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

VARS_MAP = {
    "particulate_matter_2.5um":  "pm25",
    "nitrogen_dioxide":          "no2",
    "ozone":                     "o3",
    "sulphur_dioxide":           "so2",
    "carbon_monoxide":           "co",
    "particulate_matter_10um":   "pm10",
}
COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)

c = cdsapi.Client()

# Download month by month
start = date(YEAR, 1, 1)
days_by_month = {}
d = start
while d <= END_DATE:
    m = d.month
    days_by_month.setdefault(m, [])
    days_by_month[m].append(d.strftime("%Y-%m-%d"))
    d += timedelta(days=1)

all_daily = {}  # var -> list of (365, 241, 480) daily arrays

for month, day_list in sorted(days_by_month.items()):
    tmp_nc = TMP_DIR / f"cams_2025_{month:02d}.nc"
    if not tmp_nc.exists():
        print(f"Downloading CAMS 2025-{month:02d} ({len(day_list)} days)...", flush=True)
        c.retrieve("cams-global-atmospheric-composition-forecasts", {
            "variable": list(VARS_MAP.keys()),
            "date": day_list,
            "time": ["00:00"],
            "leadtime_hour": ["0"],
            "type": "forecast",
            "format": "netcdf",
        }, str(tmp_nc))
    else:
        print(f"  Already downloaded: {tmp_nc.name}", flush=True)

    ds = xr.open_dataset(str(tmp_nc))
    for cds_var, short in VARS_MAP.items():
        # Try both short and long names
        arr = None
        for vname in [cds_var, short, cds_var.replace("particulate_matter_", "pm"),
                      "pm2p5", "pm10", "go3", "no2", "so2", "co"]:
            if vname in ds:
                arr = ds[vname].values  # (days, lat, lon)
                break
        if arr is None:
            print(f"  WARNING: {cds_var} not found in dataset, skipping")
            continue
        # Convert units if needed (kg/kg → µg/m³ rough conversion for CAMS)
        if arr.mean() < 0.01:  # likely kg/kg or kg/m²
            CONV = {"pm25": 1e9, "pm10": 1e9, "no2": 1e9, "o3": 1e9, "so2": 1e9, "co": 1e6}
            arr = arr * CONV.get(short, 1e9)
        # Regrid 0.4° → 0.75° (241×480) via nearest-neighbour if needed
        if arr.shape[-2:] != (241, 480):
            import scipy.ndimage
            zoom_h = 241 / arr.shape[-2]
            zoom_w = 480 / arr.shape[-1]
            arr = np.stack([
                scipy.ndimage.zoom(arr[i], (zoom_h, zoom_w), order=1)
                for i in range(arr.shape[0])
            ])
        all_daily.setdefault(short, [])
        all_daily[short].append(arr.astype(np.float32))
    ds.close()

# Write to zarr
n_days = (END_DATE - start).days + 1
out_path = OUT_DIR / f"{YEAR}.zarr"
print(f"\nWriting {n_days} days → {out_path}", flush=True)

root = zarr.open_group(str(out_path), mode="w", zarr_format=2)
for var, chunks in all_daily.items():
    data = np.concatenate(chunks, axis=0)  # (n_days, 241, 480)
    root.create_dataset(var, data=data, chunks=(1, 241, 480),
                        compressor=COMPRESSOR, overwrite=True)
    print(f"  {var}: {data.shape}")

# Cleanup tmp
import shutil
shutil.rmtree(str(TMP_DIR))
print("DONE")
