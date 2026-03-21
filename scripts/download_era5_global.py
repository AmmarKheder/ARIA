#!/usr/bin/env python3
"""
Download ERA5 GLOBAL daily data (2017-2022).
Variables: u10, v10, t2m, msl, sp (surface) + t/u/v/q/z at 5 pressure levels = 30 channels.
Output: /scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily/YYYY.zarr
       Shape: (365, 30, 721, 1440) at 0.25° global grid
"""
import sys
import numpy as np
import xarray as xr
import zarr
import numcodecs
from pathlib import Path
import cdsapi
import os

OUTPUT_DIR = Path("/scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily")
TMP_DIR    = Path(f"/tmp/era5_global_download_{os.getpid()}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Global domain — no AREA filter = full globe
# ERA5 global at 0.25°: 721 lat (-90 to 90) × 1440 lon (0 to 359.75)
NLAT, NLON = 721, 1440

SURFACE_VARS = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature":          "t2m",
    "mean_sea_level_pressure": "msl",
    "surface_pressure":        "sp",
}
PRESSURE_VARS = {
    "temperature":          "t",
    "u_component_of_wind":  "u",
    "v_component_of_wind":  "v",
    "specific_humidity":    "q",
    "geopotential":         "z",
}
PRESSURE_LEVELS = ["1000", "925", "850", "700", "500"]

N_SURFACE  = 5
N_PRESSURE = 25  # 5 vars × 5 levels
N_CHANNELS = 30

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)

# Same normalization as Europe training (will recompute global stats later)
ERA5_MEANS = np.array([
    0., 0., 280., 101325., 97000.,
    *([260.]*5 + [0.]*5 + [0.]*5 + [0.003]*5 + [50000.]*5)
], dtype=np.float32)
ERA5_STDS = np.array([
    10., 10., 20., 1500., 5000.,
    *([30.]*5 + [15.]*5 + [10.]*5 + [0.004]*5 + [40000.]*5)
], dtype=np.float32)


def download_surface(client, year, months):
    out = TMP_DIR / f"era5_surface_{year}_{months[0]}-{months[-1]}.nc"
    if out.exists() and out.stat().st_size > 1e6:
        print(f"  Surface cache: {out} ({out.stat().st_size/1e9:.2f} GB)")
        return out
    print(f"  Downloading surface vars for {year} months {months}...")
    for attempt in range(5):
        try:
            if out.exists(): out.unlink()
            client.retrieve("reanalysis-era5-single-levels", {
                "product_type": "reanalysis",
                "variable":     list(SURFACE_VARS.keys()),
                "year":         str(year),
                "month":        months,
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "data_format":  "netcdf",
            }, str(out))
            print(f"  Downloaded: {out.stat().st_size/1e9:.2f} GB")
            return out
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"  Attempt {attempt+1} failed: {e} — retry in {wait}s", flush=True)
            import time; time.sleep(wait)
    raise RuntimeError(f"Surface download failed after 5 attempts")


def download_pressure(client, year, months):
    out = TMP_DIR / f"era5_pressure_{year}_{months[0]}-{months[-1]}.nc"
    if out.exists() and out.stat().st_size > 1e6:
        print(f"  Pressure cache: {out} ({out.stat().st_size/1e9:.2f} GB)")
        return out
    print(f"  Downloading pressure vars for {year} months {months}...")
    for attempt in range(5):
        try:
            if out.exists(): out.unlink()
            client.retrieve("reanalysis-era5-pressure-levels", {
                "product_type": "reanalysis",
                "variable":     list(PRESSURE_VARS.keys()),
                "pressure_level": PRESSURE_LEVELS,
                "year":         str(year),
                "month":        months,
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "data_format":  "netcdf",
            }, str(out))
            print(f"  Downloaded: {out.stat().st_size/1e9:.2f} GB")
            return out
        except Exception as e:
            wait = 60 * (attempt + 1)
            print(f"  Attempt {attempt+1} failed: {e} — retry in {wait}s", flush=True)
            import time; time.sleep(wait)
    raise RuntimeError(f"Pressure download failed after 5 attempts")


def process_year(year):
    print(f"\n{'='*60}")
    print(f"ERA5 GLOBAL {year} — 30 channels, 0.25° ({NLAT}×{NLON})")
    print(f"{'='*60}")

    expected_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    out_path = OUTPUT_DIR / f"{year}.zarr"

    if out_path.exists():
        try:
            z = zarr.open(str(out_path), mode='r')
            # Check actual chunk files exist (not just empty zarr skeleton)
            chunk_files = [f for f in os.listdir(str(out_path)) if not f.startswith('.')]
            if z.shape[0] >= expected_days and len(chunk_files) >= expected_days * 0.9:
                print(f"Already complete: {z.shape}, {len(chunk_files)} chunks")
                return
            else:
                print(f"Incomplete zarr: {z.shape}, only {len(chunk_files)} chunks — re-downloading")
        except:
            pass

    client = cdsapi.Client()

    # Create zarr upfront, write month by month to avoid OOM
    # 1 day × 30 ch × 721 × 1440 × 4 bytes ≈ 125 MB per chunk (Blosc limit: 2 GB)
    z = zarr.open(str(out_path), mode='w', shape=(expected_days, N_CHANNELS, NLAT, NLON),
                  chunks=(1, N_CHANNELS, NLAT, NLON), dtype=np.float32,
                  compressor=COMPRESSOR, zarr_format=2)

    day_offset = 0
    for m in range(1, 13):
        months = [f"{m:02d}"]
        print(f"\n--- Month {m:02d} ---", flush=True)

        nc_surf = download_surface(client, year, months)
        nc_pres = download_pressure(client, year, months)

        # Surface — use chunks for lazy loading
        ds_surf = xr.open_dataset(str(nc_surf), chunks={"time": 4})
        surf_data = []
        for long_name, short_name in SURFACE_VARS.items():
            matched = [v for v in ds_surf.data_vars if short_name.lower() in v.lower()]
            if not matched:
                matched = list(ds_surf.data_vars)
            da = ds_surf[matched[0]]
            if "valid_time" in da.dims and "time" not in da.dims:
                da = da.rename({"valid_time": "time"})
            dm = da.resample(time="1D").mean("time").values
            surf_data.append(dm)
        ds_surf.close()
        del ds_surf
        surf_arr = np.stack(surf_data, axis=1)
        n_days = surf_arr.shape[0]
        print(f"  Surface: {surf_arr.shape}", flush=True)

        # Pressure — use chunks for lazy loading
        ds_pres = xr.open_dataset(str(nc_pres), chunks={"time": 4})
        pres_channels = []
        for long_name, short_name in PRESSURE_VARS.items():
            matched = [v for v in ds_pres.data_vars if short_name.lower() in v.lower()]
            if not matched:
                matched = list(ds_pres.data_vars)
            da = ds_pres[matched[0]]
            if "valid_time" in da.dims and "time" not in da.dims:
                da = da.rename({"valid_time": "time"})
            for lev in [1000, 925, 850, 700, 500]:
                if "level" in da.dims:
                    da_lev = da.sel(level=float(lev), method="nearest")
                elif "pressure_level" in da.dims:
                    da_lev = da.sel(pressure_level=float(lev), method="nearest")
                else:
                    da_lev = da.isel({list(da.dims)[1]: 0})
                dm = da_lev.resample(time="1D").mean("time").values
                pres_channels.append(dm)
        ds_pres.close()
        del ds_pres
        pres_arr = np.stack(pres_channels, axis=1)
        print(f"  Pressure: {pres_arr.shape}", flush=True)

        # Combine and normalize this month
        month_data = np.concatenate([surf_arr, pres_arr], axis=1)
        del surf_arr, pres_arr

    # Normalize
        month_norm = (month_data - ERA5_MEANS[None, :, None, None]) / ERA5_STDS[None, :, None, None]
        month_norm = month_norm.astype(np.float32)
        del month_data

        # Write directly to zarr
        end_day = min(day_offset + n_days, expected_days)
        actual_days = end_day - day_offset
        z[day_offset:end_day] = month_norm[:actual_days]
        print(f"  Written days {day_offset}-{end_day} to zarr", flush=True)
        day_offset = end_day
        del month_norm

        # Clean up temp files to save space
        nc_surf.unlink()
        nc_pres.unlink()
        print(f"  Cleaned temp files", flush=True)

    print(f"\nSaved: {out_path} — shape {z.shape}")


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if year:
        process_year(year)
    else:
        for y in range(2017, 2023):
            process_year(y)
    print("\nDONE!")


if __name__ == "__main__":
    main()
