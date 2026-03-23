#!/usr/bin/env python3
"""
Download ERA5 global daily data for full year 2025 for zero-shot inference.
Downloads by quarter to avoid CDS request limits.
Output: (365, 30, 721, 1440) zarr — same format as training years.
"""
import sys, os
import numpy as np
import xarray as xr
import zarr, numcodecs
from pathlib import Path
import cdsapi
import time

OUTPUT_DIR = Path("/scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily")
TMP_DIR    = Path(f"/tmp/era5_2025_{os.getpid()}")
TMP_DIR.mkdir(parents=True, exist_ok=True)

NLAT, NLON = 721, 1440
YEAR = 2025

# Download by quarter to keep CDS requests manageable
QUARTERS = [
    ["01", "02", "03"],
    ["04", "05", "06"],
    ["07", "08", "09"],
    ["10", "11", "12"],
]

SURFACE_VARS = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature":          "t2m",
    "mean_sea_level_pressure": "msl",
    "surface_pressure":        "sp",
}
PRESSURE_VARS = {
    "temperature":         "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity":   "q",
    "geopotential":        "z",
}
PRESSURE_LEVELS = ["1000", "925", "850", "700", "500"]

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)

ERA5_MEANS = np.array([
    0., 0., 280., 101325., 97000.,
    *([260.]*5 + [0.]*5 + [0.]*5 + [0.003]*5 + [50000.]*5)
], dtype=np.float32)
ERA5_STDS = np.array([
    10., 10., 20., 1500., 5000.,
    *([30.]*5 + [15.]*5 + [10.]*5 + [0.004]*5 + [40000.]*5)
], dtype=np.float32)


def get_client():
    return cdsapi.Client()


def download_surface(client, months, label):
    out = TMP_DIR / f"era5_surface_2025_{label}.nc"
    if out.exists() and out.stat().st_size > 1e6:
        print(f"  Surface cache: {out.stat().st_size/1e9:.2f} GB")
        return out
    print(f"  Downloading ERA5 surface 2025 months={months}...")
    sys.stdout.flush()
    for attempt in range(3):
        try:
            client.retrieve("reanalysis-era5-single-levels", {
                "product_type": "reanalysis",
                "variable":     list(SURFACE_VARS.keys()),
                "year":         "2025",
                "month":        months,
                "day":          [f"{d:02d}" for d in range(1, 32)],
                "time":         ["00:00", "06:00", "12:00", "18:00"],
                "data_format":  "netcdf",
            }, str(out))
            print(f"  Downloaded surface: {out.stat().st_size/1e9:.2f} GB")
            return out
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(60)
    return None


def download_pressure(client, months, label):
    out = TMP_DIR / f"era5_pressure_2025_{label}.nc"
    if out.exists() and out.stat().st_size > 1e6:
        print(f"  Pressure cache: {out.stat().st_size/1e9:.2f} GB")
        return out
    print(f"  Downloading ERA5 pressure 2025 months={months}...")
    sys.stdout.flush()
    for attempt in range(3):
        try:
            client.retrieve("reanalysis-era5-pressure-levels", {
                "product_type":   "reanalysis",
                "variable":       list(PRESSURE_VARS.keys()),
                "pressure_level": PRESSURE_LEVELS,
                "year":           "2025",
                "month":          months,
                "day":            [f"{d:02d}" for d in range(1, 32)],
                "time":           ["00:00", "06:00", "12:00", "18:00"],
                "data_format":    "netcdf",
            }, str(out))
            print(f"  Downloaded pressure: {out.stat().st_size/1e9:.2f} GB")
            return out
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(60)
    return None


def process_quarter(surf_nc, pres_nc):
    """Process one quarter of data and return normalized array (n_days, 30, 721, 1440)."""
    ds_s = xr.open_dataset(surf_nc)
    ds_p = xr.open_dataset(pres_nc)

    ds_s = ds_s.resample(valid_time="1D").mean()
    ds_p = ds_p.resample(valid_time="1D").mean()

    n_days = len(ds_s.valid_time)
    print(f"  Processing {n_days} days...")
    sys.stdout.flush()

    days_data = []
    for d in range(n_days):
        channels = []
        for vlong, vshort in SURFACE_VARS.items():
            if vshort in ds_s:
                arr = ds_s[vshort].isel(valid_time=d).values.astype(np.float32)
            else:
                arr = ds_s[vlong.replace("_", "-")].isel(valid_time=d).values.astype(np.float32)
            channels.append(arr)

        for vlong, vshort in PRESSURE_VARS.items():
            for lev in PRESSURE_LEVELS:
                arr = ds_p[vshort].sel(pressure_level=int(lev)).isel(valid_time=d).values.astype(np.float32)
                channels.append(arr)

        data = np.stack(channels, axis=0)  # (30, 721, 1440)
        data = (data - ERA5_MEANS[:, None, None]) / ERA5_STDS[:, None, None]
        days_data.append(data)

        if (d + 1) % 10 == 0:
            print(f"    Day {d+1}/{n_days}")
            sys.stdout.flush()

    ds_s.close()
    ds_p.close()
    return np.stack(days_data, axis=0)  # (n_days, 30, 721, 1440)


def main():
    print("=" * 55)
    print("ERA5 2025 Global Download — Full Year (by quarter)")
    print("=" * 55)
    sys.stdout.flush()

    out_zarr = OUTPUT_DIR / "2025.zarr"
    client = get_client()

    all_quarters = []
    for qi, months in enumerate(QUARTERS):
        label = f"Q{qi+1}"
        print(f"\n{'='*40}")
        print(f"Quarter {qi+1}: months {months}")
        print(f"{'='*40}")
        sys.stdout.flush()

        surf_nc = download_surface(client, months, label)
        if surf_nc is None:
            print(f"  Surface download failed for {label}!")
            sys.exit(1)

        pres_nc = download_pressure(client, months, label)
        if pres_nc is None:
            print(f"  Pressure download failed for {label}!")
            sys.exit(1)

        quarter_data = process_quarter(surf_nc, pres_nc)
        all_quarters.append(quarter_data)
        print(f"  {label} done: {quarter_data.shape[0]} days")

    # Concatenate all quarters
    full_year = np.concatenate(all_quarters, axis=0)  # (365, 30, 721, 1440)
    print(f"\nFull year shape: {full_year.shape}")

    # Remove old partial zarr and write full year
    if out_zarr.exists():
        import shutil
        shutil.rmtree(out_zarr)
        print(f"  Removed old {out_zarr}")

    z = zarr.open(
        str(out_zarr), mode="w",
        shape=full_year.shape,
        chunks=(1, 30, NLAT, NLON),
        dtype="float32",
        compressor=COMPRESSOR,
        zarr_format=2,
    )
    z[:] = full_year
    z.attrs["year"] = 2025
    z.attrs["description"] = "ERA5 global daily 2025, normalized, 30ch, full year"
    print(f"  Saved: {out_zarr} {full_year.shape}")
    print("\nDone!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
