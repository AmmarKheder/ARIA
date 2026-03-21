#!/usr/bin/env python3
"""
Download CAMS EAC4 Global Reanalysis (2017-2022).
Dataset: cams-global-reanalysis-eac4
Resolution: 0.75° (~80 km) global
Variables: NO2, O3, SO2, CO, PM10 (5 pollutants for global branch)

Output: /scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily/YYYY.zarr
        Zarr group with keys: no2, o3, so2, co, pm10
        Each: (365, 241, 480) at 0.75° global

Note: CAMS EAC4 is available 2003-2022. For PM2.5 we use GHAP as target, not CAMS.
"""
import sys
import numpy as np
import xarray as xr
import zarr
import numcodecs
from pathlib import Path
import cdsapi
import os

OUTPUT_DIR = Path("/scratch/project_462001140/ammar/eccv/data/zarr/cams_global_daily")
TMP_DIR    = Path(f"/tmp/cams_global_download_{os.getpid()}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

# CAMS EAC4 variables
# 3D vars need model_level="60" (surface), surface-only vars don't
CAMS_VARS_3D = {
    "nitrogen_dioxide":              "no2",
    "ozone":                         "o3",
    "sulphur_dioxide":               "so2",
    "carbon_monoxide":               "co",
}
CAMS_VARS_SFC = {
    "particulate_matter_10um":       "pm10",
}

# Also download PM2.5 separately (useful as CAMS baseline / coarse input)
PM25_VAR = "particulate_matter_2.5um"

COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)

# Normalization (from Europe training — will recompute global stats)
CAMS_NORM = {
    "no2":  (2.15,   3.52),    # µg/m³
    "o3":   (71.75,  18.18),
    "so2":  (0.85,   1.69),
    "co":   (140.8,  36.7),
    "pm10": (14.0,   13.7),
}


def download_cams_eac4(client, year, variable, months):
    """Download CAMS EAC4 global reanalysis for one variable via ADS API."""
    out = TMP_DIR / f"cams_eac4_{variable}_{year}_{months[0]}-{months[-1]}.nc"
    if out.exists() and out.stat().st_size > 1e5:
        print(f"    Cache: {out.name} ({out.stat().st_size/1e6:.0f} MB)")
        return out

    import calendar
    # Build date range: YYYY-MM-01/YYYY-MM-end
    m_start = int(months[0])
    m_end = int(months[-1])
    d_end = calendar.monthrange(year, m_end)[1]
    date_range = f"{year}-{m_start:02d}-01/{year}-{m_end:02d}-{d_end:02d}"

    # Determine if 3D variable (needs model_level) or surface-only
    is_3d = variable in CAMS_VARS_3D

    print(f"    Downloading {variable} {year} {date_range} ({'3D→L60' if is_3d else 'sfc'})...", end=" ", flush=True)
    request = {
        "variable":     variable,
        "date":         date_range,
        "time":         ["00:00", "06:00", "12:00", "18:00"],
        "data_format":  "netcdf_zip",
    }
    if is_3d:
        request["model_level"] = "60"  # lowest level = surface
    try:
        client.retrieve("cams-global-reanalysis-eac4", request, str(out))
        # If zip, extract
        if out.suffix == '.nc' and out.stat().st_size > 0:
            import zipfile
            if zipfile.is_zipfile(str(out)):
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    with zipfile.ZipFile(str(out), 'r') as zf:
                        zf.extractall(td)
                    nc_files = list(Path(td).glob("*.nc"))
                    if nc_files:
                        import shutil
                        shutil.move(str(nc_files[0]), str(out))
        print(f"OK ({out.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        # Retry with netcdf format
        request2 = {
            "variable":     variable,
            "date":         date_range,
            "time":         ["00:00", "06:00", "12:00", "18:00"],
            "data_format":  "netcdf",
        }
        if is_3d:
            request2["model_level"] = "60"
        try:
            client.retrieve("cams-global-reanalysis-eac4", request2, str(out))
            print(f"OK (retry) ({out.stat().st_size/1e6:.0f} MB)")
        except Exception as e2:
            print(f"FAILED again: {e2}")
            return None
    return out


def download_cams_eac4_monthly(client, year, variable, months):
    """Download using monthly single-level approach."""
    out = TMP_DIR / f"cams_eac4_sfc_{variable}_{year}_{months[0]}-{months[-1]}.nc"
    if out.exists() and out.stat().st_size > 1e5:
        print(f"    Cache: {out.name} ({out.stat().st_size/1e6:.0f} MB)")
        return out

    print(f"    Downloading {variable} (single-level) {year} months {months}...", end=" ", flush=True)
    try:
        client.retrieve("cams-global-reanalysis-eac4-monthly", {
            "variable":     variable,
            "year":         str(year),
            "month":        months,
            "product_type": "monthly_mean",
            "format":       "netcdf",
        }, str(out))
        print(f"OK ({out.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")
        return None
    return out


def nc_to_daily(nc_path):
    """Load NetCDF, resample to daily mean, return numpy array."""
    ds = xr.open_dataset(str(nc_path))
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # Handle time dimension name
    if "valid_time" in da.dims:
        da = da.rename({"valid_time": "time"})

    # Drop level dimension if present
    for dim in ["level", "model_level", "pressure_level"]:
        if dim in da.dims:
            da = da.isel({dim: 0})

    dm = da.resample(time="1D").mean("time")
    data = dm.values.astype(np.float32)
    ds.close()
    return data


def process_year(year):
    print(f"\n{'='*60}")
    print(f"CAMS EAC4 GLOBAL {year}")
    print(f"{'='*60}")

    expected_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    out_path = OUTPUT_DIR / f"{year}.zarr"

    if out_path.exists():
        try:
            z = zarr.open(str(out_path), mode='r')
            if 'no2' in z and z['no2'].shape[0] >= expected_days:
                print(f"Already complete: no2={z['no2'].shape}")
                return
        except:
            pass

    # CAMS is on ADS (Atmosphere Data Store), NOT CDS
    client = cdsapi.Client(
        url="https://ads.atmosphere.copernicus.eu/api",
        key="88f0dfc1-3328-4392-8b73-adaddc2931c5",
    )

    # Download each variable in 2-month batches, then concatenate
    month_batches = [
        ["01", "02"], ["03", "04"], ["05", "06"],
        ["07", "08"], ["09", "10"], ["11", "12"],
    ]

    all_vars = {}  # var_name → list of arrays
    ALL_CAMS_VARS = {**CAMS_VARS_3D, **CAMS_VARS_SFC}

    for cds_name, short_name in ALL_CAMS_VARS.items():
        print(f"\n  Variable: {short_name} ({cds_name})")
        var_data = []

        for batch in month_batches:
            nc_path = download_cams_eac4(client, year, cds_name, batch)
            if nc_path is None:
                print(f"    WARNING: Failed batch {batch}, will try monthly fallback")
                nc_path = download_cams_eac4_monthly(client, year, cds_name, batch)

            if nc_path is not None:
                data = nc_to_daily(nc_path)
                var_data.append(data)
                print(f"    Batch {batch}: {data.shape}")
                nc_path.unlink()
            else:
                print(f"    WARNING: Batch {batch} completely failed")

        if var_data:
            full = np.concatenate(var_data, axis=0)[:expected_days]
            all_vars[short_name] = full
            print(f"  {short_name} final: {full.shape}")

    # Also download PM2.5
    print(f"\n  Variable: pm25 ({PM25_VAR})")
    pm25_data = []
    for batch in month_batches:
        nc_path = download_cams_eac4(client, year, PM25_VAR, batch)
        if nc_path is not None:
            data = nc_to_daily(nc_path)
            pm25_data.append(data)
            nc_path.unlink()
    if pm25_data:
        all_vars["pm25"] = np.concatenate(pm25_data, axis=0)[:expected_days]
        print(f"  pm25 final: {all_vars['pm25'].shape}")

    # Save as zarr group
    print(f"\nSaving to {out_path}...")
    root = zarr.open_group(str(out_path), mode='w', zarr_format=2)
    for var_name, data in all_vars.items():
        root.create_array(var_name, data=data, chunks=(1, data.shape[1], data.shape[2]),
                          compressor=COMPRESSOR, overwrite=True)
        print(f"  {var_name}: {data.shape}")

    root.attrs['year'] = year
    root.attrs['source'] = 'CAMS EAC4 global reanalysis'
    root.attrs['units'] = 'kg/m3 (raw CAMS units)'

    zarr_size = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e9
    print(f"  Zarr size: {zarr_size:.2f} GB")


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=" * 60)
    print("CAMS EAC4 GLOBAL REANALYSIS → ZARR")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")

    if year:
        process_year(year)
    else:
        for y in range(2017, 2023):
            process_year(y)

    print("\nDONE!")


if __name__ == "__main__":
    main()
