#!/usr/bin/env python3
"""
Download GHAP PM2.5 Daily (D1K) GLOBAL from Zenodo → Save as Zarr
=================================================================

Source: GlobalHighPM2.5 (Wei et al.)
Resolution: 1 km (0.01°), daily
Coverage: Global (approximately -60°S to 70°N land areas)

GHAP global grid: ~13000 lat × 36000 lon (varies by file)
Output: /scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily/YYYY.zarr

WARNING: Each year ≈ 700 GB uncompressed. With Blosc compression ≈ 100-200 GB/year.
Total for 6 years: ~600 GB - 1.2 TB.
"""
import os
import sys
import zipfile
import tempfile
import urllib.request
from pathlib import Path
from calendar import monthrange

import numpy as np
import zarr
import numcodecs

# Zenodo record IDs
ZENODO_RECORDS = {
    2017: "10801181",
    2018: "10795801",
    2019: "10799037",
    2020: "10800555",
    2021: "10799203",
    2022: "10795662",
}

RAW_DIR = Path("/scratch/project_462001140/ammar/eccv/data/raw/ghap_daily_global")
OUT_DIR = Path("/scratch/project_462001140/ammar/eccv/data/zarr/ghap_global_daily")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=2)


def download_month(year, month):
    """Download one monthly ZIP from Zenodo"""
    record_id = ZENODO_RECORDS[year]
    filename = f"GHAP_PM2.5_D1K_{year}{month:02d}_V1.zip"
    url = f"https://zenodo.org/records/{record_id}/files/{filename}?download=1"
    zip_path = RAW_DIR / filename

    if zip_path.exists() and zip_path.stat().st_size > 1e8:
        print(f"    Already downloaded: {filename} ({zip_path.stat().st_size/1e9:.1f} GB)")
        return zip_path

    print(f"    Downloading {filename}...", end=" ", flush=True)

    def progress(block_num, block_size, total_size):
        if total_size > 0 and block_num % 5000 == 0:
            pct = block_num * block_size / total_size * 100
            print(f"{pct:.0f}%", end=" ", flush=True)

    try:
        urllib.request.urlretrieve(url, str(zip_path), reporthook=progress)
        print(f"OK ({zip_path.stat().st_size/1e9:.1f} GB)")
    except Exception as e:
        print(f"FAILED: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return None

    return zip_path


def read_nc_global(nc_path):
    """Read a single GHAP NetCDF file (full globe, no cropping)"""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    var_name = 'PM2.5' if 'PM2.5' in ds.data_vars else list(ds.data_vars)[0]
    data = ds[var_name].values.squeeze().astype(np.float32)

    # Get grid info on first call
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    ds.close()

    # NaN → 0
    data = np.nan_to_num(data, nan=0.0)
    return data, lats, lons


def process_year(year):
    """Download + process one year of daily GHAP data (global)"""
    print(f"\n{'='*60}")
    print(f"GHAP D1K GLOBAL {year}")
    print(f"{'='*60}")

    expected_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    output_path = OUT_DIR / f"{year}.zarr"

    if output_path.exists():
        try:
            z = zarr.open(str(output_path), mode='r')
            if z.shape[0] >= expected_days:
                print(f"  Already complete: {z.shape}")
                return
        except:
            pass

    # We don't know the grid size until we read the first file
    grid_shape = None
    zarr_store = None
    day_idx = 0

    for month in range(1, 13):
        print(f"\n  Month {month:02d}:")
        n_days = monthrange(year, month)[1]

        zip_path = download_month(year, month)
        if zip_path is None:
            print(f"    Skipping month {month} (download failed)")
            if zarr_store is not None:
                # Fill with zeros
                for d in range(n_days):
                    zarr_store[day_idx] = np.zeros(grid_shape, dtype=np.float32)
                    day_idx += 1
            else:
                day_idx += n_days
            continue

        print(f"    Extracting...", end=" ", flush=True)

        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                nc_members = sorted([m for m in zf.namelist() if m.endswith('.nc')])
                print(f"{len(nc_members)} files")

                for i, nc_file in enumerate(nc_members):
                    if i >= n_days:
                        break

                    zf.extract(nc_file, tmpdir)
                    nc_path = os.path.join(tmpdir, nc_file)

                    try:
                        day_data, lats, lons = read_nc_global(nc_path)

                        # Initialize zarr on first successful read
                        if zarr_store is None:
                            grid_shape = day_data.shape
                            print(f"\n    Global grid: {grid_shape} (lat={lats.min():.2f} to {lats.max():.2f}, lon={lons.min():.2f} to {lons.max():.2f})")
                            full_shape = (expected_days, grid_shape[0], grid_shape[1])
                            print(f"    Full zarr: {full_shape} = {np.prod(full_shape)*4/1e9:.1f} GB uncompressed")

                            zarr_store = zarr.open(
                                store=str(output_path),
                                mode='w',
                                shape=full_shape,
                                chunks=(1, 512, 672),
                                dtype='float32',
                                compressor=COMPRESSOR,
                                zarr_format=2,
                            )
                            zarr_store.attrs['year'] = year
                            zarr_store.attrs['source'] = 'GHAP GlobalHighPM2.5 D1K'
                            zarr_store.attrs['units'] = 'ug/m3'
                            zarr_store.attrs['lat_min'] = float(lats.min())
                            zarr_store.attrs['lat_max'] = float(lats.max())
                            zarr_store.attrs['lon_min'] = float(lons.min())
                            zarr_store.attrs['lon_max'] = float(lons.max())

                            # Save coordinates once
                            coords_path = OUT_DIR / "ghap_global_coords.npz"
                            if not coords_path.exists():
                                np.savez(coords_path, lat=lats, lon=lons)
                                print(f"    Saved coords: {coords_path}")

                            # Fill any skipped days with zeros
                            for skip_d in range(day_idx):
                                zarr_store[skip_d] = np.zeros(grid_shape, dtype=np.float32)

                        zarr_store[day_idx] = day_data
                    except Exception as e:
                        print(f"      Error {nc_file}: {e}")
                        if zarr_store is not None:
                            zarr_store[day_idx] = np.zeros(grid_shape, dtype=np.float32)

                    os.remove(nc_path)
                    day_idx += 1

        # Fill remaining days if month had fewer files
        while day_idx < sum(monthrange(year, m)[1] for m in range(1, month + 1)):
            if zarr_store is not None:
                zarr_store[day_idx] = np.zeros(grid_shape, dtype=np.float32)
            day_idx += 1

        print(f"    Day {day_idx}/{expected_days} done, range: [0, ?]")

        # Delete ZIP to save space
        zip_path.unlink()
        print(f"    Deleted ZIP")

    zarr_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e9
    print(f"\n  Saved: {output_path}")
    print(f"  Shape: {zarr_store.shape}")
    print(f"  Zarr size: {zarr_size:.1f} GB")


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=" * 60)
    print("GHAP PM2.5 DAILY (D1K) GLOBAL → ZARR")
    print("=" * 60)
    print(f"Output: {OUT_DIR}")
    print(f"No cropping — full global grid")

    if year:
        process_year(year)
    else:
        for y in range(2017, 2023):
            process_year(y)

    print("\nDONE!")


if __name__ == "__main__":
    main()
