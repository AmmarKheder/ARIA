#!/usr/bin/env python3
"""
Download GMTED2010 global elevation at 7.5 arc-second (~250m) → Zarr
=====================================================================

Source: Zenodo (cdholmes/GMTED2010-netcdf)
Resolution: 7.5 arc-seconds ≈ 0.00208° ≈ 250m
Coverage: Global (-90 to 84°N, -180 to 180°E)

Output:
  /scratch/.../elevation/gmted2010_global.zarr      — hi-res (~250m)
  /scratch/.../elevation/elevation_global_025deg.zarr — coarse (ERA5 grid, 0.25°)
"""
import os
import numpy as np
import xarray as xr
import zarr
import numcodecs
from pathlib import Path
from scipy.ndimage import zoom

RAW_DIR = Path("/scratch/project_462001140/ammar/eccv/data/raw/elevation")
OUT_DIR = Path("/scratch/project_462001140/ammar/eccv/data/zarr/elevation")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://zenodo.org/records/14537811/files/GMTED2010_mean_7p5arcsec.nc4?download=1"
NC_FILE = RAW_DIR / "GMTED2010_mean_7p5arcsec.nc4"

COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=2)


def download():
    """Download GMTED2010 from Zenodo"""
    if NC_FILE.exists() and NC_FILE.stat().st_size > 2e9:
        print(f"Already downloaded: {NC_FILE} ({NC_FILE.stat().st_size/1e9:.1f} GB)")
        return

    print(f"Downloading GMTED2010 7.5 arc-second (~2.8 GB)...")
    import urllib.request

    def report(block_num, block_size, total_size):
        if total_size > 0 and block_num % 500 == 0:
            pct = block_num * block_size / total_size * 100
            print(f"  {pct:.0f}%", flush=True)

    urllib.request.urlretrieve(URL, str(NC_FILE), reporthook=report)
    print(f"Downloaded: {NC_FILE.stat().st_size/1e9:.1f} GB")


def extract_global_hires():
    """Save full global elevation as Zarr (hi-res ~250m)"""
    output_path = OUT_DIR / "gmted2010_global.zarr"
    if output_path.exists():
        z = zarr.open(str(output_path), mode='r')
        if hasattr(z, 'shape') and z.shape[0] > 1000:
            print(f"Hi-res already exists: {z.shape}")
            return z.shape

    print("\nLoading global elevation...")
    ds = xr.open_dataset(NC_FILE)
    var_name = list(ds.data_vars)[0]
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    print(f"  Global shape: {ds[var_name].shape}")
    print(f"  Lat: {lats.min():.2f} to {lats.max():.2f}")
    print(f"  Lon: {lons.min():.2f} to {lons.max():.2f}")

    # Save in chunks to avoid memory issues
    full_shape = ds[var_name].shape
    print(f"  Saving {full_shape} to zarr (chunked write)...")

    store = zarr.open(
        store=str(output_path),
        mode='w',
        shape=full_shape,
        chunks=(512, 512),
        dtype='float32',
        compressor=COMPRESSOR,
        zarr_format=2,
    )

    # Write in latitude bands
    band_size = 512
    for i in range(0, full_shape[0], band_size):
        end = min(i + band_size, full_shape[0])
        band = ds[var_name].isel(**{lat_name: slice(i, end)}).values.squeeze().astype(np.float32)
        band = np.nan_to_num(band, nan=0.0)
        store[i:end] = band
        if (i // band_size) % 10 == 0:
            print(f"    Written rows {i}-{end} / {full_shape[0]}")

    store.attrs['lat_min'] = float(lats.min())
    store.attrs['lat_max'] = float(lats.max())
    store.attrs['lon_min'] = float(lons.min())
    store.attrs['lon_max'] = float(lons.max())
    store.attrs['resolution'] = '7.5 arc-seconds (~250m)'
    store.attrs['source'] = 'GMTED2010 mean elevation'
    store.attrs['units'] = 'meters'

    # Save coordinates
    np.savez(OUT_DIR / "gmted2010_global_coords.npz",
             lat=lats.astype(np.float64),
             lon=lons.astype(np.float64))

    ds.close()
    zarr_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e9
    print(f"  Saved: {output_path} ({zarr_size:.1f} GB)")
    return full_shape


def create_coarse_elevation():
    """Create ERA5-grid (0.25°) coarse elevation with mean + std"""
    output_path = OUT_DIR / "elevation_global_025deg.zarr"
    if output_path.exists():
        try:
            z = zarr.open(str(output_path), mode='r')
            if z.shape[0] == 2:
                print(f"Coarse already exists: {z.shape}")
                return
        except:
            pass

    print("\nCreating coarse elevation at 0.25° (ERA5 grid)...")

    # ERA5 global grid
    era5_lats = np.arange(90, -90.25, -0.25)   # 721 points
    era5_lons = np.arange(0, 360, 0.25)         # 1440 points (0-359.75)

    hires = zarr.open(str(OUT_DIR / "gmted2010_global.zarr"), mode='r')
    coords = np.load(OUT_DIR / "gmted2010_global_coords.npz")
    hr_lats = coords['lat']
    hr_lons = coords['lon']

    # Resolution ratio: 0.25° / 0.00208° ≈ 120 pixels per ERA5 cell
    n_per_cell = int(round(0.25 / abs(hr_lats[1] - hr_lats[0])))
    print(f"  Hi-res pixels per ERA5 cell: ~{n_per_cell}")

    coarse_mean = np.zeros((len(era5_lats), len(era5_lons)), dtype=np.float32)
    coarse_std  = np.zeros((len(era5_lats), len(era5_lons)), dtype=np.float32)

    # Process row by row
    for j, lat_center in enumerate(era5_lats):
        lat_lo = lat_center - 0.125
        lat_hi = lat_center + 0.125

        # Find hi-res indices for this latitude band
        if hr_lats[0] > hr_lats[-1]:  # descending
            i_start = np.searchsorted(-hr_lats, -lat_hi)
            i_end = np.searchsorted(-hr_lats, -lat_lo)
        else:
            i_start = np.searchsorted(hr_lats, lat_lo)
            i_end = np.searchsorted(hr_lats, lat_hi)

        if i_start >= i_end or i_start >= hires.shape[0]:
            continue

        band = np.array(hires[i_start:i_end])  # (n_rows, full_lon)

        for k, lon_center in enumerate(era5_lons):
            lon_lo = lon_center - 0.125
            lon_hi = lon_center + 0.125

            # Handle lon wrapping
            j_start = np.searchsorted(hr_lons, lon_lo)
            j_end = np.searchsorted(hr_lons, lon_hi)

            if j_start < j_end and j_start < band.shape[1]:
                cell = band[:, j_start:min(j_end, band.shape[1])]
                if cell.size > 0:
                    coarse_mean[j, k] = cell.mean()
                    coarse_std[j, k] = cell.std()

        if j % 50 == 0:
            print(f"    Row {j}/{len(era5_lats)} (lat={lat_center:.2f})")

    # Stack: (2, 721, 1440)
    coarse = np.stack([coarse_mean, coarse_std], axis=0)
    print(f"  Coarse shape: {coarse.shape}")
    print(f"  Mean elev range: [{coarse_mean.min():.0f}, {coarse_mean.max():.0f}] m")

    store = zarr.open(
        store=str(output_path),
        mode='w',
        shape=coarse.shape,
        chunks=(2, 721, 1440),
        dtype='float32',
        compressor=COMPRESSOR,
        zarr_format=2,
    )
    store[:] = coarse
    store.attrs['channels'] = ['mean_elevation', 'std_elevation']
    store.attrs['resolution'] = '0.25 deg (ERA5 grid)'
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("GMTED2010 GLOBAL ELEVATION")
    print("=" * 60)

    download()
    extract_global_hires()
    create_coarse_elevation()

    print("\nDONE!")


if __name__ == "__main__":
    main()
