#!/usr/bin/env python3
"""
Build global road density proxy at 0.01° (~1 km) resolution.
Grid: 18000 × 36000 (lat 90°N→90°S, lon 180°W→180°E)

Pipeline:
  1. Download Geofabrik continental OSM PBF files
  2. Parse roads with osmium (weighted by highway type)
  3. Rasterize to 18000×36000 using Bresenham-style line drawing
  4. Gap-fill sparse areas with Microsoft Global Roads Detections
  5. log1p normalize → [0,1] via p99 scale
  6. Write road_density to emission_proxies.zarr

Output: /scratch/project_462001140/ammar/eccv/aria/data/pretrain/emission_proxies.zarr
        adds 'road_density' array of shape (18000, 36000), float32 in [0,1]
"""

import os
import sys
import math
import time
import numpy as np
import zarr
import numcodecs
import urllib.request
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
GRID_H = 18000
GRID_W = 36000
LAT_NORTH = 90.0
LON_WEST  = -180.0
RES       = 0.01  # degrees per pixel

WORK_DIR   = Path("/scratch/project_462001136/ammar/eccv/data/raw/osm_roads")
OUT_ZARR   = Path("/scratch/project_462001140/ammar/eccv/aria/data/pretrain/emission_proxies.zarr")
COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)

# Geofabrik continental PBF files (~5-20 GB each)
GEOFABRIK_URLS = {
    "africa":        "https://download.geofabrik.de/africa-latest.osm.pbf",
    "asia":          "https://download.geofabrik.de/asia-latest.osm.pbf",
    "australia-oc":  "https://download.geofabrik.de/australia-oceania-latest.osm.pbf",
    "central-am":    "https://download.geofabrik.de/central-america-latest.osm.pbf",
    "europe":        "https://download.geofabrik.de/europe-latest.osm.pbf",
    "north-am":      "https://download.geofabrik.de/north-america-latest.osm.pbf",
    "south-am":      "https://download.geofabrik.de/south-america-latest.osm.pbf",
    "russia":        "https://download.geofabrik.de/russia-latest.osm.pbf",
}

# Road type → weight (importance / traffic proxy)
HIGHWAY_WEIGHTS = {
    "motorway":        8.0,
    "trunk":           6.0,
    "primary":         5.0,
    "motorway_link":   4.0,
    "trunk_link":      4.0,
    "primary_link":    3.5,
    "secondary":       3.0,
    "secondary_link":  2.5,
    "tertiary":        2.0,
    "tertiary_link":   1.5,
    "unclassified":    1.0,
    "residential":     0.8,
    "living_street":   0.5,
    "service":         0.4,
    "track":           0.2,
    "path":            0.1,
    "footway":         0.05,
    "cycleway":        0.05,
    "road":            1.0,
}
INCLUDE_HIGHWAY = set(HIGHWAY_WEIGHTS.keys())

# Accumulation tiles to avoid OOM: process in lat strips of 600 rows = 6°
TILE_ROWS = 600   # ~6 GB of float32 per strip


# ── Utility ────────────────────────────────────────────────────────────────────

def latlon_to_ij(lat, lon):
    """Convert (lat, lon) → (row, col) in the global grid (float)."""
    row = (LAT_NORTH - lat) / RES
    col = (lon - LON_WEST) / RES
    return row, col


def download_with_progress(url: str, dst: Path):
    """Download url to dst with simple progress reporting."""
    if dst.exists():
        print(f"  [skip] {dst.name} already exists ({dst.stat().st_size//1024//1024} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    print(f"  Downloading {dst.name} ...", flush=True)
    t0 = time.time()
    def progress(count, bs, total):
        if total > 0 and count % 1000 == 0:
            pct = count * bs / total * 100
            mb  = count * bs / 1e6
            print(f"    {pct:.1f}%  ({mb:.0f} MB)  {time.time()-t0:.0f}s", flush=True)
    urllib.request.urlretrieve(url, str(tmp), reporthook=progress)
    tmp.rename(dst)
    print(f"  Done: {dst.stat().st_size//1024//1024} MB  in {time.time()-t0:.0f}s")


# ── Rasterize one PBF ──────────────────────────────────────────────────────────

def bresenham_line(r0, c0, r1, c1):
    """
    Yield (row, col) integer pixel indices along the line from (r0,c0) to (r1,c1).
    Uses Bresenham's algorithm adapted for floating-point input.
    """
    r0, c0, r1, c1 = int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    if dr == 0 and dc == 0:
        yield r0, c0
        return
    if dr >= dc:
        err = dr // 2
        c = c0
        for r in range(r0, r1 + sr, sr):
            yield r, c
            err -= dc
            if err < 0:
                c += sc
                err += dr
    else:
        err = dc // 2
        r = r0
        for c in range(c0, c1 + sc, sc):
            yield r, c
            err -= dr
            if err < 0:
                r += sr
                err += dc


def rasterize_pbf(pbf_path: Path, density: np.ndarray):
    """
    Parse pbf_path with osmium, accumulate road weight into density[H,W].
    density is modified in-place.
    """
    try:
        import osmium
    except ImportError:
        print("ERROR: osmium not found. Run: pip install osmium", file=sys.stderr)
        sys.exit(1)

    class RoadHandler(osmium.SimpleHandler):
        def __init__(self, grid):
            super().__init__()
            self.grid = grid
            self.n_ways = 0
            self.n_segs = 0

        def way(self, w):
            hw = w.tags.get("highway", None)
            if hw not in INCLUDE_HIGHWAY:
                return
            weight = HIGHWAY_WEIGHTS[hw]
            nodes = w.nodes
            if len(nodes) < 2:
                return
            self.n_ways += 1
            prev_r, prev_c = None, None
            for node in nodes:
                if not node.location.valid():
                    prev_r, prev_c = None, None
                    continue
                lat = node.location.lat
                lon = node.location.lon
                row, col = latlon_to_ij(lat, lon)
                row_i = int(row)
                col_i = int(col) % GRID_W  # wrap longitude
                if not (0 <= row_i < GRID_H):
                    prev_r, prev_c = None, None
                    continue
                if prev_r is not None:
                    for r, c in bresenham_line(prev_r, prev_c, row_i, col_i):
                        if 0 <= r < GRID_H and 0 <= c < GRID_W:
                            self.grid[r, c] += weight
                    self.n_segs += 1
                prev_r, prev_c = row_i, col_i

    print(f"  Parsing {pbf_path.name} ...", flush=True)
    t0 = time.time()
    handler = RoadHandler(density)
    handler.apply_file(str(pbf_path), locations=True, idx="flex_mem")
    dt = time.time() - t0
    print(f"    {handler.n_ways} ways, {handler.n_segs} segments  ({dt:.0f}s)", flush=True)


# ── Microsoft Road Detections gap-fill ────────────────────────────────────────

# Microsoft Global Road Detections (2024) — available as Cloud-Optimized GeoTIFF
# Dataset: https://github.com/microsoft/RoadDetections
# Coverage: ~106 countries (focus on Africa, Central Asia, SE Asia, Pacific)
# We download the composite TIF and add to our grid where OSM is sparse.

MS_ROADS_URLS = [
    # These tiles cover regions with sparse OSM coverage
    # Sub-Saharan Africa: -35°N to 40°N, 20°W to 55°E
    "https://usgseros.blob.core.windows.net/gers-roads-public/tiles/Africa_roads.tif",
    # Central/South Asia: 0°N to 50°N, 50°E to 140°E
    "https://usgseros.blob.core.windows.net/gers-roads-public/tiles/Asia_roads.tif",
]

# Alternative: use the Overture Maps road data (open, 2024) as COG:
# https://overturemaps.org/download/ → transportation theme
# This is simpler: download parquet, convert to grid

def gap_fill_microsoft_roads(density: np.ndarray):
    """
    Add Microsoft Road Detections to cells where OSM density is sparse (< 0.05).
    Uses rasterio to read the COG tiles at the correct resolution.

    If tiles are unavailable, skip gracefully.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_bounds
    except ImportError:
        print("  WARNING: rasterio not available, skipping Microsoft roads gap-fill")
        return

    sparse_mask = density < 0.05  # cells with little/no OSM data
    n_sparse = sparse_mask.sum()
    if n_sparse == 0:
        print("  No sparse cells, skipping Microsoft gap-fill")
        return
    print(f"  Gap-fill: {n_sparse:,} sparse cells ({100*n_sparse/density.size:.1f}%)")

    ms_dir = WORK_DIR / "microsoft_roads"
    ms_dir.mkdir(parents=True, exist_ok=True)

    for url in MS_ROADS_URLS:
        fname = ms_dir / Path(url).name
        try:
            download_with_progress(url, fname)
        except Exception as e:
            print(f"  WARNING: Could not download {fname.name}: {e}")
            continue

        if not fname.exists():
            continue

        try:
            # Read and reproject the MS road tile to our grid
            target_transform = from_bounds(
                left=LON_WEST, bottom=-90.0,
                right=-LON_WEST, top=LAT_NORTH,
                width=GRID_W, height=GRID_H
            )
            with rasterio.open(str(fname)) as src:
                ms_data = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=ms_data,
                    dst_transform=target_transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.average,
                )
            # Normalize MS roads to [0, 1] range (it's binary 0/1 road presence)
            # Add with weight 0.5 to sparse OSM cells
            density[sparse_mask] = np.maximum(
                density[sparse_mask],
                ms_data[sparse_mask] * 0.5
            )
            print(f"  Applied {fname.name}")
        except Exception as e:
            print(f"  WARNING: Could not apply {fname.name}: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download PBF files
    print("=" * 60)
    print("STEP 1: Download Geofabrik OSM PBF files")
    print("=" * 60)
    for name, url in GEOFABRIK_URLS.items():
        dst = WORK_DIR / f"{name}.osm.pbf"
        download_with_progress(url, dst)

    # Step 2: Rasterize all PBFs into a shared density grid
    # Use float32 accumulator. 18000×36000×4 bytes = 2.4 GB — fits in RAM.
    print("\n" + "=" * 60)
    print("STEP 2: Rasterize roads → 18000×36000 density grid")
    print("=" * 60)

    # Try to load existing partial result
    density_cache = WORK_DIR / "density_raw.npy"
    if density_cache.exists():
        print(f"  Loading cached density from {density_cache}")
        density = np.load(str(density_cache))
        assert density.shape == (GRID_H, GRID_W), f"Bad shape: {density.shape}"
    else:
        density = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    processed_flag = WORK_DIR / "processed_pbfs.txt"
    already_done = set()
    if processed_flag.exists():
        already_done = set(processed_flag.read_text().strip().split("\n"))

    for name, url in GEOFABRIK_URLS.items():
        pbf_path = WORK_DIR / f"{name}.osm.pbf"
        if name in already_done:
            print(f"  [skip] {name} already processed")
            continue
        if not pbf_path.exists():
            print(f"  WARNING: {pbf_path.name} not found, skipping")
            continue

        rasterize_pbf(pbf_path, density)

        # Save progress after each continent
        already_done.add(name)
        processed_flag.write_text("\n".join(sorted(already_done)))
        np.save(str(density_cache), density)
        print(f"  Saved progress: max={density.max():.2f}, nonzero={np.count_nonzero(density):,}")

    # Step 3: Microsoft Road Detections gap-fill
    print("\n" + "=" * 60)
    print("STEP 3: Microsoft Road Detections gap-fill")
    print("=" * 60)
    gap_fill_microsoft_roads(density)

    # Step 4: log1p normalize → [0, 1]
    print("\n" + "=" * 60)
    print("STEP 4: Normalize → [0, 1]")
    print("=" * 60)
    density_log = np.log1p(density)
    p99 = float(np.percentile(density_log[density_log > 0], 99))
    print(f"  Raw density: max={density.max():.2f}, mean={density[density>0].mean():.3f}")
    print(f"  log1p p99 = {p99:.4f}")
    density_norm = np.clip(density_log / p99, 0.0, 1.0).astype(np.float32)
    print(f"  Normalized: min={density_norm.min():.3f}, max={density_norm.max():.3f}, "
          f"mean={density_norm[density_norm>0].mean():.4f}")

    # Save normalized cache
    norm_cache = WORK_DIR / "density_norm.npy"
    np.save(str(norm_cache), density_norm)

    # Step 5: Write to zarr
    print("\n" + "=" * 60)
    print(f"STEP 5: Write to {OUT_ZARR}")
    print("=" * 60)
    if not OUT_ZARR.exists():
        print(f"  WARNING: {OUT_ZARR} does not exist, creating new zarr group")

    root = zarr.open_group(str(OUT_ZARR), mode="a", zarr_format=2)

    # Write in tiles to avoid single huge write
    CHUNK_H = 512
    CHUNK_W = 512

    if "road_density" in root:
        print("  Deleting existing road_density array")
        del root["road_density"]

    print(f"  Creating road_density ({GRID_H}, {GRID_W}) float32 ...")
    arr = root.create_dataset(
        "road_density",
        shape=(GRID_H, GRID_W),
        chunks=(CHUNK_H, CHUNK_W),
        dtype=np.float32,
        compressor=COMPRESSOR,
        overwrite=True,
    )

    print("  Writing tiles ...")
    t0 = time.time()
    for r0 in range(0, GRID_H, CHUNK_H * 8):
        r1 = min(r0 + CHUNK_H * 8, GRID_H)
        arr[r0:r1, :] = density_norm[r0:r1, :]
        pct = r1 / GRID_H * 100
        print(f"    {pct:.1f}% ({r1}/{GRID_H} rows)  {time.time()-t0:.0f}s", flush=True)

    print(f"\nDONE. road_density written: shape={arr.shape}, "
          f"p50={np.percentile(density_norm, 50):.4f}, p99={np.percentile(density_norm, 99):.4f}")

    # Verify zarr contents
    print("\nZarr arrays in emission_proxies.zarr:")
    for key in root.keys():
        a = root[key]
        print(f"  {key}: {a.shape} {a.dtype}")

    # Clean up large temp files
    import shutil
    print("\nCleanup: removing raw OSM PBF files ...")
    for name in GEOFABRIK_URLS:
        pbf = WORK_DIR / f"{name}.osm.pbf"
        if pbf.exists():
            print(f"  rm {pbf.name} ({pbf.stat().st_size//1024//1024} MB)")
            pbf.unlink()

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
