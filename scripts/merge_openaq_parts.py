#!/usr/bin/env python3
"""
Merge OpenAQ yearly parts into a single npz matching the 2025 format.
Usage: python merge_openaq_parts.py <year>

Input:  /scratch/project_462000640/ammar/openaq_{year}_parts/part_*.npz
Output: /scratch/project_462001140/ammar/eccv/aria/data/finetune/openaq_{year}_pm25.npz

Converts from 2023-2024 format (dates str, pm25, station_ids)
       to 2025 format (timestamps float64, values, location_ids)
"""
import sys, numpy as np
from pathlib import Path
from datetime import datetime

YEAR = int(sys.argv[1])
PARTS_DIR = Path(f"/scratch/project_462000640/ammar/openaq_{YEAR}_parts")
OUT_DIR = Path("/scratch/project_462001140/ammar/eccv/aria/data/finetune")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"openaq_{YEAR}_pm25.npz"

parts = sorted(PARTS_DIR.glob("part_*.npz"))
print(f"Merging {len(parts)} parts for OpenAQ {YEAR}")

all_lats, all_lons, all_vals, all_ts, all_ids = [], [], [], [], []

for p in parts:
    d = np.load(str(p), allow_pickle=True)
    n = len(d["pm25"])
    all_lats.append(d["lats"])
    all_lons.append(d["lons"])
    all_vals.append(d["pm25"])
    all_ids.append(d["station_ids"])

    # Convert date strings "YYYY-MM-DD" → Unix timestamps
    timestamps = np.array([
        datetime.strptime(s, "%Y-%m-%d").timestamp()
        for s in d["dates"]
    ], dtype=np.float64)
    all_ts.append(timestamps)
    print(f"  {p.name}: {n:,} records")

lats = np.concatenate(all_lats)
lons = np.concatenate(all_lons)
values = np.concatenate(all_vals)
timestamps = np.concatenate(all_ts)
location_ids = np.concatenate(all_ids)

n_records = len(values)
n_stations = len(set(location_ids.tolist()))

np.savez_compressed(
    str(OUT_FILE),
    lats=lats.astype(np.float32),
    lons=lons.astype(np.float32),
    values=values.astype(np.float32),
    timestamps=timestamps,
    location_ids=location_ids,
    n_records=np.int64(n_records),
    n_stations=np.int64(n_stations),
    year=np.int64(YEAR),
)
print(f"\nSaved: {OUT_FILE}")
print(f"  {n_stations:,} stations, {n_records:,} records")
