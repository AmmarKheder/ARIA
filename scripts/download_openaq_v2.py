#!/usr/bin/env python3
"""
Download OpenAQ PM2.5 from S3 — same method as 2025 (raw hourly/minute data).
Usage: python download_openaq_v2.py <year>

Output: /scratch/project_462001140/ammar/eccv/aria/data/finetune/openaq_{year}_pm25.npz
Parts:  /scratch/project_462001140/ammar/eccv/aria/data/finetune/openaq_{year}_parts/
"""
import boto3, io, gzip, csv
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime
import os, sys, gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
BASE_DIR = Path("/scratch/project_462001140/ammar/eccv/aria/data/finetune")
OUTPUT_FILE = BASE_DIR / f"openaq_{YEAR}_pm25.npz"
PARTIAL_DIR = BASE_DIR / f"openaq_{YEAR}_parts"
BUCKET = "openaq-data-archive"
MAX_WORKERS = 32
BATCH_SIZE = 2000

BASE_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_DIR.mkdir(parents=True, exist_ok=True)


def get_s3():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def list_all_location_ids():
    print(f"Listing all location IDs from S3...")
    sys.stdout.flush()
    s3 = get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    loc_ids = []
    pages = paginator.paginate(
        Bucket=BUCKET,
        Prefix="records/csv.gz/locationid=",
        Delimiter="/",
        PaginationConfig={"PageSize": 1000},
    )
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            p = cp["Prefix"]
            loc_id = p.split("locationid=")[1].rstrip("/")
            try:
                loc_ids.append(int(loc_id))
            except ValueError:
                pass
    print(f"  Found {len(loc_ids):,} total location IDs")
    sys.stdout.flush()
    return loc_ids


def has_year_data(loc_id):
    s3 = get_s3()
    resp = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=f"records/csv.gz/locationid={loc_id}/year={YEAR}/",
        MaxKeys=1,
    )
    return bool(resp.get("Contents"))


def download_location(loc_id):
    """Download all PM2.5 data for a single location for YEAR."""
    s3 = get_s3()
    records = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(
        Bucket=BUCKET,
        Prefix=f"records/csv.gz/locationid={loc_id}/year={YEAR}/",
    )
    for page in pages:
        for obj in page.get("Contents", []):
            try:
                content = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
                with gzip.open(io.BytesIO(content), "rt", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        param = row.get("parameter", "").lower()
                        if param not in ("pm25", "pm2.5"):
                            continue
                        try:
                            val = float(row["value"])
                            if val < 0 or val > 2000:
                                continue
                            lat = float(row["lat"])
                            lon = float(row["lon"])
                            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                                continue
                            ts = datetime.fromisoformat(row["datetime"][:19]).timestamp()
                            records.append((lat, lon, val, ts, loc_id))
                        except (ValueError, KeyError):
                            continue
            except Exception:
                continue
    return records


def save_batch(batch_records, batch_idx):
    if not batch_records:
        return
    lats = np.array([r[0] for r in batch_records], dtype=np.float32)
    lons = np.array([r[1] for r in batch_records], dtype=np.float32)
    vals = np.array([r[2] for r in batch_records], dtype=np.float32)
    tss  = np.array([r[3] for r in batch_records], dtype=np.float64)
    lids = np.array([r[4] for r in batch_records], dtype=np.int64)
    path = PARTIAL_DIR / f"part_{batch_idx:04d}.npz"
    np.savez_compressed(path, lats=lats, lons=lons, values=vals, timestamps=tss, location_ids=lids)
    print(f"  Saved batch {batch_idx}: {len(batch_records):,} records → {path.name}")
    sys.stdout.flush()


def merge_parts():
    parts = sorted(PARTIAL_DIR.glob("part_*.npz"))
    print(f"\nMerging {len(parts)} part files...")
    all_lats, all_lons, all_vals, all_tss, all_lids = [], [], [], [], []
    for p in parts:
        d = np.load(p)
        all_lats.append(d["lats"])
        all_lons.append(d["lons"])
        all_vals.append(d["values"])
        all_tss.append(d["timestamps"])
        all_lids.append(d["location_ids"])
    lats = np.concatenate(all_lats)
    lons = np.concatenate(all_lons)
    vals = np.concatenate(all_vals)
    tss  = np.concatenate(all_tss)
    lids = np.concatenate(all_lids)
    n_sta = len(np.unique(lids))
    np.savez_compressed(
        str(OUTPUT_FILE),
        lats=lats, lons=lons, values=vals, timestamps=tss,
        location_ids=lids.astype(str),
        n_records=np.int64(len(vals)),
        n_stations=np.int64(n_sta),
        year=np.int64(YEAR),
    )
    print(f"Saved: {OUTPUT_FILE}")
    print(f"  {n_sta:,} stations, {len(vals):,} records")


def main():
    print("=" * 60)
    print(f"OpenAQ {YEAR} — Full resolution download from S3")
    print("=" * 60)

    # Resume support
    existing_parts = sorted(PARTIAL_DIR.glob("part_*.npz"))
    start_batch = len(existing_parts)
    if start_batch > 0:
        print(f"  Resuming: found {start_batch} existing parts")

    # Phase 1: List all locations
    loc_ids = list_all_location_ids()

    # Phase 2: Filter locations that have data for YEAR
    print(f"\nPhase 1: Scanning {len(loc_ids):,} locations for {YEAR} data ({MAX_WORKERS} threads)...")
    sys.stdout.flush()
    locs_with_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(has_year_data, lid): lid for lid in loc_ids}
        for i, f in enumerate(futures):
            if f.result():
                locs_with_data.append(futures[f])
            if (i + 1) % 10000 == 0:
                print(f"  Scanned {i+1:,}/{len(loc_ids):,} — {len(locs_with_data):,} with {YEAR} data")
                sys.stdout.flush()

    print(f"  {len(locs_with_data):,} locations have {YEAR} data")
    sys.stdout.flush()

    # Phase 3: Download in batches
    skip_locs = start_batch * BATCH_SIZE
    print(f"\nPhase 2: Downloading PM2.5 from {len(locs_with_data):,} locations (batch_size={BATCH_SIZE})...")
    if skip_locs > 0:
        print(f"  Skipping first {skip_locs} locations (already done)")
    sys.stdout.flush()

    for bi in range(start_batch, (len(locs_with_data) + BATCH_SIZE - 1) // BATCH_SIZE):
        batch_locs = locs_with_data[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        print(f"\nBatch {bi} ({bi*BATCH_SIZE}–{bi*BATCH_SIZE+len(batch_locs)-1})...")
        sys.stdout.flush()

        batch_records = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {exe.submit(download_location, lid): lid for lid in batch_locs}
            done = 0
            for f in futures:
                recs = f.result()
                batch_records.extend(recs)
                done += 1
                if done % 200 == 0:
                    print(f"  Batch {bi}: {done}/{len(batch_locs)} locs, {len(batch_records):,} records")
                    sys.stdout.flush()

        save_batch(batch_records, bi)
        del batch_records
        gc.collect()

    # Merge
    merge_parts()
    print("\nDONE.")


if __name__ == "__main__":
    main()
