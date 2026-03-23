#!/usr/bin/env python3
"""
Download CNEMC PM2.5 station data — same output format as OpenAQ 2025.
Usage: python download_cnemc_v2.py <year>

Source: quotsoft.net/air (daily CSV per day, hourly values per station)
Output: /scratch/project_462001140/ammar/eccv/aria/data/finetune/cnemc_{year}_pm25.npz

Format matches OpenAQ 2025:
  lats, lons, values (float32), timestamps (float64), location_ids (str),
  n_records (int64), n_stations (int64), year (int64)
"""
import urllib.request, csv, io, sys, time, json
import numpy as np
from datetime import date, timedelta, datetime
from pathlib import Path
from collections import defaultdict

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
BASE_DIR = Path("/scratch/project_462001140/ammar/eccv/aria/data/finetune")
OUTPUT_FILE = BASE_DIR / f"cnemc_{YEAR}_pm25.npz"
BASE_URL = "https://quotsoft.net/air/data/china_sites_{date}.csv"

# Load station coords from 2025 data
COORDS_FILE = Path("/scratch/project_462000640/ammar/cnemc_2025/station_coords.json")

BASE_DIR.mkdir(parents=True, exist_ok=True)


def load_station_coords():
    coords = {}
    if COORDS_FILE.exists():
        with open(COORDS_FILE) as f:
            coords = json.load(f)
        print(f"  Loaded {len(coords)} station coords from 2025 mapping")
    return coords


def download_day(d):
    url = BASE_URL.format(date=d.strftime("%Y%m%d"))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            content = r.read().decode("utf-8", errors="replace")
        return content
    except Exception:
        return None


def parse_day(content, d):
    """Parse daily CSV → list of (station_id, timestamp_float, pm25_value)"""
    records = []
    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if len(rows) < 3:
            return records

        header = rows[0]
        station_ids = header[3:]  # First 3 cols = date, hour, type

        # Find PM2.5 rows
        pm25_rows = [r for r in rows[1:] if len(r) > 2 and "PM2.5" in r[2]]

        # Daily average per station
        day_vals = defaultdict(list)
        for row in pm25_rows:
            for i, sid in enumerate(station_ids):
                col = i + 3
                if col >= len(row):
                    continue
                try:
                    val = float(row[col])
                    if 0 <= val <= 2000:
                        day_vals[sid.strip()].append(val)
                except (ValueError, TypeError):
                    continue

        # Convert to records with timestamp (noon UTC of that day)
        ts = datetime(d.year, d.month, d.day, 12, 0, 0).timestamp()
        for sid, vals in day_vals.items():
            if vals:
                records.append((sid, ts, float(np.mean(vals))))
    except Exception:
        pass
    return records


def main():
    print(f"=== CNEMC {YEAR} PM2.5 Download (v2 — unified format) ===")
    sys.stdout.flush()

    if OUTPUT_FILE.exists():
        print(f"Already exists: {OUTPUT_FILE}")
        return

    coords = load_station_coords()

    start = date(YEAR, 1, 1)
    end = date(YEAR, 12, 31)
    d = start
    all_sids, all_ts, all_vals = [], [], []
    n_days, n_ok = 0, 0

    while d <= end:
        n_days += 1
        content = download_day(d)
        if content:
            recs = parse_day(content, d)
            if recs:
                n_ok += 1
                for sid, ts, val in recs:
                    all_sids.append(sid)
                    all_ts.append(ts)
                    all_vals.append(val)
        if n_days % 30 == 0:
            print(f"  Day {n_days}/365 ({d}): {n_ok} OK, {len(all_vals):,} records", flush=True)
        d += timedelta(days=1)
        time.sleep(0.2)

    print(f"\nTotal: {len(all_vals):,} records from {n_ok}/{n_days} days")

    unique_sids = sorted(set(all_sids))
    print(f"Unique stations: {len(unique_sids)}")

    # Build lat/lon from coords mapping
    lats = np.array([coords.get(s, {}).get("lat", np.nan) for s in all_sids], dtype=np.float32)
    lons = np.array([coords.get(s, {}).get("lon", np.nan) for s in all_sids], dtype=np.float32)

    # Warn about missing coords
    n_nan = np.isnan(lats).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan}/{len(lats)} records have no coordinates")

    np.savez_compressed(
        str(OUTPUT_FILE),
        lats=lats,
        lons=lons,
        values=np.array(all_vals, dtype=np.float32),
        timestamps=np.array(all_ts, dtype=np.float64),
        location_ids=np.array(all_sids, dtype=str),
        n_records=np.int64(len(all_vals)),
        n_stations=np.int64(len(unique_sids)),
        year=np.int64(YEAR),
    )
    print(f"Saved: {OUTPUT_FILE}")
    print(f"  {len(unique_sids):,} stations, {len(all_vals):,} records")


if __name__ == "__main__":
    main()
