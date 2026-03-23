#!/usr/bin/env python3
"""
Zero-shot ARIA inference on 2025 — evaluated against OpenAQ stations.

Setup:
  - Model trained on 2003-2022 (GHAP target)
  - 2025 inputs: ERA5 + CAMS (downloaded Jan-Mar)
  - Local branch "current state": CAMS PM2.5 interpolated to 1km (no GHAP for 2025)
  - Ground truth: OpenAQ 2025 station measurements

Comparison: ARIA vs CAMS direct vs Aurora vs OpenAQ
"""
import sys, os
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from pathlib import Path
from datetime import datetime, timedelta, date
import warnings; warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ARIA_ROOT    = Path("/scratch/project_462001140/ammar/eccv/aria")
CKPT_DIR     = ARIA_ROOT / "checkpoints_global"
ERA5_2025    = Path("/scratch/project_462001140/ammar/eccv/data/zarr/era5_global_daily/2025.zarr")
CAMS_2025_DIR = Path("/scratch/project_462001140/ammar/eccv/data/cams_2025_forecast")
OPENAQ_FILE  = Path("/scratch/project_462000640/ammar/openaq_2025_global_pm25.npz")
OUTPUT_FILE  = Path("/scratch/project_462000640/ammar/zeroshot_2025_comparison.npz")

sys.path.insert(0, str(ARIA_ROOT))

# CAMS normalization (same as training)
CAMS_VARS = ["pm2p5", "no2", "so2", "co", "go3", "pm10"]
CAMS_MEANS_UG = np.array([15., 5., 2., 200., 60., 20.], dtype=np.float32)
CAMS_STDS_UG  = np.array([20., 8., 4., 100., 40., 30.], dtype=np.float32)
GHAP_MEAN, GHAP_STD = 15.0, 20.0

# Grid constants
LAT_NORTH, LON_WEST = 90.0, -180.0
GHAP_RES = 0.01


def find_best_checkpoint():
    """Find the checkpoint with lowest val RMSE."""
    ckpts = list(CKPT_DIR.glob("*.ckpt"))
    if not ckpts:
        return None
    # Sort by val RMSE in filename
    def rmse_from_name(p):
        try:
            return float(str(p).split("val")[-1].replace(".ckpt",""))
        except:
            return 999.
    ckpts_sorted = sorted(ckpts, key=rmse_from_name)
    return ckpts_sorted[0]


def load_model(ckpt_path, device):
    """Load ARIA model from checkpoint."""
    from cran_pm_site.cranpm.training.trainer import CranPMLightning as ARIALightning
    print(f"Loading model from {ckpt_path.name}...")
    sys.stdout.flush()
    model = ARIALightning.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval()
    model.to(device)
    return model


def load_cams_2025_nc(nc_path, day_idx):
    """Load one day from a CAMS 2025 NetCDF file."""
    import xarray as xr
    ds = xr.open_dataset(str(nc_path))
    # Find PM2.5 variable
    pm25_key = None
    for k in ds.data_vars:
        if "pm2p5" in k.lower() or "pm25" in k.lower():
            pm25_key = k
            break
    if pm25_key is None:
        return None
    # Get day_idx (leadtime=0 = analysis/forecast start)
    arr = ds[pm25_key].isel(valid_time=day_idx).values.astype(np.float32)
    # Convert kg/m3 → µg/m3 if needed
    if arr.max() < 0.001:
        arr *= 1e9
    lats = ds.latitude.values
    lons = ds.longitude.values
    return arr, lats, lons


def get_cams_pm25_at_patch(nc_path, day_idx, patch_lat, patch_lon, patch_size_deg=5.12):
    """Extract CAMS PM2.5 for a geographic patch, interpolate to 1km."""
    result = load_cams_2025_nc(nc_path, day_idx)
    if result is None:
        return None
    arr, lats, lons = result

    # Find patch bounds
    half = patch_size_deg / 2
    lat0, lat1 = patch_lat - half, patch_lat + half
    lon0, lon1 = patch_lon - half, patch_lon + half

    # Extract region
    lat_mask = (lats >= lat0) & (lats <= lat1)
    lon_mask = (lons >= lon0) & (lons <= lon1)

    if not (lat_mask.any() and lon_mask.any()):
        return None

    sub = arr[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]

    # Resize to 512×512 (1km patches)
    patch_1km = F.interpolate(
        torch.from_numpy(sub).unsqueeze(0).unsqueeze(0),
        size=(512, 512), mode="bilinear", align_corners=False
    ).squeeze().numpy()

    return patch_1km


def predict_at_stations(model, era5_zarr, cams_nc_path, openaq_data, device,
                        dates, n_max=1000):
    """
    For each OpenAQ station:
      - Find the date's ERA5 + CAMS inputs
      - Run ARIA inference
      - Extract prediction at station lat/lon
    """
    lats   = openaq_data["lats"]
    lons   = openaq_data["lons"]
    values = openaq_data["values"]
    tss    = openaq_data["timestamps"]

    # Unique stations
    loc_ids = openaq_data["location_ids"]
    unique_locs, uid = np.unique(loc_ids, return_index=True)
    sta_lats = lats[uid]
    sta_lons = lons[uid]

    # Map each station to an annual mean
    sta_values = np.full(len(unique_locs), np.nan)
    for i, lid in enumerate(unique_locs):
        mask = loc_ids == lid
        if mask.sum() >= 5:
            sta_values[i] = values[mask].mean()

    # Sample stations with valid data
    valid = np.isfinite(sta_values)
    idx_valid = np.where(valid)[0]
    if len(idx_valid) > n_max:
        rng = np.random.default_rng(42)
        idx_valid = rng.choice(idx_valid, n_max, replace=False)

    print(f"  {len(idx_valid)} stations with valid annual PM2.5 mean")
    sys.stdout.flush()

    # Day index for inference (use Jan 15 as representative)
    day_idx = 14  # Jan 15

    aria_preds = np.full(len(idx_valid), np.nan)
    obs_values = sta_values[idx_valid]
    sample_lats = sta_lats[idx_valid]
    sample_lons = sta_lons[idx_valid]

    # ERA5 day
    if era5_zarr.exists():
        era5_store = zarr.open(str(era5_zarr), mode="r")
        era5_day = np.array(era5_store[day_idx]).astype(np.float32)   # (30, 721, 1440)
        era5_prev = np.array(era5_store[max(day_idx-1,0)]).astype(np.float32)
    else:
        print("  ERA5 2025 not ready yet")
        return None, obs_values, sample_lats, sample_lons

    with torch.no_grad():
        for i, (slat, slon) in enumerate(zip(sample_lats, sample_lons)):
            # Crop ERA5 around station (168×280 window at 0.25° = ~42°×70°)
            r0 = int(np.clip((90 - slat) / 0.25 - 84, 0, 721 - 168))
            c0 = int(((slon + 180) / 0.25) % 1440)
            c_end = c0 + 280

            if c_end <= 1440:
                e_crop = era5_day[:, r0:r0+168, c0:c0+280]
                e_prev = era5_prev[:, r0:r0+168, c0:c0+280]
            else:
                # Wrap around dateline
                w1 = 1440 - c0
                e_crop = np.concatenate([era5_day[:, r0:r0+168, c0:], era5_day[:, r0:r0+168, :280-w1]], axis=2)
                e_prev = np.concatenate([era5_prev[:, r0:r0+168, c0:], era5_prev[:, r0:r0+168, :280-w1]], axis=2)

            # CAMS patch at station (use PM2.5 as local branch proxy)
            cams_patch = get_cams_pm25_at_patch(cams_nc_path, day_idx, slat, slon)
            if cams_patch is None:
                continue

            # Normalize CAMS as "GHAP current" proxy
            cams_norm = (cams_patch - GHAP_MEAN) / GHAP_STD

            # Build global input (72ch: era5_t + cams_6ch_zeros + era5_prev + cams_6ch_zeros)
            # Use zeros for CAMS channels in ERA5 stack (or interpolate)
            era5_zeros = np.zeros((6, 168, 280), dtype=np.float32)
            global_in = np.concatenate([e_crop, era5_zeros, e_prev, era5_zeros], axis=0)  # (72, 168, 280)

            # Local input: CAMS PM2.5 as current state proxy
            # elevation: zeros if not available
            local_in = np.stack([
                cams_norm,                    # ch0: current PM2.5 (proxy)
                cams_norm,                    # ch1: prev PM2.5 (same as proxy)
                np.zeros((512, 512), dtype=np.float32),  # ch2: elev coarse
                np.zeros((512, 512), dtype=np.float32),  # ch3: elev fine
            ], axis=0)  # (4, 512, 512)

            g_t = torch.from_numpy(global_in).unsqueeze(0).to(device)
            l_t = torch.from_numpy(local_in).unsqueeze(0).to(device)

            # Forward pass
            try:
                out = model.model(g_t, l_t)  # (1, 1, 512, 512) normalized delta
                pred_norm = out.squeeze().cpu().float().numpy()
                # Delta prediction: pred = cams_norm + delta
                pred_norm_final = cams_norm + pred_norm
                pred_ug = pred_norm_final * GHAP_STD + GHAP_MEAN

                # Extract center pixel
                aria_preds[i] = pred_ug[256, 256]
            except Exception as e:
                continue

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(idx_valid)} stations")
                sys.stdout.flush()

    return aria_preds, obs_values, sample_lats, sample_lons


def compute_metrics(obs, pred, label):
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0) & (pred < 500)
    if mask.sum() < 5:
        print(f"  [{label}] too few valid pairs ({mask.sum()})")
        return
    o, p = obs[mask], pred[mask]
    bias = (p - o).mean()
    rmse = np.sqrt(((p - o)**2).mean())
    mae  = np.abs(p - o).mean()
    corr = np.corrcoef(o, p)[0, 1] if len(o) > 2 else np.nan
    print(f"  [{label}]  N={mask.sum():,}  bias={bias:+.2f}  RMSE={rmse:.2f}  MAE={mae:.2f}  r={corr:.3f}")
    sys.stdout.flush()
    return {"bias": bias, "rmse": rmse, "mae": mae, "corr": corr, "n": int(mask.sum())}


def main():
    print("=" * 65)
    print("ARIA Zero-Shot 2025 vs CAMS vs OpenAQ")
    print("=" * 65)
    sys.stdout.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load OpenAQ 2025 ──
    if not OPENAQ_FILE.exists():
        print(f"OpenAQ not ready: {OPENAQ_FILE}")
        sys.exit(1)
    print("\nLoading OpenAQ 2025...")
    oaq = np.load(str(OPENAQ_FILE), allow_pickle=True)
    n_sta = int(oaq["n_stations"])
    print(f"  {int(oaq['n_records']):,} records, {n_sta:,} stations")
    sys.stdout.flush()

    openaq_data = {k: oaq[k] for k in oaq.files}

    # ── 2. Find CAMS 2025 file ──
    cams_nc_jan = CAMS_2025_DIR / "cams_pm25_2025_01.nc"
    if not cams_nc_jan.exists():
        print(f"CAMS 2025 not found: {cams_nc_jan}")
        sys.exit(1)

    # ── 3. CAMS direct comparison ──
    print("\n── CAMS Direct Comparison ──")
    import xarray as xr
    ds_cams = xr.open_dataset(str(cams_nc_jan))
    pm25_key = [k for k in ds_cams.data_vars if "pm" in k.lower()][0]
    # Jan mean
    cams_jan_mean = ds_cams[pm25_key].mean(dim=["valid_time","lead_time"] if "lead_time" in ds_cams.dims else ["valid_time"]).values.astype(np.float32)
    if cams_jan_mean.max() < 0.001:
        cams_jan_mean *= 1e9  # kg/m3 → µg/m3

    cams_lats = ds_cams.latitude.values
    cams_lons = ds_cams.longitude.values

    # Match OpenAQ stations to CAMS grid
    lats   = openaq_data["lats"]
    lons   = openaq_data["lons"]
    loc_ids = openaq_data["location_ids"]
    values = openaq_data["values"]
    unique_locs, uid = np.unique(loc_ids, return_index=True)
    sta_lats = lats[uid]
    sta_lons = lons[uid]
    sta_obs  = np.array([values[loc_ids == lid].mean() if (loc_ids == lid).sum() >= 5 else np.nan
                         for lid in unique_locs])

    lat_idx = np.searchsorted(-cams_lats, -sta_lats).clip(0, len(cams_lats)-1)
    lon_idx = np.searchsorted(cams_lons, sta_lons).clip(0, len(cams_lons)-1)
    cams_at_sta = cams_jan_mean[lat_idx, lon_idx]

    m_cams = compute_metrics(sta_obs, cams_at_sta, "CAMS forecast Jan 2025")

    # ── 4. ARIA zero-shot ──
    print("\n── ARIA Zero-Shot Inference ──")
    ckpt = find_best_checkpoint()
    if ckpt is None:
        print("No checkpoint found — training still running")
    elif not ERA5_2025.exists():
        print("ERA5 2025 not ready yet — run download_era5_2025.py first")
    else:
        model = load_model(ckpt, device)
        aria_preds, obs, slats, slons = predict_at_stations(
            model, ERA5_2025, cams_nc_jan, openaq_data, device,
            dates=None, n_max=2000
        )
        if aria_preds is not None:
            m_aria = compute_metrics(obs, aria_preds, "ARIA zero-shot Jan 2025")
            np.savez_compressed(
                str(OUTPUT_FILE),
                aria_preds=aria_preds, cams_preds=cams_at_sta[:len(obs)],
                obs=obs, lats=slats, lons=slons,
                metrics_cams=np.array([m_cams[k] for k in ["bias","rmse","mae","corr"]] if m_cams else [np.nan]*4),
                metrics_aria=np.array([m_aria[k] for k in ["bias","rmse","mae","corr"]] if m_aria else [np.nan]*4),
            )
            print(f"\nSaved → {OUTPUT_FILE}")

    print("\n── Summary ──")
    print("Comparison: OpenAQ 2025 station measurements vs:")
    print("  1. CAMS forecast  (direct model output)")
    print("  2. ARIA           (zero-shot, trained 2003-2022)")
    print("  3. Aurora         (run separately — needs ERA5 inputs)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
