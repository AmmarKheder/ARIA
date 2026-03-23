#!/usr/bin/env python3
"""
Global ARIA dataset — LUMI Final Run.

Key decisions:
  - Log-transform normalization: log1p(PM2.5) — skewness 3.35 → 1.31
  - Horizons [0, 1]: J+0 downscaling + J+1 forecast only
  - Emission proxies fix: properly passed from DataModule
  - Hotspot sampler: 70% patches from PM2.5 > 35 µg/m³ regions
  - Years: 2018-2020 (GHAP availability)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import zarr

# ── ERA5: already normalized in zarr at download time ──
ERA5_PRE_NORMALIZED = True

# ── CAMS: stored as kg/m3 → µg/m3 → linear normalize ──
CAMS_VARS = ["pm25", "no2", "o3", "so2", "co", "pm10"]
CAMS_MEANS_UG = np.array([12.0, 1.0, 50.0, 1.5, 150.0, 15.0], dtype=np.float32)
CAMS_STDS_UG  = np.array([15.0, 2.0, 20.0, 3.0,  50.0, 15.0], dtype=np.float32)

# ── GHAP: Log-space normalization log1p(PM2.5) ──
# Computed from GHAP 2018 sample: mean=16.2, skewness=3.35
# After log1p: mean=2.416, std=0.800, skewness=1.31
LOG_GHAP_MEAN = 2.416
LOG_GHAP_STD  = 0.800

# Legacy linear constants kept for RMSE denormalization
GHAP_MEAN = 12.0
GHAP_STD  = 18.0
ELEV_MEAN = 300.0
ELEV_STD  = 500.0

# ── Hotspot threshold in log-space ──
# PM2.5 = 35 µg/m³ → log1p(35) = 3.58 → normalized = (3.58 - 2.416) / 0.800 = 1.455
LOG_HOTSPOT_THRESH = (np.log1p(35.0) - LOG_GHAP_MEAN) / LOG_GHAP_STD

# Global domain
LAT_NORTH  = 90.0
LON_WEST   = -180.0
GHAP_RES   = 0.01
ERA5_RES   = 0.25
GHAP_H     = 18000
GHAP_W     = 36000
ERA5_H     = 721
ERA5_W     = 1440
ERA5_CROP_H = 168
ERA5_CROP_W = 280


def ghap_to_lognorm(x: np.ndarray) -> np.ndarray:
    """PM2.5 µg/m³ → log1p normalized. Clip negatives (ocean/missing = 0)."""
    return (np.log1p(np.maximum(x, 0.0)) - LOG_GHAP_MEAN) / LOG_GHAP_STD


def lognorm_to_ghap(x: np.ndarray) -> np.ndarray:
    """Inverse: log1p normalized → PM2.5 µg/m³."""
    return np.expm1(x * LOG_GHAP_STD + LOG_GHAP_MEAN)


def _era5_crop(era5_full, center_lat, center_lon):
    c_row = int((LAT_NORTH - center_lat) / ERA5_RES)
    c_col = int((center_lon - LON_WEST)  / ERA5_RES)
    r0 = int(np.clip(c_row - ERA5_CROP_H // 2, 0, ERA5_H - ERA5_CROP_H))
    c0 = int(np.clip(c_col - ERA5_CROP_W // 2, 0, ERA5_W - ERA5_CROP_W))
    return era5_full[:, r0:r0 + ERA5_CROP_H, c0:c0 + ERA5_CROP_W], r0, c0


class GlobalARIADataset(Dataset):
    """Global dataset for ARIA training on GHAP 2018-2022.

    Normalization: log1p for GHAP (skewness 3.35 → 1.31).
    Horizons: [0, 1] — J+0 downscaling + J+1 forecast.
    """

    def __init__(
        self,
        era5_dir: str,
        ghap_dir: str,
        elev_coarse_path: str,
        elev_hires_path: str,
        years: list,
        cams_dir: str = None,
        emission_proxies_path: str = None,
        horizons: list = None,
        patch_size: int = 512,
        normalize: bool = True,
        augment: bool = False,
        hotspot_ratio: float = 0.7,
        hotspot_power: float = 1.5,
        patches_per_day: int = 32,
        land_mask_path: str = None,
    ):
        self.era5_dir  = Path(era5_dir)
        self.ghap_dir  = Path(ghap_dir)
        self.elev_coarse_path = Path(elev_coarse_path)
        self.elev_hires_path  = Path(elev_hires_path)
        self.emission_proxies_path = Path(emission_proxies_path) if emission_proxies_path else None
        self.cams_dir  = Path(cams_dir) if cams_dir else None
        self.years     = years
        self.horizons  = horizons if horizons is not None else [0, 1]
        self.patch_size = patch_size
        self.normalize  = normalize
        self.augment    = augment
        self.hotspot_ratio = hotspot_ratio
        self.hotspot_power = hotspot_power
        self.patches_per_day = patches_per_day
        self.max_horizon   = max(self.horizons) if self.horizons else 1

        # Lazy-load zarr stores: only build the sample index now,
        # actual zarr handles are opened on first __getitem__ call
        # (avoids 128 DDP ranks stampeding Lustre simultaneously).
        self._stores_ready = False
        self._build_index_lightweight()

    def _build_index_lightweight(self):
        """Build sample index WITHOUT opening zarr — just check existence and read .zarray metadata."""
        self.year_days = {}
        self._valid_years = []
        self._has_cams = {}

        for year in self.years:
            ep = self.era5_dir / f"{year}.zarr"
            gp = self.ghap_dir / f"{year}.zarr"
            if not ep.exists():
                print(f"  Skipping {year}: ERA5 missing ({ep})")
                continue
            if not gp.exists():
                print(f"  Skipping {year}: GHAP missing ({gp})")
                continue
            # Read shape from .zarray metadata (tiny JSON, no zarr.open)
            import json
            zarray_path = ep / ".zarray"
            if zarray_path.exists():
                with open(zarray_path) as f:
                    meta = json.load(f)
                n_days = meta["shape"][0]
            else:
                # Fallback: open briefly on rank 0 only
                z = zarr.open(str(ep), mode="r")
                n_days = z.shape[0]
                del z
            self.year_days[year] = n_days
            self._valid_years.append(year)
            if self.cams_dir:
                cp = self.cams_dir / f"{year}.zarr"
                self._has_cams[year] = cp.exists()
                if not cp.exists():
                    print(f"  Warning: CAMS missing for {year} ({cp})")
            print(f"  Indexed {year}: {n_days} days")

        self.samples = []
        for year in sorted(self.year_days.keys()):
            n_days = self.year_days[year]
            for day_t in range(n_days - self.max_horizon):
                for h in self.horizons:
                    if day_t + h < n_days:
                        for p in range(self.patches_per_day):
                            self.samples.append((year, day_t, h, p))
        print(f"  Total samples: {len(self.samples):,} "
              f"(years={sorted(self.year_days.keys())}, horizons={self.horizons}, "
              f"patches/day={self.patches_per_day})")

    def _ensure_stores(self):
        """Lazy-open zarr stores on first access (called from __getitem__ in worker process)."""
        if self._stores_ready:
            return
        self.era5_stores = {}
        self.ghap_stores = {}
        self.cams_stores = {}
        for year in self._valid_years:
            self.era5_stores[year] = zarr.open(str(self.era5_dir / f"{year}.zarr"), mode="r")
            self.ghap_stores[year] = zarr.open(str(self.ghap_dir / f"{year}.zarr"), mode="r")
            if self.cams_dir and self._has_cams.get(year, False):
                self.cams_stores[year] = zarr.open(str(self.cams_dir / f"{year}.zarr"), mode="r")

        # Elevation
        self.elev_coarse_full = zarr.open(str(self.elev_coarse_path), mode="r") \
            if self.elev_coarse_path.exists() else None
        self.elev_hires_full = zarr.open(str(self.elev_hires_path), mode="r") \
            if self.elev_hires_path.exists() else None

        # Emission proxies
        # nighttime_lights and population: already in [0,1] ✅
        # road_density: currently all-zero in zarr (data issue) → treated as zeros
        if self.emission_proxies_path and self.emission_proxies_path.exists():
            g = zarr.open_group(str(self.emission_proxies_path), mode='r', zarr_format=2)
            self.proxy_roads  = g['road_density']    # ⚠ all-zero — placeholder
            self.proxy_lights = g['nighttime_lights'] # [0,1] ✅
            self.proxy_pop    = g['population']       # [0,1] ✅
        else:
            self.proxy_roads = self.proxy_lights = self.proxy_pop = None

        self._stores_ready = True

    def __len__(self):
        return len(self.samples)

    def _sample_patch(self, ghap_day, rng):
        """Sample 512×512 patch — 70% biased toward PM2.5 > 35 µg/m³ hotspots."""
        ps = self.patch_size
        if rng.random() >= self.hotspot_ratio:
            # Uniform random patch
            return int(rng.integers(0, GHAP_H - ps)), int(rng.integers(0, GHAP_W - ps))

        # Hotspot-biased patch: coarse scan for high-PM2.5 regions
        block = 512
        nr = max(1, (GHAP_H - ps) // block + 1)
        nc = max(1, (GHAP_W - ps) // block + 1)
        step_r = max(1, GHAP_H // nr)
        step_c = max(1, GHAP_W // nc)
        coarse = np.nan_to_num(
            np.array(ghap_day[::step_r, ::step_c]).astype(np.float32), nan=0.0
        )
        weights = np.maximum(coarse, 0.0).ravel() ** self.hotspot_power + 1.0
        weights = weights[:nr * nc]
        if len(weights) < nr * nc:
            weights = np.pad(weights, (0, nr * nc - len(weights)), constant_values=1.0)
        weights /= weights.sum()
        chosen = rng.choice(nr * nc, p=weights)
        r = int(min((chosen // nc) * block, GHAP_H - ps))
        c = int(min((chosen  % nc) * block, GHAP_W - ps))
        return r, c

    def _load_proxy_patch(self, store, r, c, ps):
        if store is not None:
            return np.array(store[r:r+ps, c:c+ps]).astype(np.float32)
        return np.zeros((ps, ps), dtype=np.float32)

    def __getitem__(self, idx):
        self._ensure_stores()  # lazy-open zarr on first access in this worker
        year, day_t, horizon, patch_id = self.samples[idx]
        rng = np.random.default_rng(idx)

        # ── ERA5 global (pre-normalized in zarr) ──
        era5_t    = np.array(self.era5_stores[year][day_t]).astype(np.float32)        # (30, 721, 1440)
        day_prev  = max(day_t - 1, 0)
        era5_prev = np.array(self.era5_stores[year][day_prev]).astype(np.float32)

        # ── CAMS (kg/m3 → µg/m3 → linear normalize) ──
        def load_cams(store, day):
            chs = []
            for var in CAMS_VARS:
                ch = np.nan_to_num(
                    np.array(store[var][day]).astype(np.float32)
                ) * 1e9
                chs.append(ch)
            arr = np.stack(chs, axis=0)   # (6, H_cams, W_cams)
            if self.normalize:
                arr = (arr - CAMS_MEANS_UG[:, None, None]) / CAMS_STDS_UG[:, None, None]
            return arr

        if year in self.cams_stores:
            cams_t    = load_cams(self.cams_stores[year], day_t)
            cams_prev = load_cams(self.cams_stores[year], day_prev)
            cams_t_era5 = F.interpolate(
                torch.from_numpy(cams_t).unsqueeze(0),
                size=(ERA5_H, ERA5_W), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            cams_prev_era5 = F.interpolate(
                torch.from_numpy(cams_prev).unsqueeze(0),
                size=(ERA5_H, ERA5_W), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
        else:
            cams_t_era5 = cams_prev_era5 = np.zeros((6, ERA5_H, ERA5_W), dtype=np.float32)

        era5_full = np.concatenate(
            [era5_t, cams_t_era5, era5_prev, cams_prev_era5], axis=0
        )  # (72, 721, 1440)

        # ── Sample patch ──
        ghap_day_store = self.ghap_stores[year][day_t]
        r, c = self._sample_patch(ghap_day_store, rng)
        ps   = self.patch_size

        # ── Load GHAP patches — LOG1P normalization ──
        day_tgt = day_t + horizon  # horizon=0 → downscaling, horizon=1 → forecast
        ghap_t   = np.nan_to_num(np.array(ghap_day_store[r:r+ps, c:c+ps]).astype(np.float32))
        ghap_tgt = np.nan_to_num(np.array(self.ghap_stores[year][day_tgt][r:r+ps, c:c+ps]).astype(np.float32))

        if self.normalize:
            ghap_t   = ghap_to_lognorm(ghap_t)
            ghap_tgt = ghap_to_lognorm(ghap_tgt)

        # ── Patch geometry ──
        center_lat = LAT_NORTH - (r + ps / 2) * GHAP_RES
        center_lon = LON_WEST  + (c + ps / 2) * GHAP_RES

        # ── ERA5 crop (72, 168, 280) ──
        era5_crop, er0, ec0 = _era5_crop(era5_full, center_lat, center_lon)

        # ── Coarse elevation (168, 280) ──
        if self.elev_coarse_full is not None:
            elev_c = np.array(
                self.elev_coarse_full[0, er0:er0+ERA5_CROP_H, ec0:ec0+ERA5_CROP_W]
            ).astype(np.float32)
            if elev_c.shape != (ERA5_CROP_H, ERA5_CROP_W):
                elev_c = np.zeros((ERA5_CROP_H, ERA5_CROP_W), dtype=np.float32)
            if self.normalize:
                elev_c = (elev_c - ELEV_MEAN) / ELEV_STD
        else:
            elev_c = np.zeros((ERA5_CROP_H, ERA5_CROP_W), dtype=np.float32)

        # ── Hires elevation (512, 512) ──
        if self.elev_hires_full is not None:
            scale  = 67200 / GHAP_H
            hr, hc = int(r * scale), int(c * scale)
            hps    = int(ps * scale)
            raw    = np.array(self.elev_hires_full[hr:hr+hps, hc:hc+hps]).astype(np.float32)
            elev_h = F.interpolate(
                torch.from_numpy(raw).unsqueeze(0).unsqueeze(0),
                size=(ps, ps), mode="bilinear", align_corners=False
            ).squeeze().numpy()
            if self.normalize:
                elev_h = (elev_h - ELEV_MEAN) / ELEV_STD
        else:
            elev_h = np.zeros((ps, ps), dtype=np.float32)

        # ── Lat/lon grids (normalized -1 to 1) ──
        rows = np.arange(r, r + ps, dtype=np.float32)
        cols = np.arange(c, c + ps, dtype=np.float32)
        lat_grid = ((LAT_NORTH - rows * GHAP_RES) / 90.0)[:, None] * np.ones((1, ps), dtype=np.float32)
        lon_grid = ((LON_WEST + cols * GHAP_RES) / 180.0)[None, :] * np.ones((ps, 1), dtype=np.float32)

        # ── Emission proxies (512, 512 each) ──
        proxy_r = self._load_proxy_patch(self.proxy_roads,  r, c, ps)  # roads
        proxy_l = self._load_proxy_patch(self.proxy_lights, r, c, ps)  # nighttime lights
        proxy_p = self._load_proxy_patch(self.proxy_pop,    r, c, ps)  # population

        # ── CAMS fine-grid at patch (6+6 channels) ──
        def cams_at_patch(cams_era5):
            return F.interpolate(
                torch.from_numpy(cams_era5).unsqueeze(0),
                size=(ps, ps), mode='bilinear', align_corners=False
            ).squeeze(0).numpy()

        cams_fine_t    = cams_at_patch(cams_t_era5)    # (6, 512, 512)
        cams_fine_prev = cams_at_patch(cams_prev_era5) # (6, 512, 512)

        # ── Local input: 18 channels ──
        # [CAMS_t(6) | CAMS_prev(6) | elev(1) | roads(1) | lights(1) | pop(1) | lat(1) | lon(1)]
        local_input = np.concatenate([
            cams_fine_t,         # (6, 512, 512)
            cams_fine_prev,      # (6, 512, 512)
            elev_h[None],        # (1, 512, 512)
            proxy_r[None],       # (1, 512, 512) road density
            proxy_l[None],       # (1, 512, 512) nighttime lights
            proxy_p[None],       # (1, 512, 512) population density
            lat_grid[None],      # (1, 512, 512)
            lon_grid[None],      # (1, 512, 512)
        ], axis=0)  # (18, 512, 512)

        # ── Augmentation (flip only — no rotation to preserve lat/lon grids) ──
        if self.augment:
            if rng.random() < 0.5:
                local_input = local_input[:, ::-1, :].copy()
                ghap_tgt    = ghap_tgt[::-1, :].copy()
            if rng.random() < 0.5:
                local_input = local_input[:, :, ::-1].copy()
                ghap_tgt    = ghap_tgt[:, ::-1].copy()

        # ── Wind at patch center (for wind scanner) ──
        era5_ri = int(np.clip((LAT_NORTH - center_lat) / ERA5_RES, 0, ERA5_H - 1))
        era5_ci = int(np.clip((center_lon - LON_WEST)  / ERA5_RES, 0, ERA5_W - 1))
        wind_u  = float(era5_t[0, era5_ri, era5_ci])
        wind_v  = float(era5_t[1, era5_ri, era5_ci])

        return {
            "era5":             torch.from_numpy(era5_crop),                 # (72, 168, 280)
            "elevation_coarse": torch.from_numpy(elev_c),                    # (168, 280)
            "local_input":      torch.from_numpy(local_input),               # (18, 512, 512)
            "elevation_hires":  torch.from_numpy(elev_h),                    # (512, 512)
            "target":           torch.from_numpy(ghap_tgt).unsqueeze(0),     # (1, 512, 512) log-norm
            "lead_time":        torch.tensor(float(horizon), dtype=torch.float32),
            "patch_center":     torch.tensor([center_lat, center_lon], dtype=torch.float32),
            "wind_at_patch":    torch.tensor([wind_u, wind_v], dtype=torch.float32),
            "meta": {"year": year, "day": day_t, "horizon": horizon, "row": r, "col": c},
        }


class GlobalARIADataModule(pl.LightningDataModule):

    def __init__(self, config: dict):
        super().__init__()
        self.cfg = config

    def _make_dataset(self, years, augment=False):
        d = self.cfg["data"]
        return GlobalARIADataset(
            era5_dir=d["era5_dir"],
            ghap_dir=d["ghap_dir"],
            elev_coarse_path=d["elev_coarse_path"],
            elev_hires_path=d["elev_hires_path"],
            cams_dir=d.get("cams_dir"),
            emission_proxies_path=d.get("emission_proxies_path"),  # ← fixed: was missing
            years=years,
            horizons=d.get("horizons", [0, 1]),
            patch_size=d.get("patch_size", 512),
            normalize=d.get("normalize", True),
            augment=augment,
            hotspot_ratio=d.get("hotspot_ratio", 0.7),
            hotspot_power=d.get("hotspot_power", 1.5),
            patches_per_day=d.get("patches_per_day", 32),
        )

    def setup(self, stage=None):
        d = self.cfg["data"]
        if stage in ("fit", None):
            self.train_ds = self._make_dataset(d["train_years"], augment=d.get("augment", True))
            self.val_ds   = self._make_dataset(d["val_years"],   augment=False)
        if stage in ("test", None):
            self.test_ds  = self._make_dataset(d["test_years"],  augment=False)

    def train_dataloader(self):
        nw = self.cfg["data"].get("num_workers", 4)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg["train"]["batch_size"],
            shuffle=True, num_workers=nw,
            pin_memory=True, persistent_workers=(nw > 0), drop_last=True,
        )

    def val_dataloader(self):
        nw = self.cfg["data"].get("num_workers", 4)
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg["train"].get("val_batch_size", 2),
            shuffle=False, num_workers=nw, pin_memory=True,
        )

    def test_dataloader(self):
        nw = self.cfg["data"].get("num_workers", 4)
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg["train"].get("val_batch_size", 2),
            shuffle=False, num_workers=nw, pin_memory=True,
        )
