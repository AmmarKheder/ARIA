#!/usr/bin/env python3
"""
Global ARIA dataset.
Adapts the Europe dataset for global GHAP (18000x36000) + ERA5 (721x1440).
Key changes vs Europe:
  - Global lat/lon domain
  - ERA5 crop centered on tile (168x280 window, same as Europe branch)
  - CAMS global at 0.75deg → interpolated to ERA5 grid
  - No EEA stations (global inference)
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
import zarr

# ── Normalization constants (global ERA5, already normalized in zarr) ──
# ERA5 global zarr is already normalized at download time
ERA5_PRE_NORMALIZED = True   # zarr values are already (raw - mean) / std

# CAMS global: stored as kg/m3, convert to ug/m3 (* 1e9) then normalize
CAMS_VARS = ["pm25", "no2", "o3", "so2", "co", "pm10"]
CAMS_MEANS_UG = np.array([12.0, 1.0, 50.0, 1.5, 150.0, 15.0], dtype=np.float32)
CAMS_STDS_UG  = np.array([15.0, 2.0, 20.0, 3.0, 50.0,  15.0], dtype=np.float32)

GHAP_MEAN = 12.0
GHAP_STD  = 18.0
ELEV_MEAN = 300.0
ELEV_STD  = 500.0

# Global domain
LAT_NORTH  = 90.0
LON_WEST   = -180.0
GHAP_RES   = 0.01
ERA5_RES   = 0.25
GHAP_H     = 18000
GHAP_W     = 36000
ERA5_H     = 721
ERA5_W     = 1440

# ERA5 crop size around tile center (same as Europe branch input)
ERA5_CROP_H = 168
ERA5_CROP_W = 280


def _era5_crop(era5_full, center_lat, center_lon):
    """Extract ERA5_CROP_H x ERA5_CROP_W window centered on (center_lat, center_lon)."""
    c_row = int((LAT_NORTH - center_lat) / ERA5_RES)
    c_col = int((center_lon - LON_WEST) / ERA5_RES)
    r0 = np.clip(c_row - ERA5_CROP_H // 2, 0, ERA5_H - ERA5_CROP_H)
    c0 = np.clip(c_col - ERA5_CROP_W // 2, 0, ERA5_W - ERA5_CROP_W)
    return era5_full[:, r0:r0 + ERA5_CROP_H, c0:c0 + ERA5_CROP_W], r0, c0


class GlobalARIADataset(Dataset):
    """Global dataset for ARIA training on GHAP 2018-2022."""

    def __init__(
        self,
        era5_dir: str,
        ghap_dir: str,
        elev_coarse_path: str,
        elev_hires_path: str,
        years: list,
        cams_dir: str = None,
        emission_proxies_path: str = None,   # road_density, nighttime_lights, population
        horizons: list = None,
        patch_size: int = 512,
        normalize: bool = True,
        augment: bool = False,
        hotspot_ratio: float = 0.5,
        hotspot_power: float = 1.0,
        land_mask_path: str = None,
    ):
        self.era5_dir = Path(era5_dir)
        self.ghap_dir = Path(ghap_dir)
        self.elev_coarse_path = Path(elev_coarse_path)
        self.elev_hires_path = Path(elev_hires_path)
        self.emission_proxies_path = Path(emission_proxies_path) if emission_proxies_path else None
        self.cams_dir = Path(cams_dir) if cams_dir else None
        self.years = years
        self.horizons = horizons or [1, 2, 3, 4]
        self.patch_size = patch_size
        self.normalize = normalize
        self.augment = augment
        self.hotspot_ratio = hotspot_ratio
        self.hotspot_power = hotspot_power
        self.max_horizon = max(self.horizons)

        self._load_stores()
        self._load_elevation()
        self._load_emission_proxies()
        self._build_index()

    def _load_stores(self):
        self.era5_stores = {}
        self.ghap_stores = {}
        self.cams_stores = {}
        self.year_days = {}

        for year in self.years:
            ep = self.era5_dir / f"{year}.zarr"
            gp = self.ghap_dir / f"{year}.zarr"
            if not ep.exists() or not gp.exists():
                print(f"  Warning: missing data for {year}, skipping")
                continue
            self.era5_stores[year] = zarr.open(str(ep), mode="r")
            self.ghap_stores[year] = zarr.open(str(gp), mode="r")
            self.year_days[year] = self.era5_stores[year].shape[0]
            if self.cams_dir:
                cp = self.cams_dir / f"{year}.zarr"
                if cp.exists():
                    self.cams_stores[year] = zarr.open(str(cp), mode="r")

    def _load_emission_proxies(self):
        """Load static emission proxy layers: road_density, nighttime_lights, population."""
        if self.emission_proxies_path and self.emission_proxies_path.exists():
            g = zarr.open_group(str(self.emission_proxies_path), mode='r', zarr_format=2)
            # Each is (18000, 36000) float32, already normalized 0-1
            self.proxy_roads  = g['road_density']
            self.proxy_lights = g['nighttime_lights']
            self.proxy_pop    = g['population']
            print(f"  Loaded emission proxies from {self.emission_proxies_path}")
        else:
            self.proxy_roads  = None
            self.proxy_lights = None
            self.proxy_pop    = None
            print("  WARNING: No emission proxies — will use zeros for road/lights/population")

    def _load_elevation(self):
        # Coarse elevation: (2, 721, 1440) — index 0 = elev, 1 = slope or std
        if self.elev_coarse_path.exists():
            self.elev_coarse_full = zarr.open(str(self.elev_coarse_path), mode="r")
        else:
            self.elev_coarse_full = None

        # Hires elevation: (67200, 172800) GMTED2010 at ~250m
        if self.elev_hires_path.exists():
            self.elev_hires_full = zarr.open(str(self.elev_hires_path), mode="r")
        else:
            self.elev_hires_full = None

    def _build_index(self):
        self.samples = []
        for year in sorted(self.year_days.keys()):
            n_days = self.year_days[year]
            for day_t in range(n_days - self.max_horizon):
                for h in self.horizons:
                    if day_t + h < n_days:
                        self.samples.append((year, day_t, h))

    def __len__(self):
        return len(self.samples)

    def _sample_patch(self, ghap_day, rng):
        """Sample a 512x512 patch, biased toward high-PM2.5 areas."""
        ps = self.patch_size
        if rng.random() < (1.0 - self.hotspot_ratio):
            r = rng.integers(0, GHAP_H - ps)
            c = rng.integers(0, GHAP_W - ps)
        else:
            block = 512
            nr = (GHAP_H - ps) // block + 1
            nc = (GHAP_W - ps) // block + 1
            # Coarse sample to find hotspots
            step_r = max(1, GHAP_H // nr)
            step_c = max(1, GHAP_W // nc)
            coarse = np.array(ghap_day[::step_r, ::step_c]).astype(np.float32)
            coarse = np.nan_to_num(coarse, nan=0.0)
            weights = np.maximum(coarse, 0.0).ravel() ** self.hotspot_power + 1.0
            # Pad/trim to nr*nc
            weights = weights[:nr * nc]
            if len(weights) < nr * nc:
                weights = np.pad(weights, (0, nr * nc - len(weights)), constant_values=1.0)
            weights /= weights.sum()
            chosen = rng.choice(nr * nc, p=weights)
            r = min((chosen // nc) * block, GHAP_H - ps)
            c = min((chosen % nc) * block, GHAP_W - ps)
        return int(r), int(c)

    def __getitem__(self, idx):
        year, day_t, horizon = self.samples[idx]
        rng = np.random.default_rng(idx)

        # ── ERA5 (already normalized in zarr) ──
        era5_t = np.array(self.era5_stores[year][day_t]).astype(np.float32)       # (30, 721, 1440)
        day_prev = max(day_t - 1, 0)
        era5_prev = np.array(self.era5_stores[year][day_prev]).astype(np.float32)  # (30, 721, 1440)

        # ── CAMS (kg/m3 → ug/m3 → normalize) ──
        def load_cams(store, day):
            channels = []
            for var in CAMS_VARS:
                ch = np.nan_to_num(np.array(store[var][day]).astype(np.float32)) * 1e9
                channels.append(ch)
            arr = np.stack(channels, axis=0)  # (6, 241, 480)
            if self.normalize:
                arr = (arr - CAMS_MEANS_UG[:, None, None]) / CAMS_STDS_UG[:, None, None]
            return arr

        if year in self.cams_stores:
            cams_t    = load_cams(self.cams_stores[year], day_t)
            cams_prev = load_cams(self.cams_stores[year], day_prev)
            # Interpolate CAMS (241, 480) → ERA5 (721, 1440)
            cams_t_era5 = F.interpolate(
                torch.from_numpy(cams_t).unsqueeze(0),
                size=(ERA5_H, ERA5_W), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            cams_prev_era5 = F.interpolate(
                torch.from_numpy(cams_prev).unsqueeze(0),
                size=(ERA5_H, ERA5_W), mode="bilinear", align_corners=False
            ).squeeze(0).numpy()
            era5_full = np.concatenate([era5_t, cams_t_era5, era5_prev, cams_prev_era5], axis=0)  # (70, 721, 1440)
        else:
            # No CAMS: pad with zeros
            zeros = np.zeros((6, ERA5_H, ERA5_W), dtype=np.float32)
            era5_full = np.concatenate([era5_t, zeros, era5_prev, zeros], axis=0)  # (70, 721, 1440)

        # ── Sample patch ──
        ghap_day_store  = self.ghap_stores[year][day_t]
        r, c = self._sample_patch(ghap_day_store, rng)
        ps = self.patch_size

        ghap_t    = np.nan_to_num(np.array(ghap_day_store[r:r+ps, c:c+ps]).astype(np.float32))
        ghap_tgt  = np.nan_to_num(np.array(self.ghap_stores[year][day_t + horizon][r:r+ps, c:c+ps]).astype(np.float32))
        ghap_prev = np.nan_to_num(np.array(self.ghap_stores[year][day_prev][r:r+ps, c:c+ps]).astype(np.float32))

        if self.normalize:
            ghap_t    = (ghap_t    - GHAP_MEAN) / GHAP_STD
            ghap_tgt  = (ghap_tgt  - GHAP_MEAN) / GHAP_STD
            ghap_prev = (ghap_prev - GHAP_MEAN) / GHAP_STD

        # ── Patch center lat/lon ──
        center_lat = LAT_NORTH - (r + ps / 2) * GHAP_RES
        center_lon = LON_WEST  + (c + ps / 2) * GHAP_RES

        # ── ERA5 crop around tile ──
        era5_crop, er0, ec0 = _era5_crop(era5_full, center_lat, center_lon)  # (70, 168, 280)

        # ── Coarse elevation crop (same window as ERA5) ──
        if self.elev_coarse_full is not None:
            elev_c = np.array(self.elev_coarse_full[0, er0:er0+ERA5_CROP_H, ec0:ec0+ERA5_CROP_W]).astype(np.float32)
            if elev_c.shape != (ERA5_CROP_H, ERA5_CROP_W):
                elev_c = np.zeros((ERA5_CROP_H, ERA5_CROP_W), dtype=np.float32)
            if self.normalize:
                elev_c = (elev_c - ELEV_MEAN) / ELEV_STD
        else:
            elev_c = np.zeros((ERA5_CROP_H, ERA5_CROP_W), dtype=np.float32)

        # ── Hires elevation crop ──
        if self.elev_hires_full is not None:
            # GMTED2010: (67200, 172800) covers globe at ~250m (0.00208 deg)
            # GHAP: 0.01 deg → scale factor ~4.8x
            scale = 67200 / 18000  # ~3.73
            hr = int(r * scale)
            hc = int(c * scale)
            hps = int(ps * scale)
            elev_h_raw = np.array(self.elev_hires_full[hr:hr+hps, hc:hc+hps]).astype(np.float32)
            elev_h = F.interpolate(
                torch.from_numpy(elev_h_raw).unsqueeze(0).unsqueeze(0),
                size=(ps, ps), mode="bilinear", align_corners=False
            ).squeeze().numpy()
            if self.normalize:
                elev_h = (elev_h - ELEV_MEAN) / ELEV_STD
        else:
            elev_h = np.zeros((ps, ps), dtype=np.float32)

        # ── Lat/lon grids ──
        rows = np.arange(r, r + ps, dtype=np.float32)
        cols = np.arange(c, c + ps, dtype=np.float32)
        lats = (LAT_NORTH - rows * GHAP_RES) / 90.0           # normalized [-1, 1]
        lons = (LON_WEST + cols * GHAP_RES) / 180.0           # normalized [-1, 1]
        lat_grid = lats[:, None] * np.ones((1, ps), dtype=np.float32)
        lon_grid = lons[None, :] * np.ones((ps, 1), dtype=np.float32)

        # ── Emission proxies at patch (static, same for all days) ──
        def load_proxy(store, r, c, ps):
            if store is not None:
                return np.array(store[r:r+ps, c:c+ps]).astype(np.float32)
            return np.zeros((ps, ps), dtype=np.float32)

        proxy_r = load_proxy(self.proxy_roads,  r, c, ps)
        proxy_l = load_proxy(self.proxy_lights, r, c, ps)
        proxy_p = load_proxy(self.proxy_pop,    r, c, ps)

        # ── CAMS interpolated to fine patch (6ch t + 6ch t-1) ──
        # cams_t and cams_prev are (6, ERA5_H, ERA5_W) at this point before the crop
        # We need them at the local patch resolution (512×512)
        def cams_at_patch(cams_full_era5, r, c, ps):
            """Bilinearly interpolate CAMS from ERA5 grid to 512×512 patch."""
            # Convert patch pixel coords to ERA5 coords
            patch_lats = np.linspace(
                LAT_NORTH - r * GHAP_RES,
                LAT_NORTH - (r + ps) * GHAP_RES, ps, dtype=np.float32)
            patch_lons = np.linspace(
                LON_WEST + c * GHAP_RES,
                LON_WEST + (c + ps) * GHAP_RES, ps, dtype=np.float32)
            t = torch.from_numpy(cams_full_era5).unsqueeze(0)  # (1, 6, 721, 1440)
            out = F.interpolate(t, size=(ps, ps), mode='bilinear', align_corners=False)
            return out.squeeze(0).numpy()  # (6, 512, 512)

        if year in self.cams_stores:
            cams_fine_t    = cams_at_patch(cams_t_era5, r, c, ps)    # (6, 512, 512)
            cams_fine_prev = cams_at_patch(cams_prev_era5, r, c, ps) # (6, 512, 512)
        else:
            cams_fine_t    = np.zeros((6, ps, ps), dtype=np.float32)
            cams_fine_prev = np.zeros((6, ps, ps), dtype=np.float32)

        # ── Local input: CAMS fine (12ch) + elevation (1ch) + proxies (3ch) + lat/lon (2ch) ──
        # Total: 18 channels — NO GHAP at input → model works at inference without GHAP
        local_input = np.concatenate([
            cams_fine_t,                                    # (6, 512, 512) CAMS today
            cams_fine_prev,                                 # (6, 512, 512) CAMS yesterday
            elev_h[None],                                   # (1, 512, 512) elevation
            proxy_r[None],                                  # (1, 512, 512) road density
            proxy_l[None],                                  # (1, 512, 512) nighttime lights
            proxy_p[None],                                  # (1, 512, 512) population
            lat_grid[None],                                 # (1, 512, 512) latitude
            lon_grid[None],                                 # (1, 512, 512) longitude
        ], axis=0)  # (18, 512, 512)

        # ── Augmentation ──
        if self.augment:
            if rng.random() < 0.5:
                local_input = local_input[:, ::-1, :].copy()
                ghap_tgt    = ghap_tgt[::-1, :].copy()
                elev_h      = elev_h[::-1, :].copy()
            if rng.random() < 0.5:
                local_input = local_input[:, :, ::-1].copy()
                ghap_tgt    = ghap_tgt[:, ::-1].copy()
                elev_h      = elev_h[:, ::-1].copy()

        # ── Wind at patch center ──
        era5_row_idx = int(np.clip((LAT_NORTH - center_lat) / ERA5_RES, 0, ERA5_H - 1))
        era5_col_idx = int(np.clip((center_lon - LON_WEST)  / ERA5_RES, 0, ERA5_W - 1))
        wind_u = float(era5_t[0, era5_row_idx, era5_col_idx])  # u10 (already normalized)
        wind_v = float(era5_t[1, era5_row_idx, era5_col_idx])  # v10

        return {
            "era5":             torch.from_numpy(era5_crop),                          # (70, 168, 280)
            "elevation_coarse": torch.from_numpy(elev_c),                             # (168, 280)
            "local_input":      torch.from_numpy(local_input),                        # (5, 512, 512)
            "elevation_hires":  torch.from_numpy(elev_h),                             # (512, 512)
            "target":           torch.from_numpy(ghap_tgt).unsqueeze(0),              # (1, 512, 512)
            "lead_time":        torch.tensor(float(horizon), dtype=torch.float32),
            "patch_center":     torch.tensor([center_lat, center_lon], dtype=torch.float32),
            "wind_at_patch":    torch.tensor([wind_u, wind_v], dtype=torch.float32),
            # Dummy station tensors (not used globally)
            "station_pixels":   torch.zeros(64, 2, dtype=torch.float32),
            "station_values":   torch.full((64,), float("nan"), dtype=torch.float32),
            "station_count":    torch.tensor(0, dtype=torch.long),
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
            years=years,
            horizons=d.get("horizons", [1, 2, 3, 4]),
            patch_size=d.get("patch_size", 512),
            normalize=d.get("normalize", True),
            augment=augment,
            hotspot_ratio=d.get("hotspot_ratio", 0.5),
            hotspot_power=d.get("hotspot_power", 1.0),
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
