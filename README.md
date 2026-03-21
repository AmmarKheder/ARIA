# ARIA — Attention-based Resolution-Integrated Air quality

Global PM₂.₅ forecasting at 1km resolution using cross-resolution attention.

## Overview

ARIA is a deep learning model for global PM₂.₅ forecasting that bridges coarse atmospheric reanalysis (ERA5 at 0.25°, CAMS at 0.75°) with fine-scale emission proxies (road networks, nighttime lights, population density) to produce 1km resolution forecasts anywhere in the world.

### Key Features
- **Global coverage** — trained on 2018–2022 worldwide
- **1km resolution** — 25× finer than CAMS operational forecasts
- **Emission-aware** — integrates road density, nighttime lights, and population as structural priors
- **No GHAP at inference** — fully operational with ERA5 + CAMS + static layers only
- **Multi-horizon** — J+1 to J+4 daily forecasts

## Architecture

```
ERA5 (0.25°, 72ch)  →  Global Branch (ViT-L)  ─┐
                                                   ├→ Cross-Attention → CNN Decoder → PM₂.₅ @ 1km
CAMS (0.75°) +       →  Local Branch  (ViT-B)  ─┘
Elevation +
Road density +
Nighttime lights +
Population (18ch total, 512×512 patch)

Training target: GHAP PM₂.₅ at 0.01° (supervision only, not used at inference)
```

## Data

| Source | Variable | Resolution | Role |
|--------|----------|-----------|------|
| ERA5 (ECMWF) | 30 atmospheric vars | 0.25° | Global branch input |
| CAMS EAC4 | PM₂.₅, NO₂, O₃, SO₂, CO, PM₁₀ | 0.75° | Global + local branch |
| GMTED2010 | Elevation | ~250m | Static local input |
| GRIP4 | Road density | 500m | Emission proxy |
| VIIRS VNP46A4 | Nighttime lights 2019 | 500m | Emission proxy |
| WorldPop | Population density 2020 | 1km | Emission proxy |
| GHAP | PM₂.₅ ground truth | 0.01° (1km) | Training supervision only |

## Training

```bash
# Download data
sbatch scripts/submit_download_era5_global.sh
sbatch scripts/submit_download_cams_global.sh
sbatch scripts/submit_download_ghap_global.sh
sbatch scripts/submit_download_proxies.sh

# Train (128 GPUs, 16 nodes)
sbatch scripts/submit_train_global.sh
```

## Results

*Coming soon — model currently training.*

## Citation

```bibtex
@article{aria2025,
  title   = {ARIA: Attention-based Resolution-Integrated Air Quality Forecasting at 1km},
  author  = {Kheder, Ammar and ...},
  journal = {Nature Communications},
  year    = {2025}
}
```
