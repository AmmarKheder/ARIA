---
title: ARIA - Global PM2.5 Forecasting at 1km
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: true
license: mit
short_description: Global PM2.5 forecasting at 1km resolution using dual-branch attention
---

# ARIA — Attention-based Resolution-Integrated Air quality

Global PM₂.₅ forecasting at **~1 km resolution** using a dual-branch Vision Transformer
that fuses ERA5 meteorology, CAMS chemistry, elevation, road density, nighttime lights, and population.

## Features
- 🔍 **Forecast**: select any region + date → auto-download ERA5+CAMS → run ARIA → 1km map
- 📂 **Upload**: analyze your own NetCDF files (Claude-powered)
- 📥 **Download code**: generate CDS API scripts for any region/date
- 🔧 **Retrain**: generate SLURM scripts for HPC training

## Status
🔄 **Training in progress** on 128× MI250X GPUs (LUMI supercomputer).
Checkpoint will be published here when training completes.

## Links
- [GitHub](https://github.com/AmmarKheder/ARIA)
- [Website](https://ammarkheder.github.io/ARIA/)
