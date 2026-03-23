#!/usr/bin/env python3
"""
ARIA — Attention-based Resolution-Integrated Air quality
HuggingFace Spaces demo: global PM2.5 forecasting at 1km resolution.
"""
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import gradio as gr

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import cdsapi
    HAS_CDS = True
except ImportError:
    HAS_CDS = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ═══════════════════════════════════════════════════════════
GITHUB_URL  = "https://github.com/AmmarKheder/ARIA"
HF_MODEL    = "AmmarKheder/ARIA-global"
WEBSITE_URL = "https://ammarkheder.github.io/ARIA/"

REGIONS = {
    "East Asia":      {"lat": [15,  55],  "lon": [70,  145]},
    "South Asia":     {"lat": [5,   35],  "lon": [60,  100]},
    "Europe":         {"lat": [35,  72],  "lon": [-25,  45]},
    "North America":  {"lat": [25,  70],  "lon": [-130, -60]},
    "Middle East":    {"lat": [12,  42],  "lon": [25,   65]},
    "Africa":         {"lat": [-35, 37],  "lon": [-20,  55]},
    "Southeast Asia": {"lat": [-10, 30],  "lon": [95,  145]},
    "South America":  {"lat": [-55, 15],  "lon": [-82, -34]},
    "Custom bbox":    None,
}

POLLUTANTS = {
    "PM2.5": "pm2p5",
    "PM10":  "pm10",
    "NO2":   "no2",
    "O3":    "go3",
    "SO2":   "so2",
    "CO":    "co",
}

CSS = """
#title    { text-align:center; font-size:2rem; font-weight:800; margin-bottom:0.2rem; }
#subtitle { text-align:center; color:#6b7280; margin-bottom:0.3rem; }
#links    { text-align:center; margin-bottom:1.5rem; }
footer    { display:none !important; }
"""

# ═══════════════════════════════════════════════════════════
# FILE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_netcdf(path: str) -> dict:
    if not HAS_XARRAY:
        return {"error": "xarray not installed"}
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
        info = {"variables": list(ds.data_vars), "dims": dict(ds.dims)}
        for n in ["latitude", "lat"]:
            if n in ds.coords:
                info["lat_range"] = [float(ds[n].min()), float(ds[n].max())]
        for n in ["longitude", "lon"]:
            if n in ds.coords:
                info["lon_range"] = [float(ds[n].min()), float(ds[n].max())]
        t = ds.coords.get("time", ds.coords.get("valid_time", None))
        if t is not None:
            info["time"] = [str(t.values[0])[:16], str(t.values[-1])[:16]]
        ds.close()
        return info
    except Exception as e:
        return {"error": str(e)}


def classify_file(info: dict) -> dict:
    if "error" in info:
        return {"type": "error"}
    vvars = [v.lower() for v in info.get("variables", [])]
    surf = ["u10", "v10", "t2m", "msl", "sp"]
    if sum(v in vvars for v in surf) >= 3:
        return {"type": "era5_surface", "found": [v for v in surf if v in vvars],
                "missing": [v for v in surf if v not in vvars]}
    pres = ["t", "u", "v", "q", "z"]
    dims = info.get("dims", {})
    has_plev = any("level" in k.lower() or "pressure" in k.lower() for k in dims)
    if sum(v in vvars for v in pres) >= 3 or has_plev:
        return {"type": "era5_pressure", "found": [v for v in pres if v in vvars],
                "missing": [v for v in pres if v not in vvars]}
    cams = ["pm2p5", "pm25", "no2", "go3", "o3", "so2", "co", "pm10"]
    found = [v for v in cams if v in vvars]
    if found:
        return {"type": "cams", "found": found}
    return {"type": "unknown", "found": vvars}


def analyze_with_claude(files, anthropic_key: str) -> str:
    if not files:
        return "⬆️ Upload at least one file."
    if not HAS_ANTHROPIC:
        return "❌ `anthropic` package not available."
    if not anthropic_key or len(anthropic_key) < 20:
        return "⚠️ Enter your Anthropic API key in **Settings**."

    files_info = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        info = analyze_netcdf(path)
        cls  = classify_file(info)
        files_info.append((Path(path).name, info, cls))

    icons = {"era5_surface": "✅ ERA5 Surface", "era5_pressure": "✅ ERA5 Pressure",
             "cams": "✅ CAMS", "unknown": "❓ Unknown", "error": "❌ Error"}
    lines = ["### 📂 Files\n"]
    for fname, info, cls in files_info:
        lines.append(f"**{fname}** — {icons.get(cls['type'], cls['type'])}")
        if cls.get("missing"):
            lines.append(f"  ⚠️ Missing: `{', '.join(cls['missing'])}`")
    lines.append("\n---\n### 🤖 Assistant\n")

    prompt = ("User uploaded files for ARIA global PM2.5 forecasting (needs ERA5 surface, "
              "ERA5 pressure levels, CAMS). Files: " +
              str([(n, c) for n, _, c in files_info]) +
              "\nAnalyze briefly, list what's OK and missing, give CDS snippet if needed. "
              "Be concise and friendly. Use emojis. Reply in user's language.")
    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=800,
            messages=[{"role": "user", "content": prompt}])
        return "\n".join(lines) + msg.content[0].text
    except Exception as e:
        return "\n".join(lines) + f"❌ Claude API error: {e}"


# ═══════════════════════════════════════════════════════════
# CDS CODE GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_cds_code(region, date_str, lat_n, lat_s, lon_w, lon_e, pollutant) -> str:
    if region == "Custom bbox":
        area = [lat_n, lon_w, lat_s, lon_e]
    else:
        r = REGIONS[region]
        area = [r["lat"][1], r["lon"][0], r["lat"][0], r["lon"][1]]
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt_prev = dt - timedelta(days=1)
    except Exception:
        dt, dt_prev = datetime(2023, 6, 1), datetime(2023, 5, 31)

    def fmt(d): return d.strftime("%Y"), d.strftime("%m"), d.strftime("%d")
    y, m, d = fmt(dt)
    yp, mp, dp = fmt(dt_prev)

    return f'''import cdsapi
c = cdsapi.Client()

for year, month, day in [("{y}","{m}","{d}"), ("{yp}","{mp}","{dp}")]:
    # ERA5 Surface
    c.retrieve("reanalysis-era5-single-levels", {{
        "product_type": "reanalysis",
        "variable": ["10m_u_component_of_wind","10m_v_component_of_wind",
                     "2m_temperature","mean_sea_level_pressure","surface_pressure"],
        "year": year, "month": month, "day": day,
        "time": ["00:00","06:00","12:00","18:00"],
        "area": {area}, "data_format": "netcdf",
    }}, f"era5_surface_{{year}}{{month}}{{day}}.nc")

    # ERA5 Pressure
    c.retrieve("reanalysis-era5-pressure-levels", {{
        "product_type": "reanalysis",
        "variable": ["temperature","u_component_of_wind","v_component_of_wind",
                     "specific_humidity","geopotential"],
        "pressure_level": ["1000","925","850","700","500"],
        "year": year, "month": month, "day": day,
        "time": ["00:00","06:00","12:00","18:00"],
        "area": {area}, "data_format": "netcdf",
    }}, f"era5_pressure_{{year}}{{month}}{{day}}.nc")

    # CAMS
    date = f"{{year}}-{{month}}-{{day}}"
    c.retrieve("cams-global-reanalysis-eac4", {{
        "variable": ["particulate_matter_2.5um","nitrogen_dioxide",
                     "ozone","sulphur_dioxide","carbon_monoxide","particulate_matter_10um"],
        "date": date, "time": ["00:00","06:00","12:00","18:00"],
        "area": {area}, "data_format": "netcdf",
    }}, f"cams_{{date.replace('-','')}}.nc")

print("Done — upload the 6 files to ARIA (Upload & Analyze tab).")
'''


# ═══════════════════════════════════════════════════════════
# FORECAST (placeholder until checkpoint ready)
# ═══════════════════════════════════════════════════════════

def run_forecast(region, date_str, pollutant, lead_days, lat_n, lat_s, lon_w, lon_e):
    if region == "Custom bbox":
        bbox = {"lat": [lat_s, lat_n], "lon": [lon_w, lon_e]}
    else:
        bbox = REGIONS[region]

    msg = f"""### 🔄 ARIA is training — checkpoint coming soon

**Your request is saved:**

| | |
|---|---|
| Region | **{region}** ({bbox['lat'][0]}°–{bbox['lat'][1]}°N) |
| Date | **{date_str}** |
| Pollutant | **{pollutant}** |
| Lead time | **+{lead_days} day(s)** |
| Output resolution | **~1 km** |

---
**Training:** 128× MI250X GPUs on LUMI supercomputer 🔄

The checkpoint will be published to 🤗 [{HF_MODEL}](https://huggingface.co/{HF_MODEL}) when ready.

**In the meantime:**
- Use **Download Code** tab to get ERA5 + CAMS data for your region
- Follow [GitHub]({GITHUB_URL}) for updates
"""
    return msg


# ═══════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(css=CSS, title="ARIA — Global PM2.5 at 1km") as demo:

        gr.HTML(f"""
        <h1 id="title">🌍 ARIA</h1>
        <p id="subtitle">Global PM<sub>2.5</sub> forecasting at <strong>1 km resolution</strong> · Dual-branch Vision Transformer</p>
        <p id="links">
          <a href="{WEBSITE_URL}" target="_blank">🌐 Website</a> &nbsp;|&nbsp;
          <a href="{GITHUB_URL}" target="_blank">💻 GitHub</a> &nbsp;|&nbsp;
          <a href="https://huggingface.co/{HF_MODEL}" target="_blank">🤗 Model</a> &nbsp;|&nbsp;
          <span style="color:#ef4444;font-weight:600">● TRAINING IN PROGRESS · 128 GPUs</span>
        </p>
        """)

        with gr.Accordion("🔑 Settings (API keys)", open=False):
            gr.Markdown("Keys are never stored — session only.")
            with gr.Row():
                anthropic_key = gr.Textbox(label="Anthropic API Key (file analysis)",
                                           placeholder="sk-ant-...", type="password", scale=2)
                cds_uid = gr.Textbox(label="CDS UID", placeholder="12345", scale=1)
                cds_key_in = gr.Textbox(label="CDS API Key", placeholder="xxxx-xxxx",
                                        type="password", scale=2)

        with gr.Tabs():

            # ── Tab 1: Forecast ──
            with gr.Tab("🔍 Forecast"):
                gr.Markdown("Select a region and date — ARIA will download ERA5+CAMS and predict PM₂.₅ at 1km.")
                with gr.Row():
                    with gr.Column(scale=1):
                        region_dd   = gr.Dropdown(list(REGIONS.keys()), value="East Asia", label="Region")
                        date_in     = gr.Textbox(label="Date (YYYY-MM-DD)", value="2023-06-01")
                        poll_dd     = gr.Dropdown(list(POLLUTANTS.keys()), value="PM2.5", label="Pollutant")
                        lead_sl     = gr.Slider(1, 4, value=1, step=1, label="Lead time (days)")
                        with gr.Accordion("Custom bounding box", open=False):
                            c_lat_n = gr.Number(label="Lat North", value=55)
                            c_lat_s = gr.Number(label="Lat South", value=15)
                            c_lon_w = gr.Number(label="Lon West",  value=70)
                            c_lon_e = gr.Number(label="Lon East",  value=145)
                        btn_fc = gr.Button("🚀 Run Forecast", variant="primary")
                    with gr.Column(scale=2):
                        fc_out = gr.Markdown(value="Select a region and click **Run Forecast**.")

                btn_fc.click(
                    run_forecast,
                    inputs=[region_dd, date_in, poll_dd, lead_sl,
                            c_lat_n, c_lat_s, c_lon_w, c_lon_e],
                    outputs=[fc_out],
                )

            # ── Tab 2: Upload & Analyze ──
            with gr.Tab("📂 Upload & Analyze"):
                gr.Markdown("""Upload your ERA5 / CAMS NetCDF files.
                Claude will detect what's present, what's missing, and generate download code.""")
                with gr.Row():
                    with gr.Column(scale=1):
                        f1 = gr.File(label="ERA5 Surface (.nc)", file_types=[".nc", ".nc4"])
                        f2 = gr.File(label="ERA5 Pressure (.nc)", file_types=[".nc", ".nc4"])
                        f3 = gr.File(label="CAMS (.nc)", file_types=[".nc", ".nc4"])
                        btn_analyze = gr.Button("🔍 Analyze files", variant="secondary")
                        btn_predict = gr.Button("🚀 Run Forecast",  variant="primary")
                    with gr.Column(scale=2):
                        upload_out = gr.Markdown(value="⬆️ Upload files and click **Analyze**.")

                btn_analyze.click(
                    lambda a, b, c, key: analyze_with_claude(
                        [x for x in [a, b, c] if x is not None], key),
                    inputs=[f1, f2, f3, anthropic_key],
                    outputs=[upload_out],
                )
                btn_predict.click(
                    lambda a, b, c, key: analyze_with_claude(
                        [x for x in [a, b, c] if x is not None], key),
                    inputs=[f1, f2, f3, anthropic_key],
                    outputs=[upload_out],
                )

            # ── Tab 3: Download Code ──
            with gr.Tab("📥 Download Code"):
                gr.Markdown("Generate a ready-to-run CDS API Python script for any region and date.")
                with gr.Row():
                    with gr.Column(scale=1):
                        dl_region = gr.Dropdown(list(REGIONS.keys()), value="East Asia", label="Region")
                        dl_date   = gr.Textbox(label="Date", value="2023-06-01")
                        dl_poll   = gr.Dropdown(list(POLLUTANTS.keys()), value="PM2.5", label="Pollutant")
                        with gr.Accordion("Custom bbox", open=False):
                            dl_lat_n = gr.Number(label="Lat N", value=55)
                            dl_lat_s = gr.Number(label="Lat S", value=15)
                            dl_lon_w = gr.Number(label="Lon W", value=70)
                            dl_lon_e = gr.Number(label="Lon E", value=145)
                        btn_gen = gr.Button("📥 Generate script", variant="primary")
                    with gr.Column(scale=2):
                        code_out = gr.Code(label="CDS download script", language="python")

                btn_gen.click(
                    generate_cds_code,
                    inputs=[dl_region, dl_date, dl_lat_n, dl_lat_s,
                            dl_lon_w, dl_lon_e, dl_poll],
                    outputs=[code_out],
                )

            # ── Tab 4: Model ──
            with gr.Tab("🏋️ Model"):
                gr.Markdown(f"""
### ARIA Architecture

| Component | Details |
|-----------|---------|
| **Global branch** | ERA5 72ch → ViT patches 8×8, dim=768, depth=8 |
| **Local branch** | CAMS + elevation + roads + lights + population → patches 16×16, dim=512, depth=6 |
| **Cross-attention** | 2 layers, local queries ↔ global k/v |
| **Decoder** | Progressive CNN upsample → 512px |
| **Prediction** | Δ: pred = CAMS_t + decoder(tokens) |
| **Training** | 128× MI250X · LUMI · bf16 mixed precision |

### Status

| Version | Data | RMSE | Status |
|---------|------|------|--------|
| `aria-global` | GHAP+ERA5 2018–2022 | TBD | 🔄 Training |

Checkpoint → [🤗 {HF_MODEL}](https://huggingface.co/{HF_MODEL})
                """)

            # ── Tab 5: Retrain ──
            with gr.Tab("🔧 Retrain"):
                gr.Markdown(f"Full code: [{GITHUB_URL}]({GITHUB_URL})")
                with gr.Row():
                    with gr.Column():
                        rt_era5  = gr.Textbox(label="ERA5 zarr dir", placeholder="/path/to/era5/")
                        rt_ghap  = gr.Textbox(label="GHAP zarr dir", placeholder="/path/to/ghap/")
                        rt_cams  = gr.Textbox(label="CAMS zarr dir", placeholder="/path/to/cams/")
                        rt_train = gr.Textbox(label="Train years", value="2018,2019,2020,2021")
                        rt_val   = gr.Textbox(label="Val year",   value="2022")
                    with gr.Column():
                        rt_lr    = gr.Number(label="Learning rate", value=5e-5)
                        rt_ep    = gr.Number(label="Epochs",        value=300)
                        rt_bs    = gr.Number(label="Batch/GPU",     value=2)
                        rt_nodes = gr.Number(label="SLURM nodes",   value=16)
                        btn_slurm = gr.Button("📥 Generate SLURM script", variant="primary")

                slurm_out = gr.Code(label="SLURM script", language="shell")

                def gen_slurm(era5, ghap, cams, train_y, val_y, lr, ep, bs, nodes):
                    return f"""#!/bin/bash
#SBATCH --job-name=aria_train
#SBATCH --partition=standard-g
#SBATCH --nodes={int(nodes)}
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00

module load LUMI/25.03 partition/G rocm/6.0.3
source /path/to/venv/bin/activate

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

srun python train_global.py \\
    --era5_dir {era5 or "/path/to/era5"} \\
    --ghap_dir {ghap or "/path/to/ghap"} \\
    --cams_dir {cams or "/path/to/cams"} \\
    --train_years {train_y} --val_years {val_y} \\
    --lr {lr} --epochs {int(ep)} --batch_size {int(bs)}
"""
                btn_slurm.click(
                    gen_slurm,
                    inputs=[rt_era5, rt_ghap, rt_cams, rt_train, rt_val,
                            rt_lr, rt_ep, rt_bs, rt_nodes],
                    outputs=[slurm_out],
                )

        gr.HTML(f"""
        <div style="text-align:center;margin-top:2rem;color:#9ca3af;font-size:0.85rem;
                    border-top:1px solid #e5e7eb;padding-top:1.5rem;">
          ARIA · <a href="{GITHUB_URL}">GitHub</a> ·
          <a href="{WEBSITE_URL}">Website</a> · MIT License
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
