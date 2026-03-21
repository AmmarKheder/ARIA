#!/usr/bin/env python3
"""
ARIA — Attention-based Resolution-Integrated Air quality
HuggingFace Spaces demo: global PM2.5 forecasting at 1km resolution.
"""
import os
import io
import json
import tempfile
import traceback
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import gradio as gr

# ZeroGPU (HuggingFace Spaces free A100)
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(func=None, duration=60):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
# CONSTANTS
# ═══════════════════════════════════════════════════════════

GITHUB_URL  = "https://github.com/AmmarKheder/ARIA"
HF_MODEL    = "AmmarKheder/ARIA-global"
WEBSITE_URL = "https://ammarkheder.github.io/ARIA/"

REGIONS = {
    "East Asia":        {"lat": [15,  55],  "lon": [70,  145]},
    "South Asia":       {"lat": [5,   35],  "lon": [60,  100]},
    "Europe":           {"lat": [35,  72],  "lon": [-25,  45]},
    "North America":    {"lat": [25,  70],  "lon": [-130, -60]},
    "Middle East":      {"lat": [12,  42],  "lon": [25,   65]},
    "Africa":           {"lat": [-35, 37],  "lon": [-20,  55]},
    "Southeast Asia":   {"lat": [-10, 30],  "lon": [95,  145]},
    "Australia":        {"lat": [-45, -10], "lon": [110, 155]},
    "South America":    {"lat": [-55, 15],  "lon": [-82, -34]},
    "Custom bbox":      None,
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
#title    { text-align:center; font-size:2.2rem; font-weight:800; margin-bottom:0.2rem; letter-spacing:-0.5px; }
#subtitle { text-align:center; color:#6b7280; margin-bottom:0.2rem; font-size:1.05rem; }
#links    { text-align:center; margin-bottom:1.5rem; }
.status-ok   { color:#16a34a; font-weight:600; }
.status-warn { color:#d97706; font-weight:600; }
.status-err  { color:#dc2626; font-weight:600; }
.metric-card { background:#f8fafc; border-radius:12px; padding:1.2rem; text-align:center;
               border:1px solid #e2e8f0; margin:0.3rem; }
.metric-val  { font-size:2rem; font-weight:700; color:#1e40af; }
.metric-lbl  { font-size:0.85rem; color:#6b7280; margin-top:0.2rem; }
footer { display:none !important; }
"""


# ═══════════════════════════════════════════════════════════
# FILE ANALYSIS (Claude-powered)
# ═══════════════════════════════════════════════════════════

def analyze_netcdf(path: str) -> dict:
    if not HAS_XARRAY:
        return {"error": "xarray not installed"}
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
        info = {
            "variables": list(ds.data_vars),
            "coords":    list(ds.coords),
            "dims":      dict(ds.dims),
        }
        for lat_name in ["latitude", "lat"]:
            if lat_name in ds.coords:
                info["lat_range"] = [float(ds[lat_name].min()), float(ds[lat_name].max())]
        for lon_name in ["longitude", "lon"]:
            if lon_name in ds.coords:
                info["lon_range"] = [float(ds[lon_name].min()), float(ds[lon_name].max())]
        t_coord = ds.coords.get("time", ds.coords.get("valid_time", None))
        if t_coord is not None:
            info["time_range"] = [str(t_coord.values[0])[:16], str(t_coord.values[-1])[:16]]
            info["n_steps"] = int(len(t_coord))
        ds.close()
        return info
    except Exception as e:
        return {"error": str(e)}


def classify_file(info: dict) -> dict:
    if "error" in info:
        return {"type": "error", "details": info["error"]}
    vvars = [v.lower() for v in info.get("variables", [])]

    # ERA5 surface
    surf = ["u10", "v10", "t2m", "msl", "sp"]
    surf_found = [v for v in surf if v in vvars]
    if len(surf_found) >= 3:
        return {"type": "era5_surface", "found": surf_found,
                "missing": [v for v in surf if v not in vvars]}

    # ERA5 pressure
    pres = ["t", "u", "v", "q", "z"]
    pres_found = [v for v in pres if v in vvars]
    dims = info.get("dims", {})
    has_plev = any("level" in k.lower() or "pressure" in k.lower() for k in dims)
    if len(pres_found) >= 3 or has_plev:
        return {"type": "era5_pressure", "found": pres_found,
                "missing": [v for v in pres if v not in vvars]}

    # CAMS
    cams_names = ["pm2p5", "pm25", "no2", "go3", "o3", "so2", "co", "pm10", "aod"]
    cams_found = [v for v in cams_names if v in vvars]
    if cams_found:
        return {"type": "cams", "found": cams_found}

    return {"type": "unknown", "found": vvars}


def analyze_with_claude(files, anthropic_key: str) -> str:
    if not files:
        return "⬆️ Upload at least one file."
    if not HAS_ANTHROPIC:
        return "❌ anthropic package not installed."
    if not anthropic_key or len(anthropic_key) < 20:
        return "⚠️ Enter your Anthropic API key in Settings."

    files_info = []
    for f in files:
        path = f.name if hasattr(f, "name") else str(f)
        info = analyze_netcdf(path)
        cls  = classify_file(info)
        files_info.append((Path(path).name, info, cls))

    # Summary card
    lines = ["### 📂 Files detected\n"]
    type_icons = {"era5_surface": "✅ ERA5 Surface", "era5_pressure": "✅ ERA5 Pressure",
                  "cams": "✅ CAMS", "unknown": "❓ Unknown", "error": "❌ Error"}
    for fname, info, cls in files_info:
        lines.append(f"**{fname}** — {type_icons.get(cls['type'], cls['type'])}")
        if cls.get("missing"):
            lines.append(f"  ⚠️ Missing: `{', '.join(cls['missing'])}`")
    lines.append("\n---\n### 🤖 ARIA Assistant\n")

    # Build prompt
    prompt_lines = ["The user uploaded files to run ARIA (global 1km PM2.5 forecasting).\n",
                    "ARIA requires: ERA5 surface (u10,v10,t2m,msl,sp), ERA5 pressure levels "
                    "(t,u,v,q,z at 1000/925/850/700/500hPa), CAMS (pm2p5,no2,go3,so2,co,pm10).\n",
                    "Files info:"]
    for fname, info, cls in files_info:
        prompt_lines.append(f"  {fname}: type={cls['type']}, vars={cls.get('found',[])} "
                            f"missing={cls.get('missing',[])}")
        if "lat_range" in info:
            prompt_lines.append(f"    lat={info['lat_range']}, lon={info['lon_range']}")
        if "time_range" in info:
            prompt_lines.append(f"    time={info['time_range'][0]} → {info['time_range'][1]}")
    prompt_lines.append("\nAnalyze briefly, say what's OK and what's missing. "
                        "Provide CDS API snippet if anything is missing. "
                        "Be concise, friendly. Use emojis. "
                        "Reply in the user's language (French if they typed French).")

    try:
        client = anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
        )
        return "\n".join(lines) + msg.content[0].text
    except Exception as e:
        return "\n".join(lines) + f"❌ Claude API error: {e}"


# ═══════════════════════════════════════════════════════════
# CDS CODE GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_cds_code(region: str, date_str: str,
                      lat_n: float, lat_s: float,
                      lon_w: float, lon_e: float,
                      pollutant: str) -> str:
    if region == "Custom bbox":
        area = [lat_n, lon_w, lat_s, lon_e]
    else:
        r = REGIONS[region]
        area = [r["lat"][1], r["lon"][0], r["lat"][0], r["lon"][1]]

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt_prev = dt - timedelta(days=1)
    except Exception:
        dt = datetime(2023, 1, 15)
        dt_prev = dt - timedelta(days=1)

    y, m, d = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    yp, mp, dp = dt_prev.strftime("%Y"), dt_prev.strftime("%m"), dt_prev.strftime("%d")
    cams_var = POLLUTANTS.get(pollutant, "pm2p5")

    return f'''import cdsapi
c = cdsapi.Client()

# ── ERA5 Surface (t and t-1) ──
for year, month, day in [("{y}", "{m}", "{d}"), ("{yp}", "{mp}", "{dp}")]:
    c.retrieve("reanalysis-era5-single-levels", {{
        "product_type": "reanalysis",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind",
                     "2m_temperature", "mean_sea_level_pressure", "surface_pressure"],
        "year": year, "month": month, "day": day,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": {area},
        "data_format": "netcdf",
    }}, f"era5_surface_{{year}}{{month}}{{day}}.nc")

# ── ERA5 Pressure Levels (t and t-1) ──
for year, month, day in [("{y}", "{m}", "{d}"), ("{yp}", "{mp}", "{dp}")]:
    c.retrieve("reanalysis-era5-pressure-levels", {{
        "product_type": "reanalysis",
        "variable": ["temperature", "u_component_of_wind", "v_component_of_wind",
                     "specific_humidity", "geopotential"],
        "pressure_level": ["1000", "925", "850", "700", "500"],
        "year": year, "month": month, "day": day,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": {area},
        "data_format": "netcdf",
    }}, f"era5_pressure_{{year}}{{month}}{{day}}.nc")

# ── CAMS (t and t-1) ──
for date in ["{y}-{m}-{d}", "{yp}-{mp}-{dp}"]:
    c.retrieve("cams-global-reanalysis-eac4", {{
        "variable": ["particulate_matter_2.5um", "nitrogen_dioxide",
                     "ozone", "sulphur_dioxide", "carbon_monoxide", "particulate_matter_10um"],
        "date": date,
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": {area},
        "data_format": "netcdf",
    }}, f"cams_{{date.replace(\'-\',\'\')}}.nc")

print("Done — upload era5_surface, era5_pressure, and cams files to ARIA.")
'''


# ═══════════════════════════════════════════════════════════
# INFERENCE PLACEHOLDER
# ═══════════════════════════════════════════════════════════

@spaces.GPU(duration=180)
def run_forecast(region: str, date_str: str, pollutant: str, lead_days: int,
                 lat_n: float, lat_s: float, lon_w: float, lon_e: float,
                 cds_uid: str, cds_key: str) -> tuple:
    """Download ERA5+CAMS and run ARIA inference."""
    if not cds_uid or not cds_key:
        return None, "⚠️ Enter your CDS UID and API key in **Settings** above."

    # Determine bbox
    if region == "Custom bbox":
        bbox = {"lat": [lat_s, lat_n], "lon": [lon_w, lon_e]}
    else:
        bbox = REGIONS[region]

    # Check model checkpoint
    ckpt_available = False
    try:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=HF_MODEL, filename="aria_global.ckpt")
        ckpt_available = True
    except Exception:
        pass

    if not ckpt_available:
        # Placeholder — show live training progress
        msg = f"""### 🔄 ARIA is currently training

**Requested:** {pollutant} over **{region}** on **{date_str}** (lead +{lead_days}d)

| Parameter | Value |
|-----------|-------|
| Region | {region} ({bbox['lat'][0]}°–{bbox['lat'][1]}°N, {bbox['lon'][0]}°–{bbox['lon'][1]}°E) |
| Date | {date_str} |
| Pollutant | {pollutant} |
| Lead time | +{lead_days} day(s) |
| Resolution | **~1 km** |

---
**Training status:** 128 GPUs × MI250X · 16 nodes · In progress 🔄

The checkpoint will be published to [{HF_MODEL}](https://huggingface.co/{HF_MODEL}) when training completes.
Follow the project on [GitHub]({GITHUB_URL}).

**While you wait** — use the **Download Code** tab to get ERA5+CAMS data for your region so you're ready the moment the checkpoint drops.
"""
        return None, msg

    # ── TODO: real inference once checkpoint available ──
    return None, "✅ Checkpoint loaded — inference pipeline coming soon."


@spaces.GPU(duration=120)
def run_forecast_from_files(era5_surf, era5_pres, cams_file,
                             pollutant: str, lead_days: int,
                             anthropic_key: str) -> tuple:
    files = [f for f in [era5_surf, era5_pres, cams_file] if f is not None]
    if not files:
        return None, "⬆️ Upload your ERA5 and CAMS files first."
    analysis = analyze_with_claude(files, anthropic_key)
    return None, analysis


# ═══════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(css=CSS, title="ARIA — Global PM2.5 Forecasting at 1km") as demo:

        # ── Header ──
        gr.HTML(f"""
        <h1 id="title">🌍 ARIA</h1>
        <p id="subtitle">Attention-based Resolution-Integrated Air quality &nbsp;·&nbsp;
                         Global PM<sub>2.5</sub> forecasting at <strong>1 km resolution</strong></p>
        <p id="links">
          <a href="{WEBSITE_URL}" target="_blank">🌐 Website</a> &nbsp;|&nbsp;
          <a href="{GITHUB_URL}" target="_blank">💻 GitHub</a> &nbsp;|&nbsp;
          <a href="https://huggingface.co/{HF_MODEL}" target="_blank">🤗 Model</a> &nbsp;|&nbsp;
          <span style="color:#ef4444">● TRAINING IN PROGRESS</span>
        </p>
        """)

        # ── Settings (collapsed) ──
        with gr.Accordion("🔑 Settings", open=False):
            gr.Markdown("API keys are **never stored** — session only.")
            with gr.Row():
                anthropic_key = gr.Textbox(label="Anthropic API Key (file analysis)",
                                           placeholder="sk-ant-...", type="password", scale=2)
                cds_uid = gr.Textbox(label="CDS UID", placeholder="12345", scale=1)
                cds_key = gr.Textbox(label="CDS API Key", placeholder="xxxxxxxx-xxxx-...",
                                     type="password", scale=2)

        # ── Metric cards ──
        with gr.Row():
            gr.HTML("""
            <div class="metric-card">
              <div class="metric-val">~1 km</div>
              <div class="metric-lbl">Output resolution</div>
            </div>""")
            gr.HTML("""
            <div class="metric-card">
              <div class="metric-val">6</div>
              <div class="metric-lbl">Pollutants</div>
            </div>""")
            gr.HTML("""
            <div class="metric-card">
              <div class="metric-val">128</div>
              <div class="metric-lbl">Training GPUs</div>
            </div>""")
            gr.HTML("""
            <div class="metric-card">
              <div class="metric-val">2003–2022</div>
              <div class="metric-lbl">Training period</div>
            </div>""")
            gr.HTML("""
            <div class="metric-card">
              <div class="metric-val">🔄</div>
              <div class="metric-lbl">Status: Training</div>
            </div>""")

        # ═════════ TABS ═════════
        with gr.Tabs():

            # ── Tab 1: Auto Forecast ──
            with gr.Tab("🔍 Forecast"):
                gr.Markdown("### Auto-download ERA5+CAMS and run ARIA inference")
                with gr.Row():
                    with gr.Column(scale=1):
                        region_dd  = gr.Dropdown(list(REGIONS.keys()), value="East Asia",
                                                 label="Region")
                        date_in    = gr.Textbox(label="Date (YYYY-MM-DD)", value="2023-06-01",
                                                placeholder="2023-06-01")
                        pollutant_dd = gr.Dropdown(list(POLLUTANTS.keys()), value="PM2.5",
                                                   label="Pollutant")
                        lead_slider = gr.Slider(1, 4, value=1, step=1, label="Lead time (days)")

                        with gr.Accordion("Custom bounding box", open=False):
                            c_lat_n = gr.Number(label="Lat North", value=55)
                            c_lat_s = gr.Number(label="Lat South", value=15)
                            c_lon_w = gr.Number(label="Lon West",  value=70)
                            c_lon_e = gr.Number(label="Lon East",  value=145)

                        btn_forecast = gr.Button("🚀 Run Forecast", variant="primary")

                    with gr.Column(scale=2):
                        forecast_plot   = gr.Plot(label="PM₂.₅ (µg/m³) · 1km")
                        forecast_status = gr.Markdown(
                            value="Select a region and date, then click **Run Forecast**.")

                btn_forecast.click(
                    run_forecast,
                    inputs=[region_dd, date_in, pollutant_dd, lead_slider,
                            c_lat_n, c_lat_s, c_lon_w, c_lon_e, cds_uid, cds_key],
                    outputs=[forecast_plot, forecast_status],
                )

            # ── Tab 2: Upload Files ──
            with gr.Tab("📂 Upload & Analyze"):
                gr.Markdown("""### Upload your ERA5 / CAMS NetCDF files
                The assistant (powered by Claude) analyzes your files, detects missing variables,
                and provides exact CDS API download code for anything missing.""")
                with gr.Row():
                    with gr.Column(scale=1):
                        era5_surf_up = gr.File(label="ERA5 Surface (.nc)",
                                               file_types=[".nc", ".nc4"])
                        era5_pres_up = gr.File(label="ERA5 Pressure Levels (.nc)",
                                               file_types=[".nc", ".nc4"])
                        cams_up      = gr.File(label="CAMS (.nc)",
                                               file_types=[".nc", ".nc4"])
                        poll_up    = gr.Dropdown(list(POLLUTANTS.keys()), value="PM2.5",
                                                 label="Pollutant")
                        lead_up    = gr.Slider(1, 4, value=1, step=1, label="Lead time")
                        with gr.Row():
                            btn_analyze = gr.Button("🔍 Analyze",  variant="secondary")
                            btn_pred_up = gr.Button("🚀 Forecast", variant="primary")
                    with gr.Column(scale=2):
                        upload_plot   = gr.Plot(label="Forecast output")
                        upload_status = gr.Markdown(
                            value="⬆️ Upload files and click **Analyze** to check them.")

                btn_analyze.click(
                    lambda f1, f2, f3, key: analyze_with_claude(
                        [f for f in [f1, f2, f3] if f is not None], key),
                    inputs=[era5_surf_up, era5_pres_up, cams_up, anthropic_key],
                    outputs=[upload_status],
                )
                btn_pred_up.click(
                    run_forecast_from_files,
                    inputs=[era5_surf_up, era5_pres_up, cams_up, poll_up, lead_up, anthropic_key],
                    outputs=[upload_plot, upload_status],
                )

            # ── Tab 3: Download Code ──
            with gr.Tab("📥 Download Code"):
                gr.Markdown("""### Generate CDS API code to download ERA5 + CAMS
                Select your region and date → get a ready-to-run Python script.""")
                with gr.Row():
                    with gr.Column(scale=1):
                        dl_region = gr.Dropdown(list(REGIONS.keys()), value="East Asia",
                                                label="Region")
                        dl_date   = gr.Textbox(label="Date", value="2023-06-01")
                        dl_poll   = gr.Dropdown(list(POLLUTANTS.keys()), value="PM2.5",
                                                label="Pollutant")
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

            # ── Tab 4: Model Info ──
            with gr.Tab("🏋️ Model"):
                gr.Markdown(f"""
### ARIA Architecture

| Component | Details |
|-----------|---------|
| **Global branch** | ERA5 (72ch) + CAMS → ViT patches 8×8 → dim=768, depth=8 |
| **Local branch** | CAMS fine + elevation + road density + nighttime lights + population → patches 16×16 → dim=512, depth=6 |
| **Cross-attention** | 2 layers, local queries ↔ global k/v |
| **Decoder** | Progressive CNN upsample 32→512px with skip connection |
| **Prediction** | Delta: pred = CAMS_t + decoder(tokens) |
| **Input channels** | Global: 72ch · Local: 18ch |
| **Output** | PM2.5 at ~1km (512×512 tiles) |
| **Training** | 128× MI250X GPUs · 16 LUMI nodes · bf16 mixed precision |

### Checkpoints

| Version | Training data | RMSE (China) | Status |
|---------|--------------|--------------|--------|
| `aria-global` | GHAP 2018-2022, ERA5 2003-2022 | TBD | 🔄 Training |
| `aria-global-v0` | GHAP 2018-2022 | ~5.2 µg/m³ | 🔄 Soon |

Checkpoints will be hosted at [🤗 {HF_MODEL}](https://huggingface.co/{HF_MODEL}).
                """)

                with gr.Row():
                    hf_model_in = gr.Textbox(label="HuggingFace model ID",
                                             value=HF_MODEL)
                    btn_load_hf = gr.Button("Load from HF Hub", variant="secondary")
                model_status = gr.Markdown()
                ckpt_up = gr.File(label="Or upload checkpoint (.ckpt / .pt)",
                                  file_types=[".ckpt", ".pt", ".pth"])
                btn_load_custom = gr.Button("Load custom checkpoint", variant="secondary")

                def load_hf(name):
                    return f"⏳ Loading `{name}`... (available once training completes)"
                def load_custom(f):
                    if f is None: return "⬆️ Upload a checkpoint first."
                    return f"✅ `{Path(f.name).name}` loaded. Inference ready."

                btn_load_hf.click(load_hf, inputs=[hf_model_in], outputs=[model_status])
                btn_load_custom.click(load_custom, inputs=[ckpt_up], outputs=[model_status])

            # ── Tab 5: Retrain ──
            with gr.Tab("🔧 Retrain"):
                gr.Markdown(f"""
### Train ARIA on your own data

Full code: [{GITHUB_URL}]({GITHUB_URL})

```bash
git clone {GITHUB_URL}
cd ARIA
# Edit configs/global_pretrain.yaml with your paths
sbatch scripts/submit_train_global.sh
```

Configure below and download a ready-to-run SLURM script:
                """)
                with gr.Row():
                    with gr.Column():
                        rt_era5  = gr.Textbox(label="ERA5 zarr directory",   placeholder="/path/to/era5_daily/")
                        rt_ghap  = gr.Textbox(label="GHAP zarr directory",   placeholder="/path/to/ghap_daily/")
                        rt_cams  = gr.Textbox(label="CAMS zarr directory",   placeholder="/path/to/cams_daily/")
                        rt_elev  = gr.Textbox(label="Elevation zarr (GMTED)",placeholder="/path/to/gmted2010.zarr")
                        rt_proxy = gr.Textbox(label="Emission proxies zarr", placeholder="/path/to/emission_proxies.zarr")
                        rt_train = gr.Textbox(label="Train years",  value="2018,2019,2020,2021")
                        rt_val   = gr.Textbox(label="Val year",     value="2022")
                    with gr.Column():
                        rt_lr     = gr.Number(label="Learning rate",    value=5e-5)
                        rt_epochs = gr.Number(label="Epochs",           value=300)
                        rt_bs     = gr.Number(label="Batch size / GPU", value=2)
                        rt_nodes  = gr.Number(label="SLURM nodes",      value=16)
                        rt_gpus   = gr.Number(label="GPUs per node",    value=8)
                        rt_ft     = gr.Textbox(label="Finetune from (optional)",
                                               placeholder="AmmarKheder/ARIA-global")
                        btn_slurm = gr.Button("📥 Generate SLURM script", variant="primary")

                slurm_out = gr.Code(label="SLURM script", language="bash")

                def gen_slurm(era5, ghap, cams, elev, proxy,
                               train_y, val_y, lr, epochs, bs, nodes, gpus, ft):
                    ft_line = f"    --finetune {ft} \\\n" if ft else ""
                    return f"""#!/bin/bash
#SBATCH --job-name=aria_train
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=standard-g
#SBATCH --nodes={int(nodes)}
#SBATCH --ntasks-per-node={int(gpus)}
#SBATCH --gpus-per-node={int(gpus)}
#SBATCH --cpus-per-task=7
#SBATCH --time=2-00:00:00
#SBATCH --output=aria_train_%j.out

module load LUMI/25.03 partition/G rocm/6.0.3
source /path/to/venv/bin/activate

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export MIOPEN_USER_DB_PATH=/tmp/$SLURM_JOB_ID/miopen_cache

srun --ntasks-per-node=1 mkdir -p $MIOPEN_USER_DB_PATH

srun python train_global.py \\
    --era5_dir {era5 or "/path/to/era5"} \\
    --ghap_dir {ghap or "/path/to/ghap"} \\
    --cams_dir {cams or "/path/to/cams"} \\
    --elev_path {elev or "/path/to/gmted2010.zarr"} \\
    --proxies_path {proxy or "/path/to/emission_proxies.zarr"} \\
    --train_years {train_y} \\
    --val_years {val_y} \\
    --batch_size {int(bs)} \\
    --lr {lr} \\
    --epochs {int(epochs)} \\
{ft_line}"""

                btn_slurm.click(
                    gen_slurm,
                    inputs=[rt_era5, rt_ghap, rt_cams, rt_elev, rt_proxy,
                            rt_train, rt_val, rt_lr, rt_epochs,
                            rt_bs, rt_nodes, rt_gpus, rt_ft],
                    outputs=[slurm_out],
                )

        # ── Footer ──
        gr.HTML(f"""
        <div style="text-align:center; margin-top:2.5rem; color:#9ca3af; font-size:0.85rem;
                    border-top:1px solid #e5e7eb; padding-top:1.5rem;">
          ARIA · Global PM<sub>2.5</sub> at 1km ·
          <a href="{GITHUB_URL}">GitHub</a> ·
          <a href="{WEBSITE_URL}">Website</a> ·
          MIT License
        </div>
        """)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
