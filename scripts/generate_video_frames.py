#!/usr/bin/env python3
"""
ARIA vs CAMS — Video Frame Generator
=====================================
Generates one 300-DPI PNG per day from a comparison .npz produced by
zero_shot_inference_2025.py, then assembles with ffmpeg.

Output layout (side-by-side + diff):
  ┌─────────────────────────────────────────────────────────┐
  │   CAMS Operational (80 km)  │   ARIA Foundation (1 km)  │
  │                             │                           │
  │         [colorbar]          │         [colorbar]         │
  └─────────────────────────────────────────────────────────┘

Usage:
  # Generate frames
  python generate_video_frames.py \
      --input  /scratch/.../zeroshot_2025_comparison.npz \
      --outdir /scratch/.../frames \
      --region global            # or: europe / china / india

  # Assemble video (run locally after rsync)
  ffmpeg -framerate 12 -pattern_type glob -i 'frames/*.png' \
         -vcodec libx264 -pix_fmt yuv420p -crf 18 aria_vs_cams.mp4
"""
import argparse
import numpy as np
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

# ── Matplotlib / Cartopy ──────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not found — using plain imshow (no map projection)")

# ── Regions ───────────────────────────────────────────────────────────────────
REGIONS = {
    "global":  dict(lon0=-180, lon1=180,  lat0=-60,  lat1=85),
    "europe":  dict(lon0=-25,  lon1=45,   lat0=34,   lat1=72),
    "china":   dict(lon0=73,   lon1=135,  lat0=18,   lat1=54),
    "india":   dict(lon0=67,   lon1=97,   lat0=6,    lat1=38),
    "eastasia":dict(lon0=100,  lon1=145,  lat0=20,   lat1=50),
    "us":      dict(lon0=-130, lon1=-60,  lat0=24,   lat1=52),
    "mena":    dict(lon0=-20,  lon1=65,   lat0=10,   lat1=42),
}

# ── Major cities (lon, lat, label) ───────────────────────────────────────────
CITIES = [
    (2.35,   48.85, "Paris"),
    (13.40,  52.52, "Berlin"),
    (28.97,  41.01, "Istanbul"),
    (37.62,  55.75, "Moscow"),
    (72.88,  19.08, "Mumbai"),
    (77.21,  28.61, "Delhi"),
    (88.36,  22.57, "Kolkata"),
    (104.07, 30.67, "Chengdu"),
    (116.40, 39.90, "Beijing"),
    (121.47, 31.23, "Shanghai"),
    (126.98, 37.57, "Seoul"),
    (139.69, 35.69, "Tokyo"),
    (31.24,  30.04, "Cairo"),
    (3.39,   6.45,  "Lagos"),
    (-43.18, -22.91,"Rio"),
    (-58.38, -34.61,"Buenos Aires"),
    (-99.13, 19.43, "Mexico City"),
    (-87.63, 41.88, "Chicago"),
    (-74.01, 40.71, "New York"),
    (-118.24,34.05, "Los Angeles"),
]

# ── PM2.5 colormap: WHO levels ────────────────────────────────────────────────
CMAP   = "RdYlBu_r"
VMIN   = 0.0
VMAX   = 80.0   # µg/m³  (saturates at severe pollution)

# WHO annual guideline = 5, daily = 15
WHO_LEVELS = [0, 5, 15, 35, 75, 150]


def _norm():
    return mcolors.Normalize(vmin=VMIN, vmax=VMAX)


def _make_map_ax(fig, rect, region):
    """Return a GeoAxes (cartopy) or plain Axes (fallback)."""
    r = REGIONS[region]
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        ax = fig.add_axes(rect, projection=proj)
        ax.set_extent([r["lon0"], r["lon1"], r["lat0"], r["lat1"]], crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="0.4")
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="0.5",
                       linestyle="--")
        ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f0", zorder=0)
        ax.add_feature(cfeature.LAND,  facecolor="#f5f0e8", zorder=0)
        ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor="#7ec8e3",
                       alpha=0.5)
        return ax
    else:
        ax = fig.add_axes(rect)
        return ax


def _draw_cities(ax, region, fontsize=5.5):
    r = REGIONS[region]
    for lon, lat, name in CITIES:
        if r["lon0"] <= lon <= r["lon1"] and r["lat0"] <= lat <= r["lat1"]:
            kwargs = {}
            if HAS_CARTOPY:
                kwargs["transform"] = ccrs.PlateCarree()
            ax.plot(lon, lat, "o", ms=1.8, color="k", zorder=5, **kwargs)
            ax.text(lon + 0.5, lat + 0.5, name, fontsize=fontsize,
                    color="k", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.6),
                    **kwargs)


def _imshow_on_ax(ax, data, region, **kwargs):
    r = REGIONS[region]
    if HAS_CARTOPY:
        ax.imshow(data, origin="upper", extent=[r["lon0"], r["lon1"],
                  r["lat0"], r["lat1"]],
                  transform=ccrs.PlateCarree(),
                  interpolation="bilinear", zorder=1, **kwargs)
    else:
        ax.imshow(data, origin="upper", aspect="auto",
                  interpolation="bilinear", **kwargs)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")


def render_frame(
    date_str: str,
    cams_pm25: np.ndarray,   # (H, W) global 0.25°, µg/m³
    aria_pm25: np.ndarray,   # (H, W) global 0.01°, µg/m³  (or same grid)
    openaq_lons: np.ndarray, # station longitudes
    openaq_lats: np.ndarray, # station latitudes
    openaq_vals: np.ndarray, # station PM2.5 µg/m³
    outpath: Path,
    region: str = "global",
    title_suffix: str = "",
):
    r = REGIONS[region]
    fig = plt.figure(figsize=(20, 8), dpi=300)
    fig.patch.set_facecolor("#1a1a2e")   # dark navy background

    norm = _norm()

    # ── Shared colorbar (bottom center) ──────────────────────────────────────
    cbar_ax = fig.add_axes([0.35, 0.04, 0.30, 0.018])
    cb = ColorbarBase(cbar_ax, cmap=CMAP, norm=norm, orientation="horizontal")
    cb.set_label("PM₂.₅  (µg/m³)", color="white", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=7)
    # WHO guide lines
    for lvl in WHO_LEVELS:
        if VMIN <= lvl <= VMAX:
            cb.ax.axvline(lvl, color="white", lw=0.6, alpha=0.7)

    # ── Left panel: CAMS ─────────────────────────────────────────────────────
    ax_cams = _make_map_ax(fig, [0.02, 0.12, 0.44, 0.82], region)
    _imshow_on_ax(ax_cams, cams_pm25, region, cmap=CMAP, norm=norm)
    _draw_cities(ax_cams, region)
    ax_cams.set_title("CAMS Operational  (0.25° ≈ 25 km)", color="white",
                      fontsize=11, pad=6, fontweight="bold")

    # ── Right panel: ARIA ─────────────────────────────────────────────────────
    ax_aria = _make_map_ax(fig, [0.54, 0.12, 0.44, 0.82], region)
    _imshow_on_ax(ax_aria, aria_pm25, region, cmap=CMAP, norm=norm)
    _draw_cities(ax_aria, region)

    # Overlay OpenAQ stations (right panel only — ground truth)
    if openaq_vals is not None and len(openaq_vals) > 0:
        # Filter to region
        m = ((openaq_lons >= r["lon0"]) & (openaq_lons <= r["lon1"]) &
             (openaq_lats >= r["lat0"]) & (openaq_lats <= r["lat1"]))
        if m.any():
            sc_kw = dict(s=12, edgecolors="white", linewidths=0.3,
                         cmap=CMAP, norm=norm, zorder=6)
            if HAS_CARTOPY:
                sc_kw["transform"] = ccrs.PlateCarree()
            ax_aria.scatter(openaq_lons[m], openaq_lats[m], c=openaq_vals[m],
                            **sc_kw)

    ax_aria.set_title("ARIA Foundation Model  (0.01° ≈ 1 km)", color="white",
                      fontsize=11, pad=6, fontweight="bold")

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             f"Global PM₂.₅ Air Quality · {date_str}{title_suffix}",
             ha="center", va="top", fontsize=13, color="white",
             fontweight="bold",
             fontfamily="DejaVu Sans")
    fig.text(0.5, 0.93,
             "● OpenAQ ground stations  |  WHO annual guideline: 5 µg/m³  |  "
             "WHO daily guideline: 15 µg/m³",
             ha="center", va="top", fontsize=7.5, color="#aaaacc")

    # ── Watermark ─────────────────────────────────────────────────────────────
    fig.text(0.99, 0.01, "ARIA · LUMI · 2025",
             ha="right", va="bottom", fontsize=6, color="#555577", alpha=0.7)

    fig.savefig(outpath, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main: loop over days in the npz and generate frames
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True,
                    help="Path to zeroshot_2025_comparison.npz")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for PNG frames")
    ap.add_argument("--region", default="global",
                    choices=list(REGIONS.keys()))
    ap.add_argument("--fps",    type=int, default=12,
                    help="Frames/sec hint printed for ffmpeg command")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="0 = all; N = first N days only (for quick test)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} …")
    data = np.load(args.input, allow_pickle=True)

    # Expected keys in the npz (from zero_shot_inference_2025.py):
    #   dates         — array of date strings "YYYY-MM-DD"
    #   aria_maps     — (N, H_aria, W_aria)  global 0.01°  µg/m³
    #   cams_maps     — (N, H_cams, W_cams)  global 0.25°  µg/m³
    #   station_lons  — (N, S)  per-day station lon (padded, use station_count)
    #   station_lats  — (N, S)
    #   station_pm25  — (N, S)
    #   station_count — (N,)    number of valid stations on that day

    dates        = data["dates"]
    aria_maps    = data["aria_maps"]          # (N, H, W)
    cams_maps    = data["cams_maps"]          # (N, Hc, Wc)
    st_lons      = data.get("station_lons",  None)
    st_lats      = data.get("station_lats",  None)
    st_pm25      = data.get("station_pm25",  None)
    st_count     = data.get("station_count", None)

    N = len(dates)
    if args.max_frames > 0:
        N = min(N, args.max_frames)

    print(f"Rendering {N} frames  →  {outdir}  (region={args.region})")

    for i in range(N):
        date_str = str(dates[i])
        outpath  = outdir / f"frame_{i:04d}_{date_str}.png"

        if outpath.exists():
            print(f"  [{i+1}/{N}] {date_str} — skip (exists)")
            continue

        # Per-day station arrays
        if st_count is not None:
            n_st = int(st_count[i])
            lons_i = st_lons[i, :n_st] if st_lons is not None else None
            lats_i = st_lats[i, :n_st] if st_lats is not None else None
            pm25_i = st_pm25[i, :n_st] if st_pm25 is not None else None
        else:
            lons_i = lats_i = pm25_i = None

        render_frame(
            date_str   = date_str,
            cams_pm25  = cams_maps[i],
            aria_pm25  = aria_maps[i],
            openaq_lons= lons_i,
            openaq_lats= lats_i,
            openaq_vals= pm25_i,
            outpath    = outpath,
            region     = args.region,
        )
        print(f"  [{i+1}/{N}] {date_str} ✓")

    # ── ffmpeg command ────────────────────────────────────────────────────────
    mp4 = outdir.parent / f"aria_vs_cams_{args.region}.mp4"
    print("\n✅ Done. Assemble with:\n")
    print(f"  ffmpeg -framerate {args.fps} -pattern_type glob \\")
    print(f"         -i '{outdir}/frame_*.png' \\")
    print(f"         -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' \\")
    print(f"         -vcodec libx264 -pix_fmt yuv420p -crf 18 \\")
    print(f"         {mp4}")
    print()
    print("  Or for a lossless WebM (smaller, web-friendly):")
    print(f"  ffmpeg -framerate {args.fps} -pattern_type glob \\")
    print(f"         -i '{outdir}/frame_*.png' \\")
    print(f"         -c:v libvpx-vp9 -crf 20 -b:v 0 {mp4.with_suffix('.webm')}")


if __name__ == "__main__":
    main()
