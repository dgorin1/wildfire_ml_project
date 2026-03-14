"""
visualize_samples.py — Inspect assembled training samples visually.

For each sampled fire, produces a 10-panel figure showing:
  fire mask (T) | temperature | humidity | wind speed+vectors |
  shortwave radiation | fuel load | moisture extinction |
  elevation | slope | fire mask (T+12h target)

Also produces a channel-distribution figure across all sampled fires.

Usage:
  python pipeline/visualize_samples.py               # 6 random fires
  python pipeline/visualize_samples.py --n-fires 12 --year 2020
  python pipeline/visualize_samples.py --show        # open interactively
"""

import os
import re
import glob
import argparse
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import yaml

matplotlib.use("Agg")  # non-interactive by default; overridden by --show

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAINING_DATA_DIR = config.get("TRAINING_DATA_DIR", "data/training_samples")
CHANNEL_NAMES = config.get("CHANNEL_NAMES", [
    "fire_mask", "temperature_2m", "relative_humidity_2m",
    "wind_u_10m", "wind_v_10m", "wind_u_80m", "wind_v_80m",
    "total_cloud_cover_atmosphere",
    "downward_short_wave_radiation_flux_surface",
    "fuel_load", "fuel_depth", "moisture_extinction",
    "elevation", "slope", "aspect_sin", "aspect_cos",
])

# ---------------------------------------------------------------------------
# Panel spec: (channel_index_or_special, title, colormap)
# "wind_speed" is a derived channel: sqrt(u10^2 + v10^2)
# ---------------------------------------------------------------------------
PANELS = [
    (0,            "Fire Mask (T)",       "Reds"),
    (1,            "Temperature 2m (°C)", "RdBu_r"),
    (2,            "Rel. Humidity (%)",   "YlGnBu"),
    ("wind_speed", "Wind Speed 10m (m/s)","viridis"),
    (9,            "Shortwave Rad (W/m²)","YlOrRd"),
    (10,           "Fuel Load (t/ac)",    "YlOrBr"),
    (12,           "Moisture Ext. (%)",   "BrBG"),
    (13,           "Elevation (m)",       "terrain"),
    (14,           "Slope (°)",           "YlOrBr"),
    ("target",     "Fire Mask (T+12h)",   "Reds"),
]

N_PANELS = len(PANELS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_panel_data(X, Y, spec):
    """Extract 2D array for a panel spec entry."""
    ch, _, _ = spec
    if ch == "wind_speed":
        return np.sqrt(X[3] ** 2 + X[4] ** 2)
    if ch == "target":
        return Y[0]
    return X[ch]


def find_npz_files(base_dir, year=None):
    if year:
        pattern = os.path.join(base_dir, "by_fire", str(year), "*.npz")
    else:
        pattern = os.path.join(base_dir, "by_fire", "**", "*.npz")
    return glob.glob(pattern, recursive=True)


def parse_fire_info(path):
    """Extract fireID and year from filename like fireID_12345_2019_samples.npz"""
    m = re.search(r"fireID_(\d+)_(\d{4})_samples", os.path.basename(path))
    if m:
        return m.group(1), m.group(2)
    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Per-fire figure
# ---------------------------------------------------------------------------

def plot_fire(path, out_dir, show=False):
    data = np.load(path, allow_pickle=False)
    X_all = data["X"]   # (N, 17, 100, 100)
    Y_all = data["Y"]   # (N,  1, 100, 100)

    if len(X_all) == 0:
        print(f"  Skipping {os.path.basename(path)}: no samples")
        return

    t = len(X_all) // 2   # use middle timestep
    X = X_all[t]           # (17, 100, 100)
    Y = Y_all[t]           # (1,  100, 100)

    fire_id, year = parse_fire_info(path)

    fig, axes = plt.subplots(1, N_PANELS, figsize=(N_PANELS * 3, 3.6))
    fig.suptitle(f"fireID {fire_id}  |  year {year}  |  timestep {t}/{len(X_all)-1}",
                 fontsize=11, y=1.01)

    for ax, spec in zip(axes, PANELS):
        ch, title, cmap = spec
        img = get_panel_data(X, Y, spec)

        # Fire mask panels: fix scale 0–1
        if ch in (0, "target"):
            vmin, vmax = 0, 1
        else:
            vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
            if vmin == vmax:
                vmax = vmin + 1e-6

        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                       interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Overlay T+12h target contour on the T fire mask panel
        if ch == 0 and Y[0].max() > 0:
            ax.contour(Y[0], levels=[0.5], colors=["lime"], linewidths=0.8)

        # Wind vectors on wind speed panel
        if ch == "wind_speed":
            step = 10
            ys = np.arange(0, 100, step)
            xs = np.arange(0, 100, step)
            u = X[3][np.ix_(ys, xs)]
            v = X[4][np.ix_(ys, xs)]
            ax.quiver(xs, ys, u, -v,   # -v because imshow y-axis is flipped
                      color="white", scale=150, width=0.004, headwidth=3)

    fig.tight_layout()

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out_path = os.path.join(out_dir, f"fireID_{fire_id}_{year}_t{t}.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Channel distribution figure
# ---------------------------------------------------------------------------

def plot_distributions(npz_paths, out_dir, show=False):
    """Histogram of raw values for all 17 channels across sampled fires."""
    all_X = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=False)
        X_all = data["X"]
        if len(X_all) > 0:
            all_X.append(X_all[len(X_all) // 2])   # middle timestep per fire

    if not all_X:
        print("  No data for distribution plot.")
        return

    X_stack = np.stack(all_X, axis=0)   # (n_fires, 17, 100, 100)

    n_ch = X_stack.shape[1]
    ncols = 6
    nrows = (n_ch + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes = axes.flatten()

    for i in range(n_ch):
        vals = X_stack[:, i, :, :].ravel()
        vals = vals[np.isfinite(vals)]
        axes[i].hist(vals, bins=50, color="steelblue", edgecolor="none")
        axes[i].set_title(CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"ch{i}",
                          fontsize=7)
        axes[i].tick_params(labelsize=6)

    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Channel value distributions (raw, middle timestep per fire)", fontsize=11)
    fig.tight_layout()

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out_path = os.path.join(out_dir, "channel_distributions.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize assembled training samples.")
    parser.add_argument("--n-fires", type=int, default=6,
                        help="Number of random fires to visualize (default: 6)")
    parser.add_argument("--year", type=int, default=None,
                        help="Restrict to a specific year (default: all years)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--out-dir", type=str, default="outputs/sample_viz",
                        help="Output directory for PNG files")
    parser.add_argument("--show", action="store_true",
                        help="Display figures interactively instead of saving")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_files = find_npz_files(TRAINING_DATA_DIR, year=args.year)
    if not all_files:
        print(f"No .npz files found in {TRAINING_DATA_DIR}. Has script 03 finished?")
        return

    random.seed(args.seed)
    sample_files = random.sample(all_files, min(args.n_fires, len(all_files)))

    print(f"Visualizing {len(sample_files)} fires from {len(all_files)} total...")
    for path in sample_files:
        plot_fire(path, args.out_dir, show=args.show)

    print("Building channel distribution plot...")
    plot_distributions(sample_files, args.out_dir, show=args.show)

    print(f"\nDone. Output in: {args.out_dir}/")


if __name__ == "__main__":
    main()
