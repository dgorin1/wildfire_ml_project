"""
Script 03: Assemble training samples.

For each fire zarr, extract consecutive (T, T+12h) pairs and build:
  X: (17, 100, 100) float32  — 17-channel input tensor
  Y: (1,  100, 100) float32  — binary fire mask at T+12h

Channel layout:
  0       fire_mask at T
  1–9     9 weather vars at (init_time[T], lead_time=12h)
  10–12   physical_fuels [fuel_load, fuel_depth, moisture_extinction]
  13      elevation
  14      slope (degrees)
  15      sin(aspect_radians)
  16      cos(aspect_radians)

Outputs:
  data/training_samples/by_fire/{year}/fireID_{id}_{year}_samples.npz
  data/training_samples/channel_stats.json
"""

import os
import glob
import json
import warnings
import yaml
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.ndimage import distance_transform_edt

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_WEATHER_ZARR_DIR = config["RAW_WEATHER_ZARR_DIR"]
TRAINING_DATA_DIR = config["TRAINING_DATA_DIR"]
WEATHER_LEAD_TIME_HOURS = config.get("WEATHER_LEAD_TIME_HOURS", 12)
MIN_TIMESTEPS = config.get("MIN_TIMESTEPS_PER_FIRE", 2)
TRAIN_YEARS = config["TRAIN_YEARS"]
VAL_YEARS = config["VAL_YEARS"]
TEST_YEARS = config["TEST_YEARS"]
NORM_STATS_PATH = config["NORM_STATS_PATH"]
CHANNEL_NAMES = config["CHANNEL_NAMES"]

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_80m",
    "wind_v_80m",
    "total_cloud_cover_atmosphere",
    "downward_short_wave_radiation_flux_surface",
]

LEAD_TIME_DELTA = pd.Timedelta(hours=WEATHER_LEAD_TIME_HOURS)


class SkipFireException(Exception):
    pass


def load_zarr_safe(zarr_path):
    """Open and validate a zarr file. Raises SkipFireException if required data is missing."""
    ds = xr.open_zarr(zarr_path, decode_coords="all", consolidated=False)

    if "physical_fuels" not in ds:
        ds.close()
        raise SkipFireException("missing physical_fuels")
    if "terrain" not in ds:
        ds.close()
        raise SkipFireException("missing terrain")
    if len(ds.init_time) < MIN_TIMESTEPS:
        ds.close()
        raise SkipFireException(f"only {len(ds.init_time)} timestep(s), need >= {MIN_TIMESTEPS}")

    return ds


def _fill_nan(arr, name="", nan_log=None):
    """Replace NaN with nearest valid neighbor. Falls back to 0 if entirely NaN."""
    nan_mask = np.isnan(arr)
    nan_count = int(nan_mask.sum())
    if nan_count > 0:
        if nan_count == arr.size:
            arr = np.zeros_like(arr)
        else:
            _, indices = distance_transform_edt(nan_mask, return_indices=True)
            arr = arr.copy()
            arr[nan_mask] = arr[tuple(indices[:, nan_mask])]
        if nan_log is not None:
            nan_log.append((name, nan_count))
    return arr


def build_input(ds, t_idx, nan_log=None):
    """
    Build a (17, 100, 100) float32 input tensor for timestep index t_idx.
    nan_log: optional list; NaN fill events are appended as (channel_name, count) tuples.
    """
    channels = []

    # Channel 0: fire_mask at T
    fire_ch = ds["fire_mask"].isel(init_time=t_idx).values.astype(np.float32)
    channels.append(_fill_nan(fire_ch, "fire_mask", nan_log))

    # Channels 1–9: weather variables at (T, lead_time=12h)
    for var in WEATHER_VARS:
        wx = ds[var].isel(init_time=t_idx).sel(lead_time=LEAD_TIME_DELTA, method="nearest").values.astype(np.float32)
        channels.append(_fill_nan(wx, var, nan_log))

    # Channels 10–12: physical_fuels (static, variable dim ordered: fuel_load, fuel_depth, moisture_extinction)
    fuels = ds["physical_fuels"].values.astype(np.float32)  # (3, y, x)
    for i in range(3):
        channels.append(_fill_nan(fuels[i], f"fuel_{i}", nan_log))

    # Channels 13–16: terrain (elevation, slope, sin/cos of aspect)
    terrain = ds["terrain"].values.astype(np.float32)  # (3, y, x): elevation, slope, aspect
    elev = _fill_nan(terrain[0], "elevation", nan_log)
    slope_ch = _fill_nan(terrain[1], "slope", nan_log)
    aspect_rad = np.deg2rad(_fill_nan(terrain[2], "aspect", nan_log))
    channels.append(elev)
    channels.append(slope_ch)
    channels.append(np.sin(aspect_rad).astype(np.float32))
    channels.append(np.cos(aspect_rad).astype(np.float32))

    X = np.stack(channels, axis=0)  # (N_CHANNELS, y, x)
    assert X.shape[1:] == (100, 100), f"Unexpected input shape: {X.shape}"
    return X


def build_target(ds, t_idx):
    """Build a (1, 100, 100) float32 target tensor from fire_mask at timestep t_idx."""
    mask = ds["fire_mask"].isel(init_time=t_idx).values.astype(np.float32)
    return mask[np.newaxis, :, :]  # (1, 100, 100)


def is_boundary_fire(target):
    """Return True if any fire pixel touches the spatial edge of the 100x100 grid."""
    mask = target.squeeze()
    return bool(
        mask[0, :].any() or mask[-1, :].any() or
        mask[:, 0].any() or mask[:, -1].any()
    )


def extract_samples(ds, zarr_path):
    """
    Extract all (X, Y) pairs from a fire zarr.
    Returns list of dicts with keys: X, Y, boundary, zarr_path, t_idx.
    """
    n_times = len(ds.init_time)
    samples = []
    nan_log = []
    for i in range(n_times - 1):
        X = build_input(ds, i, nan_log=nan_log)
        Y = build_target(ds, i + 1)
        boundary = is_boundary_fire(Y)
        samples.append({
            "X": X,
            "Y": Y,
            "boundary": boundary,
            "zarr_path": zarr_path,
            "t_idx": i,
        })
    if nan_log:
        from collections import Counter as _Counter
        by_ch = _Counter(ch for ch, _ in nan_log)
        total = sum(n for _, n in nan_log)
        ch_summary = ", ".join(f"{ch}×{cnt}" for ch, cnt in by_ch.most_common(3))
        warnings.warn(
            f"{os.path.basename(zarr_path)}: filled {total} NaN values across "
            f"{len(nan_log)} channel-slices (top channels: {ch_summary})",
            stacklevel=2,
        )
    return samples


def process_year(year):
    """Process all zarr files for a given year. Returns list of sample dicts."""
    year_dir = os.path.join(RAW_WEATHER_ZARR_DIR, str(year))
    zarr_files = glob.glob(os.path.join(year_dir, "*.zarr"))

    all_samples = []
    counts = Counter()

    for zarr_path in tqdm(zarr_files, desc=f"Year {year}"):
        try:
            ds = load_zarr_safe(zarr_path)
            samples = extract_samples(ds, zarr_path)
            ds.close()
            all_samples.extend(samples)
            counts["SUCCESS"] += 1
        except SkipFireException as e:
            counts[f"SKIPPED ({e})"] += 1
        except Exception as e:
            counts["ERROR"] += 1
            tqdm.write(f"  ERROR {os.path.basename(zarr_path)}: {e}")

    print(f"  Year {year}: {sum(counts.values())} fires processed")
    for k, v in sorted(counts.items()):
        print(f"    {k}: {v}")
    return all_samples


def save_samples_by_fire(samples, output_base_dir, year):
    """
    Group samples by source zarr and save one .npz per fire.
    """
    by_fire = {}
    for s in samples:
        key = s["zarr_path"]
        by_fire.setdefault(key, []).append(s)

    out_dir = os.path.join(output_base_dir, "by_fire", str(year))
    os.makedirs(out_dir, exist_ok=True)

    for zarr_path, fire_samples in by_fire.items():
        # Extract fire ID from filename: fireID_{id}_weather_{year}.zarr
        basename = os.path.basename(zarr_path)
        fire_id = basename.replace("fireID_", "").split("_weather_")[0]

        X_arr = np.stack([s["X"] for s in fire_samples], axis=0)  # (N, 17, 100, 100)
        Y_arr = np.stack([s["Y"] for s in fire_samples], axis=0)  # (N, 1, 100, 100)
        boundary_arr = np.array([s["boundary"] for s in fire_samples])
        t_idx_arr = np.array([s["t_idx"] for s in fire_samples])

        out_path = os.path.join(out_dir, f"fireID_{fire_id}_{year}_samples.npz")
        np.savez_compressed(
            out_path,
            X=X_arr,
            Y=Y_arr,
            boundary=boundary_arr,
            t_idx=t_idx_arr,
        )


def compute_norm_stats(train_npz_paths):
    """
    Compute per-channel mean and std by streaming through saved npz files.
    Uses a single-pass online algorithm — never loads more than one file at a time.
    """
    n_channels = len(CHANNEL_NAMES)
    ch_sum    = np.zeros(n_channels, dtype=np.float64)
    ch_sum_sq = np.zeros(n_channels, dtype=np.float64)
    ch_count  = np.zeros(n_channels, dtype=np.int64)

    for path in tqdm(train_npz_paths, desc="  Computing stats"):
        data = np.load(path, allow_pickle=False)
        X = data["X"].astype(np.float64)   # (N, 17, 100, 100)
        pixels = X.shape[0] * X.shape[2] * X.shape[3]
        for c in range(n_channels):
            vals = X[:, c, :, :].ravel()
            ch_sum[c]    += vals.sum()
            ch_sum_sq[c] += (vals ** 2).sum()
            ch_count[c]  += pixels

    stats = {}
    for c in range(n_channels):
        mean = ch_sum[c] / ch_count[c]
        std  = np.sqrt(ch_sum_sq[c] / ch_count[c] - mean ** 2)
        stats[c] = {
            "name": CHANNEL_NAMES[c],
            "mean": float(mean),
            "std":  float(std),
        }
    return stats


def main():
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    all_years = sorted(set(TRAIN_YEARS + VAL_YEARS))
    year_samples = {}

    for year in all_years:
        print(f"\n--- Processing year {year} ---")
        year_samples[year] = process_year(year)

    # Save samples grouped by fire
    for year, samples in year_samples.items():
        print(f"\nSaving {len(samples)} samples for year {year}...")
        save_samples_by_fire(samples, TRAINING_DATA_DIR, year)

    # Compute and save normalization stats from training years only
    print("\nComputing normalization statistics from training years...")
    train_npz_paths = []
    for y in TRAIN_YEARS:
        pattern = os.path.join(TRAINING_DATA_DIR, "by_fire", str(y), "*.npz")
        train_npz_paths.extend(glob.glob(pattern))
    stats = compute_norm_stats(train_npz_paths)

    os.makedirs(os.path.dirname(NORM_STATS_PATH), exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization stats saved to {NORM_STATS_PATH}")

    # Summary
    print("\n=== Summary ===")
    total = 0
    for year in all_years:
        n = len(year_samples.get(year, []))
        split = "train" if year in TRAIN_YEARS else "val/test"
        print(f"  {year} ({split}): {n} samples")
        total += n
    print(f"  Total: {total} samples")


if __name__ == "__main__":
    main()
