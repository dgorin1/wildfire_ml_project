"""
Mini end-to-end pipeline smoke test.

Runs the full pipeline (terrain → assembly → training → evaluation) on a
handful of small pre-existing 2020 zarrs that already have weather data and
fuel channels. HRRR weather data is NOT re-downloaded.

Usage:
    python pipeline/mini_pipeline.py            # runs everything
    python pipeline/mini_pipeline.py --clean    # removes mini_test dir first

Output: data/mini_test/
Runtime: ~5–15 min (dominated by py3dep terrain downloads for 4 fires).
"""

import os
import sys
import json
import shutil
import importlib.util
import argparse
import tempfile
import numpy as np
import xarray as xr
import torch

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def _load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Config ────────────────────────────────────────────────────────────────────

MINI_DIR = "data/mini_test"
ZARR_SOURCE_DIR = "data/raw_weather_zarr/2020"

# 4 small 2020 zarrs that already have physical_fuels (~6 MB each, 10 timesteps)
SOURCE_ZARRS = [
    "fireID_65777_weather_2020.zarr",
    "fireID_49267_weather_2020.zarr",
    "fireID_56422_weather_2020.zarr",
    "fireID_51100_weather_2020.zarr",
]

# Training config (tiny model for speed)
MINI_TRAIN_CONFIG = {
    "N_INPUT_CHANNELS": 17,
    "UNET_BASE_FILTERS": 16,
    "UNET_DEPTH": 2,
    "BATCH_SIZE": 4,
    "NUM_EPOCHS": 5,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "EARLY_STOPPING_PATIENCE": 10,
    "LR_SCHEDULER": "cosine",
    "LOSS_FUNCTION": "combo",
}

MINI_ZARR_DIR = os.path.join(MINI_DIR, "zarrs", "2020")
MINI_SAMPLES_DIR = os.path.join(MINI_DIR, "training_samples")
MINI_MODEL_DIR = os.path.join(MINI_DIR, "models")
MINI_NORM_STATS_PATH = os.path.join(MINI_SAMPLES_DIR, "channel_stats.json")


# ── Step helpers ─────────────────────────────────────────────────────────────

def step0_copy_zarrs():
    print("\n=== Step 0: Copy zarrs to mini_test/ ===")
    os.makedirs(MINI_ZARR_DIR, exist_ok=True)
    copied = 0
    for name in SOURCE_ZARRS:
        src = os.path.join(ZARR_SOURCE_DIR, name)
        dst = os.path.join(MINI_ZARR_DIR, name)
        if not os.path.exists(src):
            print(f"  WARNING: source not found: {src}")
            continue
        if os.path.exists(dst):
            print(f"  Already exists, skipping: {name}")
        else:
            shutil.copytree(src, dst)
            print(f"  Copied: {name}")
        copied += 1
    print(f"  {copied}/{len(SOURCE_ZARRS)} zarrs ready in {MINI_ZARR_DIR}")
    return copied > 0


def step1_add_terrain():
    print("\n=== Step 1: Add terrain (py3dep) ===")
    script = _load_script("script_02", "pipeline/02_add_terrain_data.py")

    import glob
    zarr_files = glob.glob(os.path.join(MINI_ZARR_DIR, "*.zarr"))
    success, skipped, errors = 0, 0, 0
    for zarr_path in zarr_files:
        name = os.path.basename(zarr_path)
        status = script.process_zarr_terrain(zarr_path)
        print(f"  {name}: {status}")
        if status == "SUCCESS":
            success += 1
        elif status == "SKIPPED":
            skipped += 1
        else:
            errors += 1

    print(f"  Results: {success} success, {skipped} skipped, {errors} errors")
    if errors == len(zarr_files):
        raise RuntimeError("All terrain fetches failed — check network/py3dep.")
    return success + skipped  # number ready


def step2_assemble_samples():
    print("\n=== Step 2: Assemble training samples ===")
    script = _load_script("script_03", "pipeline/03_assemble_training_samples.py")
    os.makedirs(MINI_SAMPLES_DIR, exist_ok=True)

    import glob
    zarr_files = sorted(glob.glob(os.path.join(MINI_ZARR_DIR, "*.zarr")))
    all_samples = []
    for zarr_path in zarr_files:
        name = os.path.basename(zarr_path)
        try:
            ds = script.load_zarr_safe(zarr_path)
            samples = script.extract_samples(ds, zarr_path)
            ds.close()
            all_samples.extend(samples)
            print(f"  {name}: {len(samples)} samples")
        except script.SkipFireException as e:
            print(f"  {name}: SKIPPED ({e})")
        except Exception as e:
            print(f"  {name}: ERROR ({e})")

    if not all_samples:
        raise RuntimeError("No samples assembled — check zarrs have terrain.")

    # 3 fires for train, 1 for val (by source zarr path)
    unique_fires = list({s["zarr_path"] for s in all_samples})
    train_fires = set(unique_fires[:3])
    val_fires = set(unique_fires[3:]) or {unique_fires[-1]}

    train_samples = [s for s in all_samples if s["zarr_path"] in train_fires]
    val_samples = [s for s in all_samples if s["zarr_path"] in val_fires]

    print(f"\n  Total: {len(all_samples)} samples  |  train: {len(train_samples)}  val: {len(val_samples)}")

    # Compute norm stats from training samples only
    stats = script.compute_norm_stats([train_samples])
    os.makedirs(os.path.dirname(MINI_NORM_STATS_PATH), exist_ok=True)
    with open(MINI_NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Save per-fire npz files
    train_dir = os.path.join(MINI_SAMPLES_DIR, "train")
    val_dir = os.path.join(MINI_SAMPLES_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    def _save(samples, out_dir):
        by_fire = {}
        for s in samples:
            by_fire.setdefault(s["zarr_path"], []).append(s)
        paths = []
        for zarr_path, fire_samples in by_fire.items():
            name = os.path.basename(zarr_path).replace(".zarr", "_samples.npz")
            X = np.stack([s["X"] for s in fire_samples])
            Y = np.stack([s["Y"] for s in fire_samples])
            boundary = np.array([s["boundary"] for s in fire_samples])
            t_idx = np.array([s["t_idx"] for s in fire_samples])
            p = os.path.join(out_dir, name)
            np.savez_compressed(p, X=X, Y=Y, boundary=boundary, t_idx=t_idx)
            paths.append(p)
        return paths

    train_paths = _save(train_samples, train_dir)
    val_paths = _save(val_samples, val_dir)

    return train_paths, val_paths


def step3_train(train_paths, val_paths):
    print("\n=== Step 3: Train tiny U-Net ===")
    script = _load_script("script_04", "pipeline/04_train_model.py")
    os.makedirs(MINI_MODEL_DIR, exist_ok=True)

    with open(MINI_NORM_STATS_PATH) as f:
        channel_stats = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    train_ds = script.WildfireDataset(train_paths, channel_stats, augment=True)
    val_ds = script.WildfireDataset(val_paths, channel_stats, augment=False)
    print(f"  Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=MINI_TRAIN_CONFIG["BATCH_SIZE"], shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=MINI_TRAIN_CONFIG["BATCH_SIZE"], shuffle=False, num_workers=0
    )

    model = script.UNet(
        n_channels=MINI_TRAIN_CONFIG["N_INPUT_CHANNELS"],
        n_classes=1,
        base_filters=MINI_TRAIN_CONFIG["UNET_BASE_FILTERS"],
        depth=MINI_TRAIN_CONFIG["UNET_DEPTH"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    pos_weight = script.compute_pos_weight(train_paths).to(device)
    print(f"  BCE pos_weight: {pos_weight.item():.1f}")

    loss_fn = script.ComboLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MINI_TRAIN_CONFIG["LEARNING_RATE"],
        weight_decay=MINI_TRAIN_CONFIG["WEIGHT_DECAY"],
    )

    best_val_iou = -1.0
    for epoch in range(1, MINI_TRAIN_CONFIG["NUM_EPOCHS"] + 1):
        train_loss = script.train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou, val_f1 = script.val_epoch(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch}/{MINI_TRAIN_CONFIG['NUM_EPOCHS']}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_iou={val_iou:.4f}  val_f1={val_f1:.4f}")
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_iou": val_iou, "config": MINI_TRAIN_CONFIG},
                       os.path.join(MINI_MODEL_DIR, "mini_best_model.pt"))

    print(f"\n  Best val IoU: {best_val_iou:.4f}")
    return best_val_iou


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mini end-to-end pipeline smoke test")
    parser.add_argument("--clean", action="store_true",
                        help="Remove data/mini_test/ before running")
    parser.add_argument("--skip-terrain", action="store_true",
                        help="Skip terrain download (use if already done)")
    args = parser.parse_args()

    if args.clean and os.path.exists(MINI_DIR):
        shutil.rmtree(MINI_DIR)
        print(f"Removed {MINI_DIR}")

    print(f"Mini pipeline output: {os.path.abspath(MINI_DIR)}")

    ok = step0_copy_zarrs()
    if not ok:
        print("No source zarrs found. Aborting.")
        sys.exit(1)

    if not args.skip_terrain:
        step1_add_terrain()
    else:
        print("\n=== Step 1: Terrain (skipped) ===")

    train_paths, val_paths = step2_assemble_samples()
    best_iou = step3_train(train_paths, val_paths)

    print("\n" + "=" * 50)
    print("MINI PIPELINE COMPLETE")
    print(f"  Best val IoU: {best_iou:.4f}")
    print(f"  Model:        {MINI_MODEL_DIR}/mini_best_model.pt")
    print(f"  Samples:      {MINI_SAMPLES_DIR}/")
    if best_iou > 0:
        print("  STATUS: PASS — model learned something (val IoU > 0)")
    else:
        print("  STATUS: WARNING — val IoU = 0, model predicts no fire")
    print("=" * 50)


if __name__ == "__main__":
    main()
