"""
Script 05: Evaluate the trained U-Net on the test set.

Outputs:
  outputs/evaluation/test_metrics.json         — aggregate metrics
  outputs/evaluation/per_sample_results.csv    — per-sample IoU, F1, spread_iou, etc.
  outputs/evaluation/prediction_grids/         — visualizations
  outputs/evaluation/error_analysis/           — spread_iou vs. fire size, wind
"""

import os
import glob
import json
import csv
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reuse model definition from script 04
import importlib.util
spec = importlib.util.spec_from_file_location("train_model", "pipeline/04_train_model.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UNet = mod.UNet
WildfireDataset = mod.WildfireDataset

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAINING_DATA_DIR = config["TRAINING_DATA_DIR"]
NORM_STATS_PATH = config["NORM_STATS_PATH"]
MODEL_DIR = config.get("MODEL_DIR", "models")
EVAL_OUTPUT_DIR = config.get("EVAL_OUTPUT_DIR", "outputs/evaluation")
TEST_YEARS = config["TEST_YEARS"]
BATCH_SIZE = config.get("BATCH_SIZE", 32)

WIND_U_IDX = 3   # wind_u_10m (normalized channel index)
WIND_V_IDX = 4   # wind_v_10m

# Threshold to recover binary fire-at-T from normalized channel 0.
# After normalization: fire pixels ≈ 26.1, non-fire ≈ -0.04.
EXISTING_FIRE_THRESHOLD = 1.0


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = UNet(
        n_channels=cfg["n_channels"],
        n_classes=1,
        base_filters=cfg["base_filters"],
        depth=cfg["depth"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']} (spread_iou={ckpt['spread_iou']:.4f})")
    return model


def compute_metrics_batch(logits, targets, spread_targets, threshold=0.5):
    """
    Compute per-sample metrics against both the full fire mask (targets) and
    the spread-only mask (spread_targets = newly burned pixels).
    Returns a list of dicts, one per sample.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    results = []
    for i in range(logits.shape[0]):
        p = preds[i].squeeze()
        t = targets[i].squeeze()
        s = spread_targets[i].squeeze()

        # --- full-mask metrics (vs Y) ---
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        union = tp + fp + fn
        iou = tp / union if union > 0 else float("nan")
        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp / denom_f1) if denom_f1 > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        # --- spread-only metrics (vs Y_spread) ---
        stp = (p * s).sum().item()
        sfp = (p * (1 - s)).sum().item()
        sfn = ((1 - p) * s).sum().item()
        s_union = stp + sfp + sfn
        spread_iou = stp / s_union if s_union > 0 else float("nan")
        s_denom_f1 = 2 * stp + sfp + sfn
        spread_f1 = (2 * stp / s_denom_f1) if s_denom_f1 > 0 else float("nan")
        spread_precision = stp / (stp + sfp) if (stp + sfp) > 0 else float("nan")
        spread_recall = stp / (stp + sfn) if (stp + sfn) > 0 else float("nan")

        results.append({
            # full-mask
            "iou": iou, "f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fire_pixels_gt": int(t.sum().item()),
            # spread-only
            "spread_iou": spread_iou,
            "spread_f1": spread_f1,
            "spread_precision": spread_precision,
            "spread_recall": spread_recall,
            "spread_pixels_gt": int(s.sum().item()),
            # arrays for visualization
            "prob_map": probs[i].squeeze().cpu().numpy(),
            "pred_map": p.cpu().numpy(),
            "target_map": t.cpu().numpy(),
            "spread_map": s.cpu().numpy(),
        })
    return results


def get_test_npz_paths():
    paths = []
    for year in TEST_YEARS:
        year_dir = os.path.join(TRAINING_DATA_DIR, "by_fire", str(year))
        paths.extend(glob.glob(os.path.join(year_dir, "*.npz")))
    return sorted(paths)


def run_evaluation(model, test_paths, channel_stats, device):
    """Run inference on the test set. Returns a list of per-sample result dicts."""
    from torch.utils.data import DataLoader
    ds = WildfireDataset(test_paths, channel_stats, augment=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_results = []
    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Evaluating"):
            X, Y = X.to(device), Y.to(device)
            # Recover binary fire-at-T from normalized channel 0
            existing_fire = (X[:, 0:1] > EXISTING_FIRE_THRESHOLD).float()
            Y_spread = Y * (1 - existing_fire)

            logits = model(X)
            batch_results = compute_metrics_batch(logits, Y, Y_spread)

            # Store channels needed for visualization
            for i, res in enumerate(batch_results):
                # Binary fire-at-T for visualization
                res["fire_mask_input"] = existing_fire[i, 0].cpu().numpy()
                res["wind_u"] = X[i, WIND_U_IDX].cpu().numpy()
                res["wind_v"] = X[i, WIND_V_IDX].cpu().numpy()
            all_results.extend(batch_results)

    return all_results


def _nanmean(vals):
    arr = np.array(vals, dtype=float)
    valid = arr[~np.isnan(arr)]
    return float(np.mean(valid)) if len(valid) > 0 else None


def save_per_sample_csv(results, output_path):
    fields = [
        "sample_idx",
        "iou", "f1", "precision", "recall",
        "tp", "fp", "fn", "tn", "fire_pixels_gt",
        "spread_iou", "spread_f1", "spread_precision", "spread_recall",
        "spread_pixels_gt",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({k: r.get(k, "") for k in fields if k != "sample_idx"} | {"sample_idx": i})


def save_aggregate_metrics(results, output_path):
    active = [r for r in results if r["fire_pixels_gt"] > 0]
    spreading = [r for r in results if r["spread_pixels_gt"] > 0]

    metrics = {
        "n_samples": len(results),
        "n_active_fire_samples": len(active),
        "n_spreading_fire_samples": len(spreading),
        # Full-mask metrics (vs Y — includes persistence)
        "mean_iou_vs_full_mask": _nanmean([r["iou"] for r in results]),
        "mean_recall_vs_full_mask": _nanmean([r["recall"] for r in results]),
        "mean_precision_vs_full_mask": _nanmean([r["precision"] for r in results]),
        # Spread-only metrics (vs Y_spread — the honest research metric)
        "mean_spread_iou": _nanmean([r["spread_iou"] for r in results]),
        "mean_spread_iou_spreading_only": _nanmean([r["spread_iou"] for r in spreading]),
        "mean_spread_recall": _nanmean([r["spread_recall"] for r in results]),
        "mean_spread_precision": _nanmean([r["spread_precision"] for r in results]),
        "mean_spread_f1": _nanmean([r["spread_f1"] for r in results]),
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def plot_prediction_grids(results, output_dir, n_samples=16):
    """
    4-panel visualization per sample:
      Col 0: Input fire mask at T
      Col 1: Predicted spread probability + wind vectors
      Col 2: Ground truth spread (Y_spread — what the model is trained to predict)
      Col 3: Full ground truth fire mask at T+12h (for reference)
    Samples selected evenly across spread_iou range (skips NaN samples).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only visualize samples where spread ground truth exists
    with_spread = [r for r in results if r["spread_pixels_gt"] > 0]
    if not with_spread:
        print("  No samples with spread pixels — skipping prediction grids.")
        return

    sorted_results = sorted(with_spread, key=lambda r: r["spread_iou"] if not math.isnan(r["spread_iou"]) else -1)
    step = max(1, len(sorted_results) // n_samples)
    selected = sorted_results[::step][:n_samples]

    for batch_i in range(0, len(selected), 8):
        batch = selected[batch_i:batch_i + 8]
        fig, axes = plt.subplots(len(batch), 4, figsize=(16, 4 * len(batch)))
        if len(batch) == 1:
            axes = axes[np.newaxis, :]

        for row_i, res in enumerate(batch):
            ax0, ax1, ax2, ax3 = axes[row_i]
            siou = res["spread_iou"]
            siou_str = f"{siou:.3f}" if not math.isnan(siou) else "n/a"

            # Input fire mask
            ax0.imshow(res["fire_mask_input"], cmap="hot", vmin=0, vmax=1)
            ax0.set_title("Input fire (T)")
            ax0.axis("off")

            # Predicted spread probability + wind
            ax1.imshow(res["prob_map"], cmap="Reds", vmin=0, vmax=1)
            h, w = res["wind_u"].shape
            step_q = max(1, h // 10)
            ys = np.arange(0, h, step_q)
            xs = np.arange(0, w, step_q)
            YY, XX = np.meshgrid(ys, xs, indexing="ij")
            ax1.quiver(XX, YY,
                       res["wind_u"][::step_q, ::step_q],
                       -res["wind_v"][::step_q, ::step_q],
                       color="white", scale=50, alpha=0.6)
            ax1.set_title(f"Pred spread prob (spread_iou={siou_str})")
            ax1.axis("off")

            # Ground truth spread only
            ax2.imshow(res["spread_map"], cmap="Reds", vmin=0, vmax=1)
            ax2.set_title(f"GT spread only ({res['spread_pixels_gt']} px)")
            ax2.axis("off")

            # Full ground truth at T+12h
            ax3.imshow(res["target_map"], cmap="hot", vmin=0, vmax=1)
            ax3.set_title(f"Full GT T+12h ({res['fire_pixels_gt']} px)")
            ax3.axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"predictions_batch{batch_i // 8:02d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()


def error_analysis(results, output_dir):
    """Plot spread_iou vs. fire size and wind speed."""
    os.makedirs(output_dir, exist_ok=True)

    spreading = [r for r in results if r["spread_pixels_gt"] > 0 and not math.isnan(r["spread_iou"])]
    if not spreading:
        print("  No spreading samples for error analysis.")
        return

    spread_ious = np.array([r["spread_iou"] for r in spreading])
    fire_sizes = np.array([r["fire_pixels_gt"] for r in spreading])
    spread_sizes = np.array([r["spread_pixels_gt"] for r in spreading])
    wind_speeds = np.array([
        np.sqrt(np.mean(r["wind_u"] ** 2) + np.mean(r["wind_v"] ** 2))
        for r in spreading
    ])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # spread_iou vs existing fire size (binned)
    bins = np.quantile(fire_sizes, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_labels = [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins) - 1)]
    binned = [spread_ious[(fire_sizes >= bins[i]) & (fire_sizes < bins[i + 1])] for i in range(len(bins) - 1)]
    binned[-1] = spread_ious[(fire_sizes >= bins[-2]) & (fire_sizes <= bins[-1])]
    axes[0].boxplot(binned, tick_labels=bin_labels)
    axes[0].set_xlabel("Existing fire size at T (pixels)")
    axes[0].set_ylabel("spread_iou")
    axes[0].set_title("spread_iou by fire size")
    axes[0].tick_params(axis="x", rotation=30)

    # spread_iou vs spread size (binned)
    sbins = np.quantile(spread_sizes, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sbin_labels = [f"{int(sbins[i])}–{int(sbins[i+1])}" for i in range(len(sbins) - 1)]
    sbinned = [spread_ious[(spread_sizes >= sbins[i]) & (spread_sizes < sbins[i + 1])] for i in range(len(sbins) - 1)]
    sbinned[-1] = spread_ious[(spread_sizes >= sbins[-2]) & (spread_sizes <= sbins[-1])]
    axes[1].boxplot(sbinned, tick_labels=sbin_labels)
    axes[1].set_xlabel("Spread pixels at T+12h (Y_spread size)")
    axes[1].set_ylabel("spread_iou")
    axes[1].set_title("spread_iou by spread size")
    axes[1].tick_params(axis="x", rotation=30)

    # spread_iou vs wind speed (scatter)
    axes[2].scatter(wind_speeds, spread_ious, alpha=0.3, s=5, c="steelblue")
    axes[2].set_xlabel("Mean wind speed (normalized units)")
    axes[2].set_ylabel("spread_iou")
    axes[2].set_title("spread_iou vs. wind speed")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=100, bbox_inches="tight")
    plt.close()


def calibration_curve(results, output_dir, n_bins=10):
    """Reliability diagram: predicted probability vs. actual spread frequency."""
    os.makedirs(output_dir, exist_ok=True)
    all_probs = np.concatenate([r["prob_map"].ravel() for r in results])
    all_spread = np.concatenate([r["spread_map"].ravel() for r in results])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means, bin_fracs = [], []
    for i in range(n_bins):
        mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(all_probs[mask].mean())
            bin_fracs.append(all_spread[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of actual spread pixels")
    ax.set_title("Calibration curve (vs Y_spread)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=100, bbox_inches="tight")
    plt.close()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pt")
    model = load_model(checkpoint_path, device)

    with open(NORM_STATS_PATH) as f:
        channel_stats = json.load(f)

    test_paths = get_test_npz_paths()
    print(f"Test files: {len(test_paths)}")

    results = run_evaluation(model, test_paths, channel_stats, device)
    print(f"Total test samples: {len(results)}")

    save_aggregate_metrics(results, os.path.join(EVAL_OUTPUT_DIR, "test_metrics.json"))
    save_per_sample_csv(results, os.path.join(EVAL_OUTPUT_DIR, "per_sample_results.csv"))

    print("Generating prediction visualizations...")
    plot_prediction_grids(results, os.path.join(EVAL_OUTPUT_DIR, "prediction_grids"))

    print("Running error analysis...")
    error_analysis(results, os.path.join(EVAL_OUTPUT_DIR, "error_analysis"))

    print("Generating calibration curve...")
    calibration_curve(results, EVAL_OUTPUT_DIR)

    print(f"\nAll outputs saved to {EVAL_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
