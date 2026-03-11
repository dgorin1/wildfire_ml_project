"""
Script 05: Evaluate the trained U-Net on the test set.

Outputs:
  outputs/evaluation/test_metrics.json         — aggregate metrics
  outputs/evaluation/per_sample_results.csv    — per-sample IoU, F1, etc.
  outputs/evaluation/prediction_grids/         — visualizations
  outputs/evaluation/error_analysis/           — IoU vs. fire size, wind, humidity
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
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# Reuse model definition from script 04
from pipeline.train_model import UNet, WildfireDataset, iou_score, f1_score  # noqa: E402

# Fall back to direct import if running as a script from the project root
try:
    from pipeline.train_model import UNet, WildfireDataset, iou_score, f1_score
except ImportError:
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("train_model", "pipeline/04_train_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    UNet = mod.UNet
    WildfireDataset = mod.WildfireDataset
    iou_score = mod.iou_score
    f1_score = mod.f1_score

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAINING_DATA_DIR = config["TRAINING_DATA_DIR"]
NORM_STATS_PATH = config["NORM_STATS_PATH"]
MODEL_DIR = config.get("MODEL_DIR", "models")
EVAL_OUTPUT_DIR = config.get("EVAL_OUTPUT_DIR", "outputs/evaluation")
TEST_YEARS = config["TEST_YEARS"]
BATCH_SIZE = config.get("BATCH_SIZE", 32)
CHANNEL_NAMES = config["CHANNEL_NAMES"]

WIND_U_IDX = 3   # wind_u_10m (pre-normalization channel index)
WIND_V_IDX = 4   # wind_v_10m


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
    print(f"Loaded model from epoch {ckpt['epoch']} (val_iou={ckpt['val_iou']:.4f})")
    return model


def compute_metrics_batch(logits, targets, threshold=0.5):
    """Compute per-sample metrics. Returns list of dicts."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    results = []
    for i in range(logits.shape[0]):
        p = preds[i].squeeze()
        t = targets[i].squeeze()
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        union = tp + fp + fn
        iou = tp / union if union > 0 else 1.0
        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        fire_pixels_gt = int(t.sum().item())
        results.append({
            "iou": iou, "f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fire_pixels_gt": fire_pixels_gt,
            "prob_map": probs[i].squeeze().cpu().numpy(),
            "pred_map": p.cpu().numpy(),
            "target_map": t.cpu().numpy(),
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
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    all_results = []
    # We also need the raw X for visualization (wind vectors, fire mask)
    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Evaluating"):
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            batch_results = compute_metrics_batch(logits, Y)
            # Store un-normalized fire_mask and wind channels for visualization
            for i, res in enumerate(batch_results):
                res["fire_mask_input"] = X[i, 0].cpu().numpy()
                res["wind_u"] = X[i, WIND_U_IDX].cpu().numpy()
                res["wind_v"] = X[i, WIND_V_IDX].cpu().numpy()
            all_results.extend(batch_results)

    return all_results


def save_per_sample_csv(results, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_idx", "iou", "f1", "precision",
                                                "recall", "tp", "fp", "fn", "tn", "fire_pixels_gt"])
        writer.writeheader()
        for i, r in enumerate(results):
            writer.writerow({k: r[k] for k in writer.fieldnames if k != "sample_idx"} | {"sample_idx": i})


def save_aggregate_metrics(results, output_path):
    ious = [r["iou"] for r in results]
    f1s = [r["f1"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    # Exclude empty-target samples from IoU mean (fire extinguished: perfect prediction = 1.0 IoU)
    active_ious = [r["iou"] for r in results if r["fire_pixels_gt"] > 0]
    metrics = {
        "n_samples": len(results),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_iou_active_fires": float(np.mean(active_ious)) if active_ious else None,
        "mean_f1": float(np.mean(f1s)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
    }
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== Test Set Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def plot_prediction_grids(results, output_dir, n_samples=16):
    """
    Save a grid of visualizations: input fire mask | predicted probability | ground truth.
    Overlays wind vectors as a quiver plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Sample evenly across IoU range
    sorted_results = sorted(results, key=lambda r: r["iou"])
    step = max(1, len(sorted_results) // n_samples)
    selected = sorted_results[::step][:n_samples]

    n_cols = 4
    n_rows = math.ceil(len(selected) / n_cols)

    for batch_i in range(0, len(selected), 12):
        batch = selected[batch_i:batch_i + 12]
        fig, axes = plt.subplots(len(batch), 3, figsize=(12, 4 * len(batch)))
        if len(batch) == 1:
            axes = axes[np.newaxis, :]

        for row_i, res in enumerate(batch):
            ax0, ax1, ax2 = axes[row_i]

            ax0.imshow(res["fire_mask_input"], cmap="hot", vmin=0, vmax=1)
            ax0.set_title(f"Input fire mask (IoU={res['iou']:.3f})")
            ax0.axis("off")

            ax1.imshow(res["prob_map"], cmap="Reds", vmin=0, vmax=1)
            # Wind quiver (subsample to 10x10 grid)
            h, w = res["wind_u"].shape
            step_q = max(1, h // 10)
            ys = np.arange(0, h, step_q)
            xs = np.arange(0, w, step_q)
            YY, XX = np.meshgrid(ys, xs, indexing="ij")
            ax1.quiver(XX, YY,
                       res["wind_u"][::step_q, ::step_q],
                       -res["wind_v"][::step_q, ::step_q],  # flip v for image coords
                       color="white", scale=50, alpha=0.6)
            ax1.set_title("Predicted probability + wind")
            ax1.axis("off")

            ax2.imshow(res["target_map"], cmap="hot", vmin=0, vmax=1)
            ax2.set_title(f"Ground truth (fire px: {res['fire_pixels_gt']})")
            ax2.axis("off")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"predictions_batch{batch_i // 12:02d}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()


def error_analysis(results, output_dir):
    """
    Plot IoU vs. fire size, wind speed, and other factors.
    """
    os.makedirs(output_dir, exist_ok=True)

    ious = np.array([r["iou"] for r in results])
    fire_sizes = np.array([r["fire_pixels_gt"] for r in results])
    wind_speeds = np.array([
        np.sqrt(np.mean(r["wind_u"] ** 2) + np.mean(r["wind_v"] ** 2))
        for r in results
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # IoU vs fire size (bin by quintiles)
    bins = np.quantile(fire_sizes, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_labels = [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins)-1)]
    binned_ious = [ious[(fire_sizes >= bins[i]) & (fire_sizes < bins[i+1])] for i in range(len(bins)-1)]
    # Include the last edge
    binned_ious[-1] = ious[(fire_sizes >= bins[-2]) & (fire_sizes <= bins[-1])]
    axes[0].boxplot(binned_ious, labels=bin_labels)
    axes[0].set_xlabel("Fire size at T (pixels)")
    axes[0].set_ylabel("IoU")
    axes[0].set_title("IoU by fire size quintile")
    axes[0].tick_params(axis="x", rotation=30)

    # IoU vs wind speed (scatter)
    axes[1].scatter(wind_speeds, ious, alpha=0.3, s=5, c="steelblue")
    axes[1].set_xlabel("Mean wind speed (normalized units)")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU vs. wind speed")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=100, bbox_inches="tight")
    plt.close()


def calibration_curve(results, output_dir, n_bins=10):
    """Reliability diagram: predicted probability vs. actual fire frequency."""
    all_probs = np.concatenate([r["prob_map"].ravel() for r in results])
    all_targets = np.concatenate([r["target_map"].ravel() for r in results])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_fracs = []
    for i in range(n_bins):
        mask = (all_probs >= bin_edges[i]) & (all_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(all_probs[mask].mean())
            bin_fracs.append(all_targets[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_means, bin_fracs, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of fire pixels")
    ax.set_title("Calibration curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=100, bbox_inches="tight")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
