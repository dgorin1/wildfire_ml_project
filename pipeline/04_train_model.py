"""
Script 04: Train a U-Net to predict wildfire spread 12 hours into the future.

Input:  (17, 100, 100) multi-channel raster at time T
Output: (1,  100, 100) binary fire mask at time T+12h

Architecture: U-Net with 4 encoder stages, ~31M parameters.
Loss: Combo = 0.5 * BCEWithLogitsLoss(pos_weight) + 0.5 * DiceLoss
Split: Train on TRAIN_YEARS, validate on VAL_YEARS (year-based to prevent leakage).
"""

import os
import glob
import json
import math
import time
import csv
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

TRAINING_DATA_DIR = config["TRAINING_DATA_DIR"]
NORM_STATS_PATH = config["NORM_STATS_PATH"]
MODEL_DIR = config.get("MODEL_DIR", "models")
N_INPUT_CHANNELS = config.get("N_INPUT_CHANNELS", 17)
UNET_BASE_FILTERS = config.get("UNET_BASE_FILTERS", 64)
UNET_DEPTH = config.get("UNET_DEPTH", 4)
BATCH_SIZE = config.get("BATCH_SIZE", 32)
NUM_EPOCHS = config.get("NUM_EPOCHS", 100)
LR = config.get("LEARNING_RATE", 1e-3)
LR_SCHEDULER = config.get("LR_SCHEDULER", "cosine")
WEIGHT_DECAY = config.get("WEIGHT_DECAY", 1e-4)
PATIENCE = config.get("EARLY_STOPPING_PATIENCE", 10)
LOSS_FUNCTION = config.get("LOSS_FUNCTION", "combo")
TRAIN_YEARS = config["TRAIN_YEARS"]
VAL_YEARS = config["VAL_YEARS"]

# Wind channel indices that need sign-flip during augmentation
WIND_U_CHANNELS = [3, 5]  # wind_u_10m, wind_u_80m
WIND_V_CHANNELS = [4, 6]  # wind_v_10m, wind_v_80m

# Max npz files to keep decompressed in each worker's in-process cache.
# Each file ≈ 24 MB (38 samples × 16 ch × 100 × 100 × 4 B).
# 150 files × 24 MB × 2 workers ≈ 7 GB — manageable on 16 GB+ Macs.
MAX_CACHE_FILES = 150


# =============================================================================
# Model
# =============================================================================

class ConvBlock(nn.Module):
    """Two-layer conv block: (Conv -> BN -> ReLU) x2."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net with configurable depth and base filter count.
    Input spatial size is padded to the next power of 2 before the encoder
    and cropped back at the output, avoiding size mismatches through pooling stages.
    """
    def __init__(self, n_channels=17, n_classes=1, base_filters=64, depth=4):
        super().__init__()
        self.depth = depth

        f = base_filters
        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = n_channels
        for i in range(depth):
            out_ch = f * (2 ** i)
            self.encoders.append(ConvBlock(in_ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = f * (2 ** depth)
        self.bottleneck = ConvBlock(in_ch, bottleneck_ch)
        in_ch = bottleneck_ch

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            skip_ch = f * (2 ** i)
            up_out_ch = skip_ch  # upconv halves channels
            self.upconvs.append(nn.ConvTranspose2d(in_ch, up_out_ch, 2, stride=2))
            self.decoders.append(ConvBlock(up_out_ch + skip_ch, skip_ch))
            in_ch = skip_ch

        # Output head
        self.output_conv = nn.Conv2d(in_ch, n_classes, 1)

    def _pad_to_pow2(self, x):
        """Pad spatial dims to the next power of 2."""
        _, _, h, w = x.shape
        ph = 2 ** math.ceil(math.log2(h)) if h > 1 else 1
        pw = 2 ** math.ceil(math.log2(w)) if w > 1 else 1
        pad_h = ph - h
        pad_w = pw - w
        # Pad: (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, h, w

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        x, h, w = self._pad_to_pow2(x)

        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Align spatial size to skip in case of rounding
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        x = self.output_conv(x)
        # Crop back to original spatial size
        x = x[:, :, :orig_h, :orig_w]
        return x


# =============================================================================
# Losses
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class ComboLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + (1.0 - self.alpha) * self.dice(logits, targets)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss as used in Zou et al. (2023) for fire spread.

    Tversky index generalises F1: alpha weights FN (recall), beta weights FP.
    alpha=0.75, beta=0.25 heavily penalises missing spread pixels (FN),
    which is appropriate when spread pixels are the rare, high-value class.
    The focal exponent gamma sharpens focus on hard examples.
    """
    def __init__(self, alpha=0.75, beta=0.25, gamma=1.33, smooth=1.0):
        super().__init__()
        self.alpha = alpha   # FN weight — emphasises recall of spread pixels
        self.beta  = beta    # FP weight
        self.gamma = gamma   # focal sharpening (Zou et al. use 4/3)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        t = targets.view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1.0 - tversky) ** self.gamma


# =============================================================================
# Dataset
# =============================================================================

class WildfireDataset(Dataset):
    """
    Loads per-fire .npz sample files. Applies per-channel normalization and augmentation.

    Augmentation (physically consistent):
    - Random horizontal flip: negates wind_u channels (indices 3, 5)
    - Random vertical flip:   negates wind_v channels (indices 4, 6)
    - Random 90-degree rotation increments
    """
    def __init__(self, npz_paths, channel_stats, augment=False):
        self.augment = augment
        self.channel_stats = channel_stats  # dict: {channel_idx: {"mean": float, "std": float}}

        # Build flat index using cumulative offsets for O(log n) lookup
        self.files = []
        cum_sizes = [0]
        for path in npz_paths:
            data = np.load(path)
            n = data["X"].shape[0]
            self.files.append(path)
            cum_sizes.append(cum_sizes[-1] + n)
        self.cum_sizes = np.array(cum_sizes)
        self.total = cum_sizes[-1]

        # Bounded FIFO cache: stores decompressed (X, Y) arrays per file index.
        # Capped at MAX_CACHE_FILES entries; oldest entry evicted when full.
        self._cache = OrderedDict()

    def __len__(self):
        return self.total

    def _load_file(self, file_idx):
        if file_idx not in self._cache:
            data = np.load(self.files[file_idx], allow_pickle=False)
            self._cache[file_idx] = (data["X"], data["Y"])
            if len(self._cache) > MAX_CACHE_FILES:
                self._cache.popitem(last=False)  # evict oldest
        return self._cache[file_idx]

    def __getitem__(self, idx):
        # Binary search for file index — O(log n) instead of O(n)
        file_idx = int(np.searchsorted(self.cum_sizes, idx, side="right")) - 1
        local_idx = idx - self.cum_sizes[file_idx]

        X_all, Y_all = self._load_file(file_idx)
        X = X_all[local_idx].copy()  # (16, 100, 100) float32
        Y = Y_all[local_idx].copy()  # (1,  100, 100) float32

        # Normalize per channel
        for c in range(X.shape[0]):
            mean = self.channel_stats[str(c)]["mean"]
            std = self.channel_stats[str(c)]["std"]
            X[c] = (X[c] - mean) / (std + 1e-8)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        if self.augment:
            X, Y = self._augment(X, Y)

        return X, Y

    def _augment(self, X, Y):
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            X = torch.flip(X, dims=[2])
            Y = torch.flip(Y, dims=[2])
            for c in WIND_U_CHANNELS:
                X[c] = -X[c]

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            X = torch.flip(X, dims=[1])
            Y = torch.flip(Y, dims=[1])
            for c in WIND_V_CHANNELS:
                X[c] = -X[c]

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            X = torch.rot90(X, k, dims=[1, 2])
            Y = torch.rot90(Y, k, dims=[1, 2])

        return X, Y


# =============================================================================
# Utilities
# =============================================================================

def get_npz_paths(years):
    paths = []
    for year in years:
        year_dir = os.path.join(TRAINING_DATA_DIR, "by_fire", str(year))
        paths.extend(glob.glob(os.path.join(year_dir, "*.npz")))
    return sorted(paths)


def compute_spread_pos_weight(npz_paths, max_files=500):
    """
    Estimate BCE pos_weight for Y_spread = newly burned pixels only.
    Reads the unnormalized X (fire_mask channel 0 > 0.5 = existing fire) and Y.
    Y_spread = Y * (1 - existing_fire).
    Typical result: ~5000-10000x (spread pixels are ~10x rarer than fire pixels).
    """
    n_pos = 0.0
    n_neg = 0.0
    sampled = npz_paths[:max_files] if len(npz_paths) > max_files else npz_paths
    for path in sampled:
        data = np.load(path, allow_pickle=False)
        X = data["X"]  # (N, 16, 100, 100) float32 — unnormalized
        Y = data["Y"]  # (N, 1,  100, 100) float32
        existing_fire = (X[:, 0:1, :, :] > 0.5).astype(np.float32)
        Y_spread = Y * (1 - existing_fire)
        n_pos += float(Y_spread.sum())
        n_neg += float((1.0 - Y_spread).sum())
    if n_pos == 0:
        return torch.tensor([500.0])
    # Cap at 500: balances recall vs precision. 3000 caused extreme false positives
    # (91% recall but 1.7% precision). Lower cap forces the model to be more selective.
    raw = n_neg / n_pos
    return torch.tensor([min(raw, 500.0)])


def compute_sample_weights(npz_paths):
    """
    Per-sample weights for WeightedRandomSampler.
    Spreading fire samples (Y_spread > 0) get 10x weight to counteract their rarity.
    Fire-present-but-not-spreading samples get 2x. No-fire samples get 1x.
    """
    weights = []
    for path in npz_paths:
        data = np.load(path, allow_pickle=False)
        X = data["X"]
        Y = data["Y"]
        existing_fire = (X[:, 0:1, :, :] > 0.5).astype(np.float32)
        for i in range(Y.shape[0]):
            spread_pixels = float((Y[i] * (1 - existing_fire[i])).sum())
            if spread_pixels > 0:
                weights.append(10.0)   # fire is actively spreading
            elif Y[i].sum() > 0:
                weights.append(2.0)    # fire present but static
            else:
                weights.append(1.0)    # no fire
    return weights


def batch_metrics(logits, targets, existing_fire=None, threshold=0.5):
    """
    Vectorized per-sample IoU, F1, and (optionally) spread IoU for a batch.

    existing_fire: (B, 1, H, W) binary tensor of fire at time T, recovered from
                   normalized channel 0 (threshold > 1.0 since fire pixels ≈ 26).
                   If provided, also computes IoU over newly burned pixels only.

    Returns dicts of per-sample numpy arrays (nan where union==0 / no spread).
    """
    B = logits.shape[0]
    preds = (torch.sigmoid(logits) > threshold).float()  # (B, 1, H, W)

    p = preds.view(B, -1)   # (B, N)
    t = targets.view(B, -1) # (B, N)

    tp        = (p * t).sum(dim=1)
    fp        = (p * (1 - t)).sum(dim=1)
    fn        = ((1 - p) * t).sum(dim=1)
    union     = (p + t).clamp(0, 1).sum(dim=1)
    denom_f1  = 2 * tp + fp + fn

    nan_val = torch.full((B,), float("nan"), device=logits.device)
    iou = torch.where(union > 0,   tp / union,          nan_val).cpu().numpy()
    f1  = torch.where(denom_f1 > 0, 2 * tp / denom_f1, nan_val).cpu().numpy()

    result = {"iou": iou, "f1": f1}

    if existing_fire is not None:
        # Spread pixels: fire at T+12h that was NOT fire at T
        spread = targets * (1 - existing_fire)            # (B, 1, H, W)
        s = spread.view(B, -1)
        sp_inter = (p * s).sum(dim=1)
        sp_union = (p + s).clamp(0, 1).sum(dim=1)
        result["spread_iou"] = torch.where(
            sp_union > 0, sp_inter / sp_union, nan_val
        ).cpu().numpy()

    return result


# =============================================================================
# Training loop
# =============================================================================

def train_epoch(model, loader, optimizer, loss_fn, device):
    """
    Train one epoch. Loss is computed against Y_spread (newly burned pixels only),
    not the full fire mask. This prevents the model from exploiting the trivial
    fire-persistence solution (copying channel 0 to output).
    At inference, union model output with the input fire mask to get the full prediction.
    """
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        # Recover binary fire-at-T: normalized channel 0 → fire≈26.1, no-fire≈-0.04
        existing_fire = (X[:, 0:1] > 1.0).float()
        Y_spread = Y * (1 - existing_fire)  # only newly burned pixels
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_spread)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)


def val_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    iou_vals, f1_vals, spread_iou_vals = [], [], []
    n = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            existing_fire = (X[:, 0:1] > 1.0).float()
            Y_spread = Y * (1 - existing_fire)
            logits = model(X)
            loss = loss_fn(logits, Y_spread)
            total_loss += loss.item() * X.size(0)
            n += X.size(0)

            m = batch_metrics(logits, Y, existing_fire=existing_fire)
            iou_vals.extend(v for v in m["iou"] if not math.isnan(v))
            f1_vals.extend(v for v in m["f1"] if not math.isnan(v))
            spread_iou_vals.extend(v for v in m["spread_iou"] if not math.isnan(v))

    mean_iou        = float(np.mean(iou_vals))        if iou_vals        else 0.0
    mean_f1         = float(np.mean(f1_vals))         if f1_vals         else 0.0
    mean_spread_iou = float(np.mean(spread_iou_vals)) if spread_iou_vals else 0.0
    return total_loss / n, mean_iou, mean_f1, mean_spread_iou


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load normalization stats
    with open(NORM_STATS_PATH) as f:
        channel_stats = json.load(f)

    # Build datasets
    train_paths = get_npz_paths(TRAIN_YEARS)
    val_paths = get_npz_paths(VAL_YEARS)
    print(f"Train files: {len(train_paths)}, Val files: {len(val_paths)}")

    train_ds = WildfireDataset(train_paths, channel_stats, augment=True)
    val_ds = WildfireDataset(val_paths, channel_stats, augment=False)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Weighted sampler to oversample fire-spreading events
    sample_weights = compute_sample_weights(train_paths)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=False, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=False, persistent_workers=True)

    # Model
    model = UNet(n_channels=N_INPUT_CHANNELS, n_classes=1,
                 base_filters=UNET_BASE_FILTERS, depth=UNET_DEPTH).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss function
    # Training objective: predict Y_spread (newly burned pixels only).
    # This prevents the trivial fire-persistence solution (copy channel 0 → ~0.97 IoU).
    print("Training objective: Y_spread = fire at T+12h NOT already burning at T")
    if LOSS_FUNCTION == "focal_tversky":
        loss_fn = FocalTverskyLoss(alpha=0.75, beta=0.25, gamma=1.33)
        print("Loss: FocalTverskyLoss (alpha=0.75, beta=0.25, gamma=1.33)")
    elif LOSS_FUNCTION == "combo":
        print("Computing pos_weight for Y_spread (may take a minute)...")
        pos_weight = compute_spread_pos_weight(train_paths).to(device)
        print(f"Loss: ComboLoss  BCE spread pos_weight={pos_weight.item():.1f}")
        loss_fn = ComboLoss(pos_weight=pos_weight)
    elif LOSS_FUNCTION == "dice":
        loss_fn = DiceLoss()
        print("Loss: DiceLoss")
    else:
        print("Computing pos_weight for Y_spread (may take a minute)...")
        pos_weight = compute_spread_pos_weight(train_paths).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCEWithLogitsLoss  spread pos_weight={pos_weight.item():.1f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    # Training loop
    # Early stopping and checkpointing on spread_iou (higher = better).
    # val_loss is noisy with extreme pos_weight; spread_iou is the actual research metric.
    best_spread_iou = -1.0
    epochs_no_improve = 0
    log_path = os.path.join(MODEL_DIR, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "val_f1", "spread_iou", "lr"])

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou, val_f1, spread_iou = val_epoch(model, val_loader, loss_fn, device)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_iou={val_iou:.4f} | spread_iou={spread_iou:.4f} | val_f1={val_f1:.4f} | "
              f"lr={current_lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_iou, val_f1, spread_iou, current_lr])

        # Scheduler step
        if LR_SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Checkpoint best model by spread_iou
        if spread_iou > best_spread_iou:
            best_spread_iou = spread_iou
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "spread_iou": spread_iou,
                "val_f1": val_f1,
                "config": {
                    "n_channels": N_INPUT_CHANNELS,
                    "base_filters": UNET_BASE_FILTERS,
                    "depth": UNET_DEPTH,
                },
            }, os.path.join(MODEL_DIR, "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs (best spread_iou={best_spread_iou:.4f})")
            break

    print(f"\nTraining complete. Best spread_iou: {best_spread_iou:.4f}")
    print(f"Model saved to {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
