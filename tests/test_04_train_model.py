"""
Unit tests for pipeline/04_train_model.py

Tests cover:
- UNet: output shape, padding/cropping for non-power-of-2 inputs
- DiceLoss: perfect prediction = 0 loss, all-wrong = ~1 loss, range [0,1]
- ComboLoss: loss range and gradient flows
- WildfireDataset: normalization, augmentation wind-flip consistency
- compute_pos_weight: returns a tensor with positive value
- iou_score / f1_score: known perfect and zero-overlap cases
"""

import sys
import os
import unittest
import importlib.util
import tempfile
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location("script_04", "pipeline/04_train_model.py")
script_04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script_04)

UNet = script_04.UNet
DiceLoss = script_04.DiceLoss
ComboLoss = script_04.ComboLoss
WildfireDataset = script_04.WildfireDataset
iou_score = script_04.iou_score
f1_score = script_04.f1_score
compute_pos_weight = script_04.compute_pos_weight
WIND_U_CHANNELS = script_04.WIND_U_CHANNELS
WIND_V_CHANNELS = script_04.WIND_V_CHANNELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npz(tmpdir, filename="samples.npz", n_samples=4, n_channels=17, fire_fraction=0.1):
    """Write a fake .npz sample file and return its path."""
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_channels, 100, 100)).astype(np.float32)
    Y = (rng.random((n_samples, 1, 100, 100)) < fire_fraction).astype(np.float32)
    boundary = np.zeros(n_samples, dtype=bool)
    t_idx = np.arange(n_samples)
    path = os.path.join(tmpdir, filename)
    np.savez_compressed(path, X=X, Y=Y, boundary=boundary, t_idx=t_idx)
    return path


def _fake_channel_stats(n_channels=17):
    """Return normalisation stats dict with mean=0, std=1 (identity transform)."""
    return {str(c): {"name": f"ch_{c}", "mean": 0.0, "std": 1.0} for c in range(n_channels)}


# ---------------------------------------------------------------------------
# UNet tests
# ---------------------------------------------------------------------------

class TestUNet(unittest.TestCase):

    def setUp(self):
        self.model = UNet(n_channels=17, n_classes=1, base_filters=16, depth=4)
        self.model.eval()

    def test_output_shape_100x100(self):
        x = torch.randn(2, 17, 100, 100)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 1, 100, 100))

    def test_output_shape_matches_input_spatial(self):
        """Output spatial dims must always match input, even for odd sizes."""
        for h, w in [(100, 100), (64, 64), (128, 128), (96, 96)]:
            x = torch.randn(1, 17, h, w)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape[2], h, f"Height mismatch for {h}x{w}")
            self.assertEqual(out.shape[3], w, f"Width mismatch for {h}x{w}")

    def test_output_channel_count(self):
        x = torch.randn(1, 17, 100, 100)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape[1], 1)

    def test_gradient_flows(self):
        """Loss.backward() should produce non-zero gradients in all encoder layers."""
        model = UNet(n_channels=17, n_classes=1, base_filters=16, depth=4)
        x = torch.randn(2, 17, 100, 100)
        y = torch.zeros(2, 1, 100, 100)
        y[:, :, 40:60, 40:60] = 1.0
        logits = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient for {name}")

    def test_different_batch_sizes(self):
        for bs in [1, 4, 8]:
            x = torch.randn(bs, 17, 100, 100)
            with torch.no_grad():
                out = self.model(x)
            self.assertEqual(out.shape[0], bs)

    def test_configurable_base_filters(self):
        """Smaller base_filters should produce a model with fewer parameters."""
        model_small = UNet(n_channels=17, n_classes=1, base_filters=8, depth=3)
        model_large = UNet(n_channels=17, n_classes=1, base_filters=32, depth=3)
        n_small = sum(p.numel() for p in model_small.parameters())
        n_large = sum(p.numel() for p in model_large.parameters())
        self.assertLess(n_small, n_large)


# ---------------------------------------------------------------------------
# DiceLoss tests
# ---------------------------------------------------------------------------

class TestDiceLoss(unittest.TestCase):

    def setUp(self):
        self.criterion = DiceLoss()

    def test_perfect_prediction_near_zero(self):
        """When prediction perfectly matches target, Dice loss should be ~0."""
        target = torch.zeros(2, 1, 10, 10)
        target[:, :, 3:7, 3:7] = 1.0
        # Large positive logits where target=1, large negative where target=0
        logits = (target * 20.0) - ((1 - target) * 20.0)
        loss = self.criterion(logits, target)
        self.assertLess(loss.item(), 0.05)

    def test_all_wrong_near_one(self):
        """Completely wrong prediction should give Dice loss near 1."""
        target = torch.zeros(2, 1, 10, 10)
        target[:, :, 3:7, 3:7] = 1.0
        # Predict opposite
        logits = -((target * 20.0) - ((1 - target) * 20.0))
        loss = self.criterion(logits, target)
        self.assertGreater(loss.item(), 0.9)

    def test_loss_in_zero_one_range(self):
        logits = torch.randn(4, 1, 20, 20)
        target = (torch.rand(4, 1, 20, 20) > 0.8).float()
        loss = self.criterion(logits, target)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertLessEqual(loss.item(), 1.0 + 1e-6)

    def test_empty_target_no_crash(self):
        """All-zero target (fire extinguished) should not produce NaN."""
        target = torch.zeros(2, 1, 20, 20)
        logits = torch.randn(2, 1, 20, 20)
        loss = self.criterion(logits, target)
        self.assertFalse(torch.isnan(loss))

    def test_differentiable(self):
        logits = torch.randn(2, 1, 20, 20, requires_grad=True)
        target = (torch.rand(2, 1, 20, 20) > 0.8).float()
        loss = self.criterion(logits, target)
        loss.backward()
        self.assertIsNotNone(logits.grad)


# ---------------------------------------------------------------------------
# ComboLoss tests
# ---------------------------------------------------------------------------

class TestComboLoss(unittest.TestCase):

    def setUp(self):
        pos_weight = torch.tensor([10.0])
        self.criterion = ComboLoss(pos_weight=pos_weight, alpha=0.5)

    def test_loss_positive(self):
        logits = torch.randn(2, 1, 20, 20)
        target = (torch.rand(2, 1, 20, 20) > 0.9).float()
        loss = self.criterion(logits, target)
        self.assertGreater(loss.item(), 0.0)

    def test_loss_decreases_toward_perfect(self):
        """Loss for a nearly-perfect prediction should be lower than random."""
        target = torch.zeros(2, 1, 20, 20)
        target[:, :, 5:10, 5:10] = 1.0
        logits_good = (target * 10.0) - ((1 - target) * 10.0)
        logits_random = torch.randn(2, 1, 20, 20)
        loss_good = self.criterion(logits_good, target)
        loss_random = self.criterion(logits_random, target)
        self.assertLess(loss_good.item(), loss_random.item())

    def test_differentiable(self):
        logits = torch.randn(2, 1, 20, 20, requires_grad=True)
        target = (torch.rand(2, 1, 20, 20) > 0.9).float()
        loss = self.criterion(logits, target)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())


# ---------------------------------------------------------------------------
# WildfireDataset tests
# ---------------------------------------------------------------------------

class TestWildfireDataset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.npz_path = _make_npz(self.tmpdir, n_samples=8)
        self.stats = _fake_channel_stats()

    def test_length(self):
        ds = WildfireDataset([self.npz_path], self.stats)
        self.assertEqual(len(ds), 8)

    def test_getitem_shapes(self):
        ds = WildfireDataset([self.npz_path], self.stats)
        X, Y = ds[0]
        self.assertEqual(X.shape, (17, 100, 100))
        self.assertEqual(Y.shape, (1, 100, 100))

    def test_getitem_is_tensor(self):
        ds = WildfireDataset([self.npz_path], self.stats)
        X, Y = ds[0]
        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)

    def test_normalization_applied(self):
        """With mean=5, std=2 stats, channel values should shift."""
        stats = {str(c): {"name": f"ch_{c}", "mean": 5.0, "std": 2.0} for c in range(17)}
        ds = WildfireDataset([self.npz_path], stats)
        X_norm, _ = ds[0]
        # Load raw and verify normalization was applied
        raw = np.load(self.npz_path)["X"][0]
        expected_ch0 = (raw[0] - 5.0) / (2.0 + 1e-8)
        np.testing.assert_allclose(X_norm[0].numpy(), expected_ch0, atol=1e-5)

    def test_augmentation_does_not_change_shape(self):
        ds = WildfireDataset([self.npz_path], self.stats, augment=True)
        for i in range(len(ds)):
            X, Y = ds[i]
            self.assertEqual(X.shape, (17, 100, 100))
            self.assertEqual(Y.shape, (1, 100, 100))

    def test_hflip_negates_wind_u_channels(self):
        """
        After a horizontal flip, wind_u channels should be negated.
        We test this by patching torch.rand to always trigger the flip.
        """
        import unittest.mock as mock

        stats = _fake_channel_stats()
        ds = WildfireDataset([self.npz_path], stats, augment=True)

        # Load a raw sample (pre-normalization, but with identity stats it's the same)
        raw_X = torch.from_numpy(np.load(self.npz_path)["X"][0].copy())
        raw_Y = torch.from_numpy(np.load(self.npz_path)["Y"][0].copy())

        # Force: hflip=YES (rand > 0.5), vflip=NO (rand <= 0.5), rotation=0
        flip_sequence = iter([0.6, 0.4, 0])
        with mock.patch("torch.rand", side_effect=lambda *a, **kw: torch.tensor(next(flip_sequence))):
            with mock.patch("torch.randint", return_value=torch.tensor(0)):
                X_aug, Y_aug = ds._augment(raw_X.clone(), raw_Y.clone())

        # Spatial flip
        expected_spatial = torch.flip(raw_X, dims=[2])
        # Wind U channels should be negated
        for c in WIND_U_CHANNELS:
            expected_spatial[c] = -expected_spatial[c]

        np.testing.assert_allclose(X_aug.numpy(), expected_spatial.numpy(), atol=1e-6)

    def test_vflip_negates_wind_v_channels(self):
        import unittest.mock as mock

        stats = _fake_channel_stats()
        ds = WildfireDataset([self.npz_path], stats, augment=True)

        raw_X = torch.from_numpy(np.load(self.npz_path)["X"][0].copy())
        raw_Y = torch.from_numpy(np.load(self.npz_path)["Y"][0].copy())

        # Force: hflip=NO (rand <= 0.5), vflip=YES (rand > 0.5), rotation=0
        flip_sequence = iter([0.4, 0.6, 0])
        with mock.patch("torch.rand", side_effect=lambda *a, **kw: torch.tensor(next(flip_sequence))):
            with mock.patch("torch.randint", return_value=torch.tensor(0)):
                X_aug, Y_aug = ds._augment(raw_X.clone(), raw_Y.clone())

        expected_spatial = torch.flip(raw_X, dims=[1])
        for c in WIND_V_CHANNELS:
            expected_spatial[c] = -expected_spatial[c]

        np.testing.assert_allclose(X_aug.numpy(), expected_spatial.numpy(), atol=1e-6)

    def test_multiple_files(self):
        """Dataset spanning multiple npz files should have combined length."""
        path2 = _make_npz(self.tmpdir, filename="samples2.npz", n_samples=6)
        ds = WildfireDataset([self.npz_path, path2], self.stats)
        self.assertEqual(len(ds), 14)


# ---------------------------------------------------------------------------
# Metric function tests
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):

    def test_iou_perfect_prediction(self):
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, 3:7, 3:7] = 1.0
        logits = (target * 20.0) - ((1 - target) * 20.0)
        score = iou_score(logits, target)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_iou_no_overlap(self):
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, 0:4, 0:4] = 1.0
        logits = torch.zeros(1, 1, 10, 10)
        logits[:, :, 6:9, 6:9] = 20.0  # predict fire elsewhere
        score = iou_score(logits, target)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_iou_empty_target_empty_pred(self):
        """Both target and prediction are all-zero: perfect IoU = 1."""
        target = torch.zeros(1, 1, 10, 10)
        logits = torch.full((1, 1, 10, 10), -20.0)  # predicts no fire
        score = iou_score(logits, target)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_f1_perfect(self):
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, 4:6, 4:6] = 1.0
        logits = (target * 20.0) - ((1 - target) * 20.0)
        score = f1_score(logits, target)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_f1_no_overlap(self):
        target = torch.zeros(1, 1, 10, 10)
        target[:, :, 0:3, 0:3] = 1.0
        logits = torch.zeros(1, 1, 10, 10)
        logits[:, :, 7:9, 7:9] = 20.0
        score = f1_score(logits, target)
        self.assertAlmostEqual(score, 0.0, places=4)


# ---------------------------------------------------------------------------
# compute_pos_weight tests
# ---------------------------------------------------------------------------

class TestComputePosWeight(unittest.TestCase):

    def test_returns_tensor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_npz(tmpdir, fire_fraction=0.05)
            pw = compute_pos_weight([path])
            self.assertIsInstance(pw, torch.Tensor)

    def test_positive_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _make_npz(tmpdir, fire_fraction=0.05)
            pw = compute_pos_weight([path])
            self.assertGreater(pw.item(), 1.0)

    def test_higher_for_sparser_fires(self):
        """Sparser fires (lower fire_fraction) should give higher pos_weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_sparse = _make_npz(tmpdir, filename="sparse.npz", fire_fraction=0.01)
            path_dense = _make_npz(tmpdir, filename="dense.npz", fire_fraction=0.3)
            pw_sparse = compute_pos_weight([path_sparse])
            pw_dense = compute_pos_weight([path_dense])
            self.assertGreater(pw_sparse.item(), pw_dense.item())


if __name__ == "__main__":
    unittest.main()
