"""
Unit tests for pipeline/03_assemble_training_samples.py

Tests cover:
- build_input: output shape (17, 100, 100), channel ordering, NaN filling
- build_target: output shape (1, 100, 100), binary values
- is_boundary_fire: correctly detects fire pixels on the grid edge
- extract_samples: correct number of pairs from multi-timestep dataset
- SkipFireException: raised for missing physical_fuels / terrain / too few timesteps
- compute_norm_stats: mean/std computed correctly per channel
"""

import sys
import os
import unittest
import importlib.util
import tempfile
import numpy as np
import xarray as xr
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

_spec = importlib.util.spec_from_file_location("script_03", "pipeline/03_assemble_training_samples.py")
script_03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script_03)

build_input = script_03.build_input
build_target = script_03.build_target
is_boundary_fire = script_03.is_boundary_fire
extract_samples = script_03.extract_samples
load_zarr_safe = script_03.load_zarr_safe
SkipFireException = script_03.SkipFireException
compute_norm_stats = script_03.compute_norm_stats
WEATHER_VARS = script_03.WEATHER_VARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_zarr_ds(n_times=3, has_fuels=True, has_terrain=True):
    """
    Build a synthetic xarray Dataset that mirrors the structure of a real fire zarr.

    Dimensions:
      weather vars: (init_time, lead_time, y, x)
      fire_mask:    (init_time, y, x)
      physical_fuels: (variable, y, x)
      terrain:      (variable, y, x)
    """
    import rioxarray  # noqa

    H, W = 100, 100
    n_lead = 3
    x = np.linspace(0.0, 99000.0, W, dtype=np.float64)
    y = np.linspace(99000.0, 0.0, H, dtype=np.float64)
    init_times = pd.date_range("2020-07-01", periods=n_times, freq="12h")
    lead_times = [pd.Timedelta(hours=h) for h in [0, 6, 12]]

    data_vars = {}

    for var in WEATHER_VARS:
        vals = np.random.rand(n_times, n_lead, H, W).astype(np.float32)
        data_vars[var] = xr.DataArray(
            vals,
            dims=["init_time", "lead_time", "y", "x"],
            coords={"init_time": init_times, "lead_time": lead_times, "y": y, "x": x},
        )

    # fire_mask: small square fire growing over time
    masks = np.zeros((n_times, H, W), dtype=np.float32)
    for t in range(n_times):
        size = 10 + t * 5
        masks[t, 40:40+size, 40:40+size] = 1.0
    data_vars["fire_mask"] = xr.DataArray(
        masks,
        dims=["init_time", "y", "x"],
        coords={"init_time": init_times, "y": y, "x": x},
    )

    if has_fuels:
        fuels = np.random.rand(3, H, W).astype(np.float32)
        data_vars["physical_fuels"] = xr.DataArray(
            fuels,
            dims=["variable", "y", "x"],
            coords={"variable": ["fuel_load", "fuel_depth", "moisture_extinction"], "y": y, "x": x},
        )

    if has_terrain:
        terrain = np.random.rand(3, H, W).astype(np.float32) * 90  # 0–90 degrees
        data_vars["terrain"] = xr.DataArray(
            terrain,
            dims=["terrain_band", "y", "x"],
            coords={"terrain_band": ["elevation", "slope", "aspect"], "y": y, "x": x},
        )

    ds = xr.Dataset(data_vars)
    ds = ds.rio.write_crs("EPSG:32611")
    return ds


def _save_fake_zarr(tmpdir, name="fire.zarr", **kwargs):
    ds = _make_fake_zarr_ds(**kwargs)
    path = os.path.join(tmpdir, name)
    ds.to_zarr(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildInput(unittest.TestCase):

    def setUp(self):
        self.ds = _make_fake_zarr_ds(n_times=3)

    def test_output_shape(self):
        X = build_input(self.ds, 0)
        self.assertEqual(X.shape, (17, 100, 100))

    def test_output_dtype(self):
        X = build_input(self.ds, 0)
        self.assertEqual(X.dtype, np.float32)

    def test_no_nan_in_output(self):
        X = build_input(self.ds, 0)
        self.assertFalse(np.isnan(X).any(), "build_input output should have no NaN")

    def test_nan_in_weather_filled(self):
        """NaN injected into a weather channel should be filled with channel mean."""
        ds = _make_fake_zarr_ds(n_times=3)
        # Inject NaN into temperature_2m
        arr = ds["temperature_2m"].values.copy()
        arr[0, 2, 5:10, 5:10] = np.nan
        ds["temperature_2m"] = xr.DataArray(arr, dims=ds["temperature_2m"].dims,
                                             coords=ds["temperature_2m"].coords)
        X = build_input(ds, 0)
        self.assertFalse(np.isnan(X).any())

    def test_fire_mask_is_channel_0(self):
        """Channel 0 should equal the fire_mask at the given timestep."""
        ds = _make_fake_zarr_ds(n_times=3)
        X = build_input(ds, 1)
        expected = ds["fire_mask"].isel(init_time=1).values.astype(np.float32)
        np.testing.assert_array_equal(X[0], expected)

    def test_sin_cos_aspect_channels(self):
        """Channels 15 (sin) and 16 (cos) should be in [-1, 1]."""
        X = build_input(self.ds, 0)
        self.assertTrue((X[15] >= -1).all() and (X[15] <= 1).all())
        self.assertTrue((X[16] >= -1).all() and (X[16] <= 1).all())

    def test_sin_cos_identity(self):
        """sin^2 + cos^2 should equal 1 everywhere."""
        X = build_input(self.ds, 0)
        sin_ch = X[15].astype(np.float64)
        cos_ch = X[16].astype(np.float64)
        np.testing.assert_allclose(sin_ch**2 + cos_ch**2, np.ones((100, 100)), atol=1e-5)


class TestBuildTarget(unittest.TestCase):

    def setUp(self):
        self.ds = _make_fake_zarr_ds(n_times=3)

    def test_output_shape(self):
        Y = build_target(self.ds, 1)
        self.assertEqual(Y.shape, (1, 100, 100))

    def test_output_dtype(self):
        Y = build_target(self.ds, 1)
        self.assertEqual(Y.dtype, np.float32)

    def test_values_binary(self):
        """Target should contain only 0.0 and 1.0."""
        Y = build_target(self.ds, 1)
        unique = np.unique(Y)
        for v in unique:
            self.assertIn(v, [0.0, 1.0])

    def test_matches_fire_mask(self):
        Y = build_target(self.ds, 2)
        expected = self.ds["fire_mask"].isel(init_time=2).values.astype(np.float32)
        np.testing.assert_array_equal(Y.squeeze(), expected)


class TestIsBoundaryFire(unittest.TestCase):

    def test_fire_on_top_edge(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        target[0, 0, 50] = 1.0
        self.assertTrue(is_boundary_fire(target))

    def test_fire_on_bottom_edge(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        target[0, 99, 50] = 1.0
        self.assertTrue(is_boundary_fire(target))

    def test_fire_on_left_edge(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        target[0, 50, 0] = 1.0
        self.assertTrue(is_boundary_fire(target))

    def test_fire_on_right_edge(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        target[0, 50, 99] = 1.0
        self.assertTrue(is_boundary_fire(target))

    def test_fire_in_center_not_boundary(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        target[0, 40:60, 40:60] = 1.0
        self.assertFalse(is_boundary_fire(target))

    def test_all_zeros_not_boundary(self):
        target = np.zeros((1, 100, 100), dtype=np.float32)
        self.assertFalse(is_boundary_fire(target))


class TestExtractSamples(unittest.TestCase):

    def test_n_samples_equals_n_times_minus_one(self):
        ds = _make_fake_zarr_ds(n_times=5)
        samples = extract_samples(ds, "fake_path.zarr")
        self.assertEqual(len(samples), 4)

    def test_sample_keys(self):
        ds = _make_fake_zarr_ds(n_times=2)
        samples = extract_samples(ds, "fake_path.zarr")
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertIn("X", s)
        self.assertIn("Y", s)
        self.assertIn("boundary", s)
        self.assertIn("t_idx", s)

    def test_sample_shapes(self):
        ds = _make_fake_zarr_ds(n_times=3)
        samples = extract_samples(ds, "fake_path.zarr")
        for s in samples:
            self.assertEqual(s["X"].shape, (17, 100, 100))
            self.assertEqual(s["Y"].shape, (1, 100, 100))

    def test_boundary_flag_center_fire(self):
        """A fire confined to the center should not be flagged as boundary."""
        ds = _make_fake_zarr_ds(n_times=2)
        samples = extract_samples(ds, "fake_path.zarr")
        self.assertFalse(samples[0]["boundary"])

    def test_boundary_flag_edge_fire(self):
        """A fire touching the grid edge should be flagged."""
        ds = _make_fake_zarr_ds(n_times=2)
        # Put fire on top row of second timestep
        mask_vals = ds["fire_mask"].values.copy()
        mask_vals[1, 0, :] = 1.0
        ds["fire_mask"] = xr.DataArray(mask_vals, dims=ds["fire_mask"].dims,
                                       coords=ds["fire_mask"].coords)
        samples = extract_samples(ds, "fake_path.zarr")
        self.assertTrue(samples[0]["boundary"])

    def test_single_timestep_returns_no_samples(self):
        """With only 1 timestep there are no consecutive pairs."""
        ds = _make_fake_zarr_ds(n_times=1)
        samples = extract_samples(ds, "fake_path.zarr")
        self.assertEqual(len(samples), 0)


class TestLoadZarrSafe(unittest.TestCase):

    def test_raises_when_missing_physical_fuels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_fake_zarr(tmpdir, "no_fuels.zarr", has_fuels=False)
            with self.assertRaises(SkipFireException) as ctx:
                load_zarr_safe(path)
            self.assertIn("physical_fuels", str(ctx.exception))

    def test_raises_when_missing_terrain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_fake_zarr(tmpdir, "no_terrain.zarr", has_terrain=False)
            with self.assertRaises(SkipFireException) as ctx:
                load_zarr_safe(path)
            self.assertIn("terrain", str(ctx.exception))

    def test_raises_when_too_few_timesteps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_fake_zarr(tmpdir, "one_time.zarr", n_times=1)
            with self.assertRaises(SkipFireException) as ctx:
                load_zarr_safe(path)
            self.assertIn("timestep", str(ctx.exception))

    def test_returns_dataset_when_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _save_fake_zarr(tmpdir, "valid.zarr", n_times=3)
            ds = load_zarr_safe(path)
            self.assertIsInstance(ds, xr.Dataset)
            ds.close()


class TestComputeNormStats(unittest.TestCase):

    def test_output_has_17_channels(self):
        # Create fake samples list (list of lists, one per year)
        rng = np.random.default_rng(42)
        fake_samples = [[
            {"X": rng.random((17, 100, 100)).astype(np.float32)}
            for _ in range(5)
        ]]
        stats = compute_norm_stats(fake_samples)
        self.assertEqual(len(stats), 17)

    def test_stats_have_mean_and_std(self):
        rng = np.random.default_rng(0)
        fake_samples = [[
            {"X": rng.random((17, 100, 100)).astype(np.float32)}
            for _ in range(3)
        ]]
        stats = compute_norm_stats(fake_samples)
        for c in range(17):
            self.assertIn("mean", stats[c])
            self.assertIn("std", stats[c])

    def test_stats_values_reasonable(self):
        """Stats for Uniform[0,1] data should have mean ~0.5, std ~0.289."""
        rng = np.random.default_rng(7)
        samples = [{"X": rng.random((17, 100, 100)).astype(np.float32)} for _ in range(20)]
        stats = compute_norm_stats([samples])
        for c in range(17):
            self.assertAlmostEqual(stats[c]["mean"], 0.5, delta=0.05)
            self.assertAlmostEqual(stats[c]["std"], 0.289, delta=0.02)

    def test_std_positive(self):
        rng = np.random.default_rng(1)
        samples = [{"X": rng.random((17, 100, 100)).astype(np.float32)} for _ in range(5)]
        stats = compute_norm_stats([samples])
        for c in range(17):
            self.assertGreater(stats[c]["std"], 0)


if __name__ == "__main__":
    unittest.main()
