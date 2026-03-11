"""
Unit tests for pipeline/02_add_terrain_data.py

Tests cover:
- get_bbox_latlon: bounding box conversion with 5km buffer
- compute_terrain_stack: output shape, dtype, variable names, NaN filling
- process_zarr_terrain: SKIPPED when terrain already present, ERROR on fetch failure
"""

import sys
import os
import unittest
import importlib.util
import tempfile
import numpy as np
import xarray as xr
from unittest.mock import patch

# Run from project root so config.yaml is found
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Load the script as a module via importlib (filename starts with a digit)
_spec = importlib.util.spec_from_file_location("script_02", "pipeline/02_add_terrain_data.py")
script_02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script_02)

get_bbox_latlon = script_02.get_bbox_latlon
compute_terrain_stack = script_02.compute_terrain_stack
process_zarr_terrain = script_02.process_zarr_terrain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_ds(crs="EPSG:32611"):
    """
    Minimal xarray Dataset resembling a fire zarr.
    EPSG:32611 = WGS 84 / UTM zone 11N (western US, ~100 km grids).
    """
    import rioxarray  # noqa – registers .rio accessor
    x = np.linspace(400000.0, 500000.0, 100, dtype=np.float64)
    y = np.linspace(3900000.0, 3800000.0, 100, dtype=np.float64)  # descending (north→south)
    data = np.random.rand(1, 1, 100, 100).astype(np.float32)
    ds = xr.Dataset(
        {"temperature_2m": xr.DataArray(data, dims=["init_time", "lead_time", "y", "x"])},
        coords={"x": x, "y": y},
    )
    ds = ds.rio.write_crs(crs)
    return ds


def _make_fake_dem(target_ds):
    """Synthetic DEM DataArray on the same grid as target_ds."""
    import rioxarray  # noqa
    elev = np.random.uniform(1000.0, 3000.0, (100, 100)).astype(np.float32)
    dem = xr.DataArray(elev, dims=["y", "x"],
                       coords={"y": target_ds.y.values, "x": target_ds.x.values})
    dem = dem.rio.write_crs(target_ds.rio.crs)
    return dem


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetBboxLatlon(unittest.TestCase):

    def setUp(self):
        self.ds = _make_fake_ds()

    def test_returns_four_floats(self):
        bbox = get_bbox_latlon(self.ds)
        self.assertEqual(len(bbox), 4)
        for v in bbox:
            self.assertIsInstance(v, float)

    def test_lon_lat_ordering(self):
        lon_min, lat_min, lon_max, lat_max = get_bbox_latlon(self.ds)
        self.assertLess(lon_min, lon_max, "lon_min should be less than lon_max")
        self.assertLess(lat_min, lat_max, "lat_min should be less than lat_max")

    def test_result_in_valid_latlon_range(self):
        lon_min, lat_min, lon_max, lat_max = get_bbox_latlon(self.ds)
        self.assertGreater(lon_min, -180)
        self.assertLess(lon_max, 180)
        self.assertGreater(lat_min, -90)
        self.assertLess(lat_max, 90)

    def test_buffer_expands_bbox_vs_raw(self):
        """Buffered bbox must be larger than the raw projected bounds."""
        from pyproj import Transformer
        crs = self.ds.rio.crs
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        raw_lon_min, raw_lat_min = tr.transform(float(self.ds.x.min()), float(self.ds.y.min()))
        raw_lon_max, raw_lat_max = tr.transform(float(self.ds.x.max()), float(self.ds.y.max()))

        lon_min, lat_min, lon_max, lat_max = get_bbox_latlon(self.ds)
        self.assertLess(lon_min, raw_lon_min, "buffered lon_min should extend beyond raw")
        self.assertGreater(lon_max, raw_lon_max, "buffered lon_max should extend beyond raw")


class TestComputeTerrainStack(unittest.TestCase):

    def setUp(self):
        self.ds = _make_fake_ds()
        self.dem = _make_fake_dem(self.ds)

    def test_output_shape(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertEqual(result.shape, (3, 100, 100))

    def test_dims(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertEqual(tuple(result.dims), ("terrain_band", "y", "x"))

    def test_variable_names(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertListEqual(result.terrain_band.values.tolist(), ["elevation", "slope", "aspect"])

    def test_dtype_float32(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertEqual(result.dtype, np.float32)

    def test_no_nan_in_output(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertFalse(np.isnan(result.values).any(),
                         "Terrain stack must have no NaN after mean-filling")

    def test_elevation_range_reasonable(self):
        result = compute_terrain_stack(self.dem, self.ds)
        elev = result.sel(terrain_band="elevation").values
        # Should stay within the synthetic DEM range [1000, 3000] (bilinear interp may extrapolate slightly)
        self.assertGreater(float(elev.min()), 800.0)
        self.assertLess(float(elev.max()), 3200.0)

    def test_slope_non_negative(self):
        result = compute_terrain_stack(self.dem, self.ds)
        slope = result.sel(terrain_band="slope").values
        self.assertTrue((slope >= 0).all(), "Slope values should be non-negative degrees")

    def test_aspect_range(self):
        result = compute_terrain_stack(self.dem, self.ds)
        aspect = result.sel(terrain_band="aspect").values
        self.assertTrue((aspect >= 0).all(), "Aspect should be >= 0 degrees")
        self.assertTrue((aspect <= 360).all(), "Aspect should be <= 360 degrees")

    def test_nan_in_dem_filled(self):
        """Inject NaN into the DEM — output should still contain no NaN."""
        dem_nan = self.dem.copy()
        dem_nan.values[10:20, 10:20] = np.nan
        result = compute_terrain_stack(dem_nan, self.ds)
        self.assertFalse(np.isnan(result.values).any())

    def test_crs_preserved(self):
        result = compute_terrain_stack(self.dem, self.ds)
        self.assertIsNotNone(result.rio.crs)


class TestProcessZarrTerrain(unittest.TestCase):

    def test_skips_when_terrain_already_present(self):
        """process_zarr_terrain should return 'SKIPPED' and never call fetch_dem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "fire.zarr")
            ds = _make_fake_ds()
            terrain = xr.DataArray(
                np.zeros((3, 100, 100), dtype=np.float32),
                dims=["terrain_band", "y", "x"],
                coords={"terrain_band": ["elevation", "slope", "aspect"],
                        "y": ds.y.values, "x": ds.x.values},
                name="terrain",
            )
            ds_with_terrain = ds.assign(terrain=terrain)
            ds_with_terrain.to_zarr(zarr_path)

            with patch.object(script_02, "fetch_dem") as mock_fetch:
                result = process_zarr_terrain(zarr_path)
                mock_fetch.assert_not_called()
            self.assertEqual(result, "SKIPPED")

    def test_returns_error_string_on_fetch_failure(self):
        """Returns an 'ERROR: ...' string when fetch_dem raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "fire.zarr")
            ds = _make_fake_ds()
            ds.to_zarr(zarr_path)

            with patch.object(script_02, "fetch_dem", side_effect=RuntimeError("network down")):
                result = process_zarr_terrain(zarr_path)
            self.assertTrue(result.startswith("ERROR"))

    def test_success_appends_terrain(self):
        """When fetch_dem succeeds, 'terrain' should be written into the zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = os.path.join(tmpdir, "fire.zarr")
            ds = _make_fake_ds()
            ds.to_zarr(zarr_path)
            fake_dem = _make_fake_dem(ds)

            with patch.object(script_02, "fetch_dem", return_value=fake_dem):
                result = process_zarr_terrain(zarr_path)
            self.assertEqual(result, "SUCCESS")

            result_ds = xr.open_zarr(zarr_path, consolidated=False)
            self.assertIn("terrain", result_ds)
            self.assertEqual(result_ds["terrain"].shape, (3, 100, 100))


if __name__ == "__main__":
    unittest.main()
