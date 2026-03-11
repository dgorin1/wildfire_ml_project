import os
import glob
import time
import warnings
import yaml
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from tqdm import tqdm
from collections import Counter
import xrspatial

try:
    import py3dep
except ImportError:
    raise ImportError("py3dep is required. Install with: pip install py3dep")

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

YEARS = config["YEARS"]
RAW_WEATHER_ZARR_DIR = config["RAW_WEATHER_ZARR_DIR"]
DEM_RESOLUTION_M = config.get("DEM_RESOLUTION_M", 30)

MAX_RETRIES = 10
BASE_DELAY = 5


def get_bbox_latlon(ds):
    """
    Convert the zarr's x/y bounds (projected, LCC meters) to EPSG:4326,
    with a 5km buffer to avoid edge artifacts after reprojection.
    Returns (minx, miny, maxx, maxy) in lon/lat degrees.
    """
    buffer_m = 5000
    minx = float(ds.x.min()) - buffer_m
    maxx = float(ds.x.max()) + buffer_m
    miny = float(ds.y.min()) - buffer_m
    maxy = float(ds.y.max()) + buffer_m

    from pyproj import Transformer
    src_crs = ds.rio.crs
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(minx, miny)
    lon_max, lat_max = transformer.transform(maxx, maxy)
    return (lon_min, lat_min, lon_max, lat_max)


def fetch_dem(bbox_latlon, resolution_m=30):
    """
    Download DEM from USGS 3DEP via py3dep for the given lat/lon bounding box.
    Returns an xr.DataArray. Raises RuntimeError after MAX_RETRIES failures.
    Note: py3dep covers CONUS, Alaska, and Hawaii only.
    """
    for attempt in range(MAX_RETRIES):
        try:
            dem = py3dep.get_dem(bbox_latlon, resolution=resolution_m, crs="EPSG:4326")
            return dem
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                raise RuntimeError(f"DEM fetch failed after {MAX_RETRIES} attempts: {e}")


def compute_terrain_stack(dem_da, target_ds):
    """
    Reproject DEM to match the target zarr grid (bilinear for continuous elevation),
    compute slope and aspect via xrspatial, and return a (3, y, x) DataArray
    with variable coordinate ["elevation", "slope", "aspect"].
    """
    # Reproject DEM to match the weather grid exactly
    dem_aligned = dem_da.rio.reproject_match(target_ds, resampling=Resampling.bilinear)

    # Drop extra dims if present (py3dep may return a 3D array with a band dim)
    if dem_aligned.ndim == 3:
        dem_aligned = dem_aligned.squeeze(drop=True)

    # Ensure we have a 2D DataArray named consistently
    dem_2d = dem_aligned.rename("elevation")

    # Compute slope (degrees) and aspect (degrees) using xrspatial
    slope_da = xrspatial.slope(dem_2d)
    aspect_da = xrspatial.aspect(dem_2d)

    # Fill any NaN values with the mean of the valid data (edge-of-coverage cases)
    for da, name in [(dem_2d, "elevation"), (slope_da, "slope"), (aspect_da, "aspect")]:
        nan_count = int(np.isnan(da.values).sum())
        if nan_count > 0:
            fill_val = float(np.nanmean(da.values))
            warnings.warn(f"{name}: filling {nan_count} NaN pixels with mean ({fill_val:.2f})")

    elev_vals = np.where(np.isnan(dem_2d.values), float(np.nanmean(dem_2d.values)), dem_2d.values)
    slope_vals = np.where(np.isnan(slope_da.values), float(np.nanmean(slope_da.values)), slope_da.values)
    aspect_vals = np.where(np.isnan(aspect_da.values), float(np.nanmean(aspect_da.values)), aspect_da.values)

    stack = np.stack([elev_vals, slope_vals, aspect_vals], axis=0).astype(np.float32)

    terrain_da = xr.DataArray(
        data=stack,
        dims=("terrain_band", "y", "x"),
        coords={
            "terrain_band": ["elevation", "slope", "aspect"],
            "y": target_ds.y,
            "x": target_ds.x,
        },
        name="terrain",
    )
    terrain_da.rio.write_crs(target_ds.rio.crs, inplace=True)
    return terrain_da


def process_zarr_terrain(zarr_path):
    """
    Fetch and append terrain data to a single zarr file.
    Returns a status string: "SUCCESS", "SKIPPED", or "ERROR: <message>".
    """
    try:
        ds = xr.open_zarr(zarr_path, decode_coords="all", consolidated=False)

        if "terrain" in ds:
            ds.close()
            return "SKIPPED"

        if ds.rio.crs is None:
            ds.close()
            return "ERROR: No CRS found in zarr"

        bbox_latlon = get_bbox_latlon(ds)
        dem_da = fetch_dem(bbox_latlon, resolution_m=DEM_RESOLUTION_M)
        terrain_da = compute_terrain_stack(dem_da, ds)
        ds.close()

        terrain_ds = terrain_da.to_dataset()
        terrain_ds.to_zarr(zarr_path, mode="a")
        return "SUCCESS"

    except Exception as e:
        return f"ERROR: {e}"


def main():
    all_zarr_files = []
    for year in YEARS:
        year_dir = os.path.join(RAW_WEATHER_ZARR_DIR, str(year))
        if not os.path.isdir(year_dir):
            print(f"Skipping year {year}: directory not found at {year_dir}")
            continue
        files = glob.glob(os.path.join(year_dir, "*.zarr"))
        all_zarr_files.extend(files)

    print(f"Found {len(all_zarr_files)} zarr files across years {YEARS}")

    counts = Counter()
    for zarr_path in tqdm(all_zarr_files, desc="Adding terrain"):
        status = process_zarr_terrain(zarr_path)
        key = status.split(":")[0]
        counts[key] += 1
        if key == "ERROR":
            tqdm.write(f"  {os.path.basename(zarr_path)}: {status}")

    print("\nDone.")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
