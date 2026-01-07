import os
import glob
import rioxarray
import numpy as np
import xarray as xr
from tqdm import tqdm
from rasterio.enums import Resampling

def create_raw_physics_stack(patch):
    """
    Transforms a 1-channel Fuel ID patch into a 3-channel Physical Property patch.
    Input: xarray.DataArray (1, H, W)
    Output: xarray.DataArray (variable: 3, y, x)
    """
    # Mapping table: {ID: [Fuel_Load, Fuel_Bed_Depth, Moisture_Extinction]}
    mapping = {
        1: [0.74, 1.0, 12], 2: [2.00, 1.0, 15], 3: [3.01, 2.5, 25],
        4: [13.00, 6.0, 20], 5: [3.50, 2.0, 20], 6: [6.00, 2.5, 25],
        7: [4.86, 2.5, 40], 8: [5.00, 0.2, 30], 9: [3.46, 0.2, 25],
        10: [12.00, 1.0, 25], 11: [11.50, 1.0, 15], 12: [34.60, 2.3, 20],
        13: [58.10, 3.0, 25]
    }

    # Extract the 2D grid (H, W)
    data_2d = patch.values.squeeze()
    h, w = data_2d.shape
    
    # Initialize a 3D numpy array (3, H, W) filled with 0.0 (for non-burnables)
    physics_array = np.zeros((3, h, w), dtype=np.float32)

    # Fill the channels using the mapping
    for fuel_id, phys_vals in mapping.items():
        mask = (data_2d == fuel_id)
        physics_array[0, mask] = phys_vals[0] # Load
        physics_array[1, mask] = phys_vals[1] # Depth
        physics_array[2, mask] = phys_vals[2] # Moisture

    # Wrap back into an xarray to keep your coordinates and CRS
    stack_xr = xr.DataArray(
        data=physics_array,
        dims=("variable", "y", "x"),
        coords={
            "variable": ["fuel_load", "fuel_depth", "moisture_extinction"],
            "y": patch.y,
            "x": patch.x
        },
        name="physical_fuels"
    )
    
    # Transfer the spatial projection (CRS) from the original patch
    stack_xr.rio.write_crs(patch.rio.crs, inplace=True)
    
    return stack_xr

def main():
    # Configuration
    FUEL_PATHS = {
        2020: "data/static/fuel_2020_data/Tif/LC19_F13_200.tif",
        2021: "data/static/fuel_2021_data/Tif/LC20_F13_200.tif"
    }
    RAW_WEATHER_DIR = "data/raw_weather_zarr"
    YEARS = [2020, 2021]

    for year in YEARS:
        if year not in FUEL_PATHS:
            print(f"Skipping year {year}: No fuel path configured.")
            continue

        fuel_path = FUEL_PATHS[year]
        if not os.path.exists(fuel_path):
            print(f"CRITICAL: Fuel data not found at {fuel_path}")
            continue

        # Load the raster lazily (chunks=True is key for large files)
        print(f"Opening Fuel Data for {year} (Lazy)...")
        try:
            fuel_data = rioxarray.open_rasterio(fuel_path, chunks={'x': 4096, 'y': 4096})
        except Exception as e:
            print(f"Error opening fuel data: {e}")
            continue

        year_dir = os.path.join(RAW_WEATHER_DIR, str(year))
        if not os.path.exists(year_dir):
            print(f"Skipping year {year}: Directory {year_dir} not found.")
            continue

        zarr_files = glob.glob(os.path.join(year_dir, "*.zarr"))
        print(f"Processing {len(zarr_files)} fires for year {year}...")

        for zarr_path in tqdm(zarr_files, desc=f"Year {year}"):
            try:
                # 1. Open Zarr to get bounds/CRS
                ds = xr.open_zarr(zarr_path, decode_coords="all", consolidated=False)
                
                # Check if already processed
                if 'physical_fuels' in ds:
                    ds.close()
                    continue

                if ds.rio.crs is None:
                    print(f"  Skipping {os.path.basename(zarr_path)}: No CRS found.")
                    ds.close()
                    continue

                # 2. Clip Fuel Data (Optimization)
                # Transform bounds to Fuel CRS
                try:
                    fuel_crs = fuel_data.rio.crs
                    # Get bounds of the weather grid
                    minx, miny, maxx, maxy = ds.rio.transform_bounds(fuel_crs)
                    
                    # Buffer by ~10km to be safe during reprojection
                    pad = 10000 
                    fuel_subset = fuel_data.rio.clip_box(
                        minx - pad, miny - pad, 
                        maxx + pad, maxy + pad
                    )
                except Exception as e:
                    print(f"  Clip failed for {os.path.basename(zarr_path)}: {e}")
                    ds.close()
                    continue

                # 3. Reproject to Weather Grid (1km)
                # Use Nearest Neighbor for categorical fuel IDs
                fuel_aligned = fuel_subset.rio.reproject_match(
                    ds,
                    resampling=Resampling.nearest
                )

                # 4. Create Physics Stack
                physics_stack = create_raw_physics_stack(fuel_aligned)

                # 5. Append to Zarr
                physics_ds = physics_stack.to_dataset()
                ds.close() # Close read handle
                physics_ds.to_zarr(zarr_path, mode='a')

            except Exception as e:
                print(f"  Error processing {os.path.basename(zarr_path)}: {e}")

if __name__ == "__main__":
    main()
