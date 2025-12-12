from shapely.ops import unary_union
from tqdm import tqdm
from geocube.api.core import make_geocube
import rioxarray
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import yaml

with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

YEARS = config["YEARS"]
RAW_FIRE_DATA_DIR = config["RAW_FIRE_DATA_DIR"]
RASTERIZED_FIRES_DIR = config["RASTERIZED_FIRES_DIR"]
EXAMPLE_WEATHER_PATH = config["EXAMPLE_WEATHER_PATH"]
BUFFER_METERS = config["BUFFER_METERS"]
GRID_SIZE_METERS = config["GRID_SIZE_METERS"]
import warnings

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_weather_crs(path):
    """Loads a single weather file to extract the target Coordinate Reference System."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference weather file not found at: {path}\nPlease check the path or point to a valid Zarr file.")
    with xr.open_zarr(path, consolidated=False) as ds:
        return ds.rio.crs

def process_year(year, target_crs):
    """
    Loads fire data for a specific year, rasterizes each fire, 
    and saves them into a year-specific output directory.
    """
    # 1. Setup Paths
    input_file = f"feds_western_us_{year}_af_postprocessed.parquet"
    input_path = os.path.join(RAW_FIRE_DATA_DIR, input_file)
    output_dir = os.path.join(RASTERIZED_FIRES_DIR, str(year))
    
    if not os.path.exists(input_path):
        print(f"Skipping {year}: File not found ({input_path})")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load and Prepare Data
    print(f"[{year}] Loading fire data...")
    fire_df = gpd.read_parquet(input_path)
    fire_df = fire_df.reset_index()

    # Ensure 'hull' is the active geometry and rename to standard 'geometry'
    # (Adjust 'hull' if your column name differs in specific files)
    if 'hull' in fire_df.columns:
        fire_df = fire_df.set_geometry("hull")
    fire_df = fire_df.rename_geometry("geometry")

    # Reproject to match HRRR Weather Data
    print(f"[{year}] Reprojecting to HRRR CRS...")
    fire_df = fire_df.to_crs(target_crs)

    # 3. Process Each Fire
    # Group by fireID to handle the full history of each fire
    grouped_fires = fire_df.groupby('fireID')
    print(f"[{year}] Processing {len(grouped_fires)} unique fires...")

    for fire_id, fire_group in tqdm(grouped_fires, desc=f"Rasterizing {year}"):
        try:
            # A. Define the Master Grid (The Union of all extents)
            #    We calculate the bounds of the entire fire history + buffer
            #    Using unary_union is generally robust across Shapely versions
            union_poly = unary_union(fire_group.geometry)
            minx, miny, maxx, maxy = union_poly.buffer(BUFFER_METERS).bounds
            
            # B. Create the Target Grid (500m Resolution)
            #    Using np.arange to snap to a clean grid starting from the bounds
            x_coords = np.arange(minx, maxx, GRID_SIZE_METERS)
            y_coords = np.arange(maxy, miny, -GRID_SIZE_METERS) # Max -> Min for Y
            
            #    Skip if grid is empty (edge case for tiny/invalid geometries)
            if len(x_coords) == 0 or len(y_coords) == 0:
                continue

            template = xr.DataArray(
                data=np.zeros((len(y_coords), len(x_coords))),
                coords={"y": y_coords, "x": x_coords},
                dims=("y", "x")
            )
            template.rio.write_crs(target_crs, inplace=True)

            # C. Rasterize
            #    Ensure time is in datetime format for stacking
            fire_group['time'] = pd.to_datetime(fire_group['t'])

            raster_cube = make_geocube(
                vector_data=fire_group,
                measurements=["fireID"], # Rasterizes the fireID (could be useful for ID checks)
                like=template,           # Forces alignment to our union grid
                group_by="time",         # Stacks snapshots along time dimension
                fill=0
            )

            # D. Cleanup & Save
            raster_cube = raster_cube.rename({'fireID': 'fire_mask'})
            
            # Optional: Convert to binary mask (0/1) instead of raw ID
            # raster_cube['fire_mask'] = (raster_cube['fire_mask'] > 0).astype('int8')
            
            save_path = os.path.join(output_dir, f"fireID_{fire_id}_mask.zarr")
            raster_cube.to_zarr(save_path, mode='w', consolidated=False)

        except Exception as e:
            # Log error but keep processing other fires
            # Common errors: Empty geometries, projection issues
            print(f"Error processing Fire {fire_id}: {e}")
            continue

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":
    print("--- STARTING MULTI-YEAR RASTERIZATION ---")
    
    # 1. Get CRS once
    try:
        print("Fetching Reference CRS...", end=" ")
        hrrr_crs = get_weather_crs(EXAMPLE_WEATHER_PATH)
        print("Done.")
        print(f"Target CRS: {hrrr_crs}")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        exit(1)

    # 2. Iterate Years
    for year in YEARS:
        print(f"\n" + "="*40)
        print(f"STARTING YEAR: {year}")
        print("="*40)
        process_year(year, hrrr_crs)
        
    print("\n--- ALL YEARS COMPLETE ---")