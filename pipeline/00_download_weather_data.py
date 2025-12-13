import yaml
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.ops import unary_union
import rioxarray
import xarray as xr
import geopandas as gpd
import pandas as pd
import time
import os
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize
from xrspatial import slope # Keeping this import for the future enrichment step

# Load up the configuration file
with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

YEARS = config["YEARS"]
RAW_FIRE_DATA_DIR = config["RAW_FIRE_DATA_DIR"]
RAW_WEATHER_ZARR_DIR = config["RAW_WEATHER_ZARR_DIR"]
START_DATE = config["START_DATE"]
HRRR_URL = config["HRRR_URL"]
N_JOBS = config["N_JOBS"]
BUFFER_METERS = config["BUFFER_METERS"]
TEST_LIMIT = config["TEST_LIMIT"]
JSON_PATH = config["JSON_PATH"]

# --- Local DEM Settings ---
# Note: I'm skipping the DEM processing in this run to speed things up,
# but I'll need this path later for the enrichment script.
LOCAL_DEM_PATH = "data/static/colorado_dem_copernicus.tif"

# --- Helper Functions ---

def combine_geoms(series):
    return unary_union(series)

def process_fire_worker(row, weather_url, output_folder, input_crs, weather_crs, year):
    # Network handling
    MAX_RETRIES = 30 # Bumped this up to 30 to handle connection flakiness
    RETRY_DELAY = 5
    TARGET_RES = 500  # Sticking to 500m resolution for now
    
    fire_id = int(row['fireID'])
    
    # 1. Define the temporal window for this fire
    init_times = pd.date_range(
        start=row['t_start'], 
        end=row['t_end'], 
        freq='12h'
    )
    
    forecast_deltas = [
        pd.Timedelta(hours=0),
        pd.Timedelta(hours=6),
        pd.Timedelta(hours=12)
    ]

    for attempt in range(MAX_RETRIES):
        try:
            # Connect to the remote Zarr store
            ds = xr.open_zarr(weather_url, decode_coords="all", storage_options={'ssl': False})

            # Geometry Prep
            # Project the fire polygon to match the weather data CRS
            fire_gs = gpd.GeoSeries([row['geometry']], crs=input_crs)
            fire_proj = fire_gs.to_crs(weather_crs)
            
            # Calculate bounds with the safety buffer
            min_x, min_y, max_x, max_y = fire_proj.buffer(BUFFER_METERS).total_bounds

            # Step A: Coarse crop (Spatial)
            # Grab a rough chunk of the weather data first
            subset_spatial = ds.sel(
                x=slice(min_x, max_x),
                y=slice(max_y, min_y)
            )

            # Step B: Filter by time (Temporal)
            subset = subset_spatial.sel(
                init_time=init_times,
                lead_time=forecast_deltas,
                method="nearest"
            )

            # Sanity checks - skip if the slice is empty
            if subset.sizes['x'] == 0 or subset.sizes['y'] == 0:
                return f"SKIPPED_OUT_OF_BOUNDS"
            
            if subset.sizes['init_time'] == 0:
                return "SKIPPED_NO_TIME_MATCH"

            # Filter Variables
            # Only keeping the features needed for the model
            vars_to_keep = [
                'temperature_2m', 'relative_humidity_2m', 'wind_u_10m', 'wind_v_10m', 
                'precipitation_surface', 'total_cloud_cover_atmosphere',
                'downward_short_wave_radiation_flux_surface', 'wind_u_80m', 'wind_v_80m'
            ]
            available_vars = [v for v in vars_to_keep if v in subset.data_vars]
            subset = subset[available_vars]

            # --- Resolution Matching & Mask Generation ---

            # 1. Create the target 500m grid
            coarse_x = subset.x.values
            coarse_y = subset.y.values
            
            # Generate new coordinates
            new_x = np.arange(coarse_x.min(), coarse_x.max(), TARGET_RES)
            
            # Handle Y-axis orientation (HRRR is usually decreasing Y)
            if coarse_y[1] < coarse_y[0]:
                new_y = np.arange(coarse_y.max(), coarse_y.min(), -TARGET_RES)
            else:
                new_y = np.arange(coarse_y.min(), coarse_y.max(), TARGET_RES)

            # 2. Interpolate weather data to the new grid (Nearest Neighbor)
            weather_500m = subset.interp(x=new_x, y=new_y, method="nearest")
            
            # 3. Create the binary fire mask
            transform = weather_500m.rio.transform()
            out_shape = (weather_500m.sizes['y'], weather_500m.sizes['x'])
            
            mask_arr = rasterize(
                [(fire_proj.geometry.iloc[0], 1)],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype='uint8'
            )
            
            # Attach the mask to the dataset
            weather_500m['fire_mask'] = (('y', 'x'), mask_arr)
            
            return "SUCCESS"

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return f"ERROR after {MAX_RETRIES} attempts: {error_msg}"

# --- Main Pipeline Execution ---

def main():
    print("--- PIPELINE STEP 1: WEATHER DOWNLOAD (LOCAL DEM EDITION) ---")
    
    # 1. Grab the CO shapefile
    print("Loading Colorado boundary...", end=" ")
    try:
        states_gdf = gpd.read_file(JSON_PATH)
        co_boundary = states_gdf[states_gdf['NAME'] == 'Colorado']
        if co_boundary.empty:
            print("\nError: 'Colorado' not found in state boundary file.")
            return
        print("Done.")
    except Exception as e:
        print(f"\nError loading state boundary: {e}")
        return

    # 2. Loop through the fire years
    print("Loading and aggregating Fire Events...")
    for year in YEARS:
        print(f"Processing year {year}...")
        input_data = os.path.join(RAW_FIRE_DATA_DIR, f"feds_western_us_{year}_af_postprocessed.parquet")
        
        if not os.path.exists(input_data):
            print(f"  Skipping {year}: Input file not found.")
            continue

        # Read the raw fire data
        df = gpd.read_parquet(input_data) 
        df_flat = df.reset_index()
        
        # Group fire pixels into single events
        fire_events = df_flat.groupby('fireID').agg({
            't': ['min', 'max'],
            'hull': combine_geoms
        }).reset_index()
        fire_events.columns = ['fireID', 't_start', 't_end', 'geometry']
        
        # Convert back to GeoDataFrame
        valid_fires = gpd.GeoDataFrame(fire_events, geometry='geometry', crs=df.crs)
        valid_fires = valid_fires[valid_fires['t_start'] >= pd.to_datetime(START_DATE)].copy()

        # Filter for Colorado
        co_proj = co_boundary.to_crs(valid_fires.crs)
        valid_fires = valid_fires[valid_fires.intersects(co_proj.geometry.iloc[0])]
        
        print(f"  Fires remaining in Colorado: {len(valid_fires)}")
        
        if len(valid_fires) == 0:
            continue

        # Test Limit check
        if TEST_LIMIT:
            print(f"\n!!! TEST MODE: Processing first {TEST_LIMIT} fires !!!\n")
            valid_fires = valid_fires.head(TEST_LIMIT)

        # 4. Inspect the Weather Zarr for CRS
        print("Getting Weather CRS...", end=" ")
        try:
            with xr.open_zarr(HRRR_URL, decode_coords="all", storage_options={'ssl': False}) as ds:
                weather_crs = ds.rio.crs
            print("Done.")
        except Exception as e:
            print(f"\nCRITICAL ERROR: Could not inspect Weather Zarr. \n{e}")
            return

        # 5. Kick off parallel workers
        print(f"Starting parallel download for {year} with {N_JOBS} workers...")
        
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_fire_worker)(
                row, HRRR_URL, RAW_WEATHER_ZARR_DIR, df.crs, weather_crs, year
            )
            for index, row in tqdm(valid_fires.iterrows(), total=len(valid_fires), desc=f"Processing year {year}")
        )

        print("\n" + "="*30)
        print(f"DOWNLOAD COMPLETE for {year}")
        print("="*30)
        
        # Reporting loop
        counts = Counter(results)
        for status, count in counts.items():
            if status is None:
                print(f"WARNING: {count} fires returned None (Worker crash).")
                continue
            if "SKIPPED_OUT_OF_BOUNDS" in status:
                print(f"SKIPPED_OUT_OF_BOUNDS for {year}: {count}")
            else:
                print(f"{status} for {year}: {count}")

if __name__ == "__main__":
    main()