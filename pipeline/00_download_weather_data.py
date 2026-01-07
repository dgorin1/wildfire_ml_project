import yaml
from collections import Counter
from tqdm import tqdm
from joblib import Parallel, delayed
from shapely.ops import unary_union
from shapely.geometry import box # Needed for the new box logic
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
from xrspatial import slope 

# Load up my configuration file
with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

YEARS = config["YEARS"]
RAW_FIRE_DATA_DIR = config["RAW_FIRE_DATA_DIR"]
RAW_WEATHER_ZARR_DIR = config["RAW_WEATHER_ZARR_DIR"]
START_DATE = config["START_DATE"]
HRRR_URL = config["HRRR_URL"]
N_JOBS = config["N_JOBS"]
# BUFFER_METERS is ignored now in favor of fixed 100km grid
TEST_LIMIT = config["TEST_LIMIT"]
JSON_PATH = config["JSON_PATH"]

# --- Local DEM Settings ---
LOCAL_DEM_PATH = "data/static/colorado_dem_copernicus.tif"

# --- Constants ---
FIXED_WINDOW_SIZE_METERS = 100000 # 100 km box [cite: 84]
HALF_WINDOW = FIXED_WINDOW_SIZE_METERS / 2

# --- Helper Functions ---

def combine_geoms(series):
    return unary_union(series)

def process_fire_worker(row, fire_rows, weather_url, output_folder, input_crs, weather_crs, year):
    # Network handling
    MAX_RETRIES = 10
    BASE_DELAY = 5
    TARGET_RES = 1000  # 1km resolution to match the paper [cite: 80]
    
    fire_id = int(row['fireID'])
    
    # 1. Define the temporal window for this fire
    init_times = pd.date_range(
        start=row['t_start'], 
        end=row['t_end'], 
        freq='12h' # 12h temporal resolution [cite: 69]
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
            
            # --- CENTROID-BASED CROPPING ---
            # 1. Get Centroid
            centroid = fire_proj.geometry.iloc[0].centroid
            c_x, c_y = centroid.x, centroid.y
            
            # 2. Create fixed 100km box around centroid [cite: 84]
            min_x = c_x - HALF_WINDOW
            max_x = c_x + HALF_WINDOW
            min_y = c_y - HALF_WINDOW
            max_y = c_y + HALF_WINDOW
            
            # Step A: Coarse crop (Spatial)
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

            # Sanity checks
            if subset.sizes['x'] == 0 or subset.sizes['y'] == 0:
                return f"SKIPPED_OUT_OF_BOUNDS"
            
            if subset.sizes['init_time'] == 0:
                return "SKIPPED_NO_TIME_MATCH"

            # Filter Variables [cite: 66, 77]
            vars_to_keep = [
                'temperature_2m', 'relative_humidity_2m', 'wind_u_10m', 'wind_v_10m', 
                'precipitation_surface', 'total_cloud_cover_atmosphere',
                'downward_short_wave_radiation_flux_surface', 'wind_u_80m', 'wind_v_80m'
            ]
            available_vars = [v for v in vars_to_keep if v in subset.data_vars]
            subset = subset[available_vars]

            # --- Resolution Matching & Mask Generation ---

            # 1. Create the target 1000m grid explicitly using the fixed bounds 
            # This ensures every single file has the exact same dimensions
            num_points = int(FIXED_WINDOW_SIZE_METERS / TARGET_RES)
            new_x = np.linspace(min_x, max_x, num_points, endpoint=False)
            new_y = np.linspace(max_y, min_y, num_points, endpoint=False)

            # 2. Upsample/Interpolate Weather 
            # Updated from "nearest" to "linear" to match bi-linear interpolation in paper 
            weather_1km = subset.interp(x=new_x, y=new_y, method="linear")
            
            # 3. Rasterize Fire Mask
            transform = weather_1km.rio.transform()
            out_shape = (weather_1km.sizes['y'], weather_1km.sizes['x'])
            
            # Project the detailed fire history to match weather CRS
            fire_rows_gdf = gpd.GeoDataFrame(fire_rows, geometry='hull', crs=input_crs)
            fire_rows_proj = fire_rows_gdf.to_crs(weather_crs)
            # Ensure time column is datetime for comparison
            fire_rows_proj['t'] = pd.to_datetime(fire_rows_proj['t'])
            
            # Generate a time-dependent mask for each init_time
            masks = []
            for t in weather_1km.init_time.values:
                ts = pd.to_datetime(t)
                # Filter for fire polygons that existed at or before this time (Cumulative) [cite: 357]
                valid_polys = fire_rows_proj[fire_rows_proj['t'] <= ts]
                
                if valid_polys.empty:
                    mask_arr = np.zeros(out_shape, dtype='uint8')
                else:
                    shapes = [(g, 1) for g in valid_polys.geometry]
                    # 1 for burned pixels and 0 for non-burned pixels [cite: 82]
                    mask_arr = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, all_touched=True, dtype='uint8')
                
                masks.append(mask_arr)
            
            # Attach the mask
            weather_1km['fire_mask'] = (('init_time', 'y', 'x'), np.stack(masks))
            
            # 5. Save results
            year_output_folder = os.path.join(output_folder, str(year))
            os.makedirs(year_output_folder, exist_ok=True)

            file_name = f"fireID_{fire_id}_weather_{year}.zarr"
            save_path = os.path.join(year_output_folder, file_name)
            
            weather_1km.to_zarr(save_path, mode='w', consolidated=False)

            return "SUCCESS"

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                # Linear backoff: 5s, 10s, 15s...
                time.sleep(BASE_DELAY * (attempt + 1))
                continue
            else:
                return f"ERROR after {MAX_RETRIES} attempts: {error_msg}"

# --- Main Pipeline Execution ---

def main():
    print("--- PIPELINE STEP 1: WEATHER DOWNLOAD (FIXED 100KM GRID) ---")
    
    # 2. Loop through the fire years
    print("Loading and aggregating Fire Events...")
    for year in YEARS:
        print(f"Processing year {year}...")
        input_data = os.path.join(RAW_FIRE_DATA_DIR, f"feds_western_us_{year}_af_postprocessed.parquet")
        
        if not os.path.exists(input_data):
            print(f"  Skipping {year}: Input file not found.")
            continue

        # Read the raw fire data [cite: 54, 69]
        df = gpd.read_parquet(input_data) 
        df_flat = df.reset_index()
        df_flat['t'] = pd.to_datetime(df_flat['t'])
        
        # Create a lookup for raw fire rows to pass to workers
        fire_groups = df_flat.groupby('fireID')
        
        # Group fire pixels into single events
        fire_events = df_flat.groupby('fireID').agg({
            't': ['min', 'max'],
            'hull': combine_geoms
        }).reset_index()
        fire_events.columns = ['fireID', 't_start', 't_end', 'geometry']
        
        # Convert back to GeoDataFrame
        valid_fires = gpd.GeoDataFrame(fire_events, geometry='geometry', crs=df.crs)
        valid_fires = valid_fires[valid_fires['t_start'] >= pd.to_datetime(START_DATE)].copy()

        # --- FILTER OVERSIZED FIRES ---
        print("Filtering oversized fires...", end=" ")
        bounds = valid_fires.geometry.bounds
        valid_fires['width_m'] = bounds['maxx'] - bounds['minx']
        valid_fires['height_m'] = bounds['maxy'] - bounds['miny']
        
        # Drop fires larger than 100km in any dimension [cite: 84]
        initial_count = len(valid_fires)
        valid_fires = valid_fires[
            (valid_fires['width_m'] <= FIXED_WINDOW_SIZE_METERS) & 
            (valid_fires['height_m'] <= FIXED_WINDOW_SIZE_METERS)
        ].copy()
        
        dropped_count = initial_count - len(valid_fires)
        print(f"Done. Dropped {dropped_count} fires > 100km.")
        
        # --- FILTER SMALL FIRES ---
        print("Filtering small fires (< 4km^2)...", end=" ")
        valid_fires['area_m2'] = valid_fires.geometry.area
        # 4 km^2 = 4,000,000 m^2 [cite: 69]
        initial_count_small = len(valid_fires)
        valid_fires = valid_fires[valid_fires['area_m2'] >= 4_000_000].copy()
        dropped_small = initial_count_small - len(valid_fires)
        print(f"Done. Dropped {dropped_small} fires < 4km^2.")
        
        print(f"  Fires remaining: {len(valid_fires)}")
        
        if len(valid_fires) == 0:
            continue

        # Test Limit check
        if TEST_LIMIT:
            print(f"\n!!! TEST MODE: Processing first {TEST_LIMIT} fires !!!\n")
            valid_fires = valid_fires.head(TEST_LIMIT)

        # 4. Inspect the Weather Zarr for CRS
        print("Getting Weather CRS...", end=" ")
        weather_crs = None
        for attempt in range(5):
            try:
                with xr.open_zarr(HRRR_URL, decode_coords="all", storage_options={'ssl': False}) as ds:
                    weather_crs = ds.rio.crs
                print("Done.")
                break
            except Exception as e:
                if attempt < 4:
                    print(f"(retry {attempt+1})...", end=" ", flush=True)
                    time.sleep(2)
                else:
                    print(f"\nCRITICAL ERROR: Could not inspect Weather Zarr. \n{e}")
                    return

        # 5. Kick off parallel workers
        print(f"Starting parallel download for {year} with {N_JOBS} workers...")
        
        results = Parallel(n_jobs=N_JOBS)(
            delayed(process_fire_worker)(
                row, fire_groups.get_group(row['fireID']), HRRR_URL, RAW_WEATHER_ZARR_DIR, df.crs, weather_crs, year
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
                print(status + f" for {year}: {count}")

if __name__ == "__main__":
    main()