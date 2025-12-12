import os
import time
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.ops import unary_union
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
from config import YEARS, RAW_FIRE_DATA_DIR, RAW_WEATHER_ZARR_DIR, START_DATE, HRRR_URL, N_JOBS, BUFFER_METERS, TEST_LIMIT
import warnings

# --- HELPER FUNCTIONS ---

def combine_geoms(series):
    return unary_union(series)

def process_fire_worker(row, weather_url, output_folder, input_crs, weather_crs, year):
    
    # RETRY CONFIGURATION
    MAX_RETRIES = 30
    RETRY_DELAY = 5
    
    fire_id = int(row['fireID'])
    
    # 1. SETUP TIME DIMENSIONS
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
            ds = xr.open_zarr(weather_url, decode_coords="all", storage_options={'ssl': False})

            # Geometry Prep
            fire_gs = gpd.GeoSeries([row['geometry']], crs=input_crs)
            fire_proj = fire_gs.to_crs(weather_crs)
            min_x, min_y, max_x, max_y = fire_proj.buffer(BUFFER_METERS).total_bounds

            # ---------------------------------------------------------
            # STEP A: SPATIAL SELECTION (No 'method' argument allowed)
            # ---------------------------------------------------------
            subset_spatial = ds.sel(
                x=slice(min_x, max_x),
                y=slice(max_y, min_y)
            )

            # ---------------------------------------------------------
            # STEP B: TEMPORAL SELECTION (Uses 'method=nearest')
            # ---------------------------------------------------------
            subset = subset_spatial.sel(
                init_time=init_times,
                lead_time=forecast_deltas,
                method="nearest"
            )

            # 3. Validation
            if subset.sizes['x'] == 0 or subset.sizes['y'] == 0:
                return f"SKIPPED_OUT_OF_BOUNDS"
            
            if subset.sizes['init_time'] == 0:
                return "SKIPPED_NO_TIME_MATCH"

            # 4. Filter Vars
            vars_to_keep = [
                'temperature_2m', 'relative_humidity_2m', 'wind_u_10m', 'wind_v_10m', 
                'precipitation_surface', 'total_cloud_cover_atmosphere',
                'downward_short_wave_radiation_flux_surface', 'wind_u_80m', 'wind_v_80m'
            ]
            available_vars = [v for v in vars_to_keep if v in subset.data_vars]
            subset = subset[available_vars]
            
            # 5. Save
            year_output_folder = os.path.join(output_folder, str(year))  # Create a subfolder for each year
            os.makedirs(year_output_folder, exist_ok=True)  # Ensure the folder exists

            file_name = f"fireID_{fire_id}_weather_{year}.zarr"
            save_path = os.path.join(year_output_folder, file_name)
            
            subset.to_zarr(save_path, mode='w', consolidated=False)

            return "SUCCESS"

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return f"ERROR after {MAX_RETRIES} attempts: {error_msg}"

# --- MAIN PIPELINE ---

def main():
    print("--- PIPELINE STEP 1: WEATHER DOWNLOAD (FULL RUN) ---")
    
    # 1. Load Fire Data for all years
    print("Loading and aggregating Fire Events...")
    for year in YEARS:
        print(f"Processing year {year}...")
        input_data = os.path.join(RAW_FIRE_DATA_DIR, f"feds_western_us_{year}_af_postprocessed.parquet")
        df = gpd.read_parquet(input_data) 
        df_flat = df.reset_index()
        
        # Aggregate logic
        fire_events = df_flat.groupby('fireID').agg({
            't': ['min', 'max'],
            'hull': combine_geoms
        }).reset_index()

        fire_events.columns = ['fireID', 't_start', 't_end', 'geometry']
        valid_fires = fire_events[fire_events['t_start'] >= pd.to_datetime(START_DATE)].copy()

        # --- APPLY TEST LIMIT (IF ANY) ---
        if TEST_LIMIT:
            print(f"\n!!! TEST MODE ENABLED: Processing only first {TEST_LIMIT} fires !!!\n")
            valid_fires = valid_fires.head(TEST_LIMIT)

        print(f"Unique fires to process in {year}: {len(valid_fires)}")

        # 2. Get Weather CRS (One-time check)
        print("Getting Weather CRS...", end=" ")
        try:
            with xr.open_zarr(HRRR_URL, decode_coords="all", storage_options={'ssl': False}) as ds:
                weather_crs = ds.rio.crs
            print("Done.")

        except Exception as e:
            print(f"\nCRITICAL ERROR: Could not inspect Weather Zarr. \n{e}")
            return

        # 3. Parallel Execution
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
        counts = Counter(results)
        for status, count in counts.items():
            if "SKIPPED_OUT_OF_BOUNDS" in status:
                print(f"SKIPPED_OUT_OF_BOUNDS for {year}: {count}")
            else:
                print(f"{status} for {year}: {count}")

if __name__ == "__main__":
    main()