"""
Pipeline Step 1: Download Weather Data for Fire Events

Description:
    This script takes a dataframe of fire vector snapshots, aggregates them 
    into unique "Fire Events" (using the union of all hulls), and downloads 
    the corresponding HRRR weather data (Temperature, Wind U/V) for the 
    entire duration of the fire.

Input:
    - A dataframe containing fire data (expected variable: `df_flat` loaded or defined)
      * Must contain columns: 'fireID', 't', 'hull' (geometry)
    - HRRR Zarr Bucket URL

Output:
    - A directory of NetCDF (.nc) files, one per fireID.
    - Each file contains the raw weather data cropped to the fire's extent.

Usage:
    python 01_download_fire_weather.py
"""

import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.ops import unary_union
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
from pathlib import Path
import certifi  # For SSL certificate bundle

# Ensure SSL uses certifi's CA bundle (fixes CERTIFICATE_VERIFY_FAILED issues)
os.environ["SSL_CERT_FILE"] = certifi.where()

# --- CONFIGURATION ---
INPUT_PICKLE_PATH = "path/to/your/fire_dataframe.pkl"   # Placeholder path if loading from disk
OUTPUT_DIR = "data/raw_weather_nc"                      # Folder where NetCDFs will be saved
START_DATE = "2018-07-15"                               # Only process fires after this date
HRRR_URL = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr?email=optional@email.com"
N_JOBS = 8                                               # Number of parallel workers
BUFFER_METERS = 2000                                     # Spatial buffer around fire geometry


# --- HELPER FUNCTIONS ---

def combine_geoms(series):
    """
    Given a series of polygons for a single fire across multiple timestamps,
    return one combined polygon representing the union of all burned areas.
    """
    return unary_union(series)


def process_fire(row, weather_ds_url, output_folder, input_crs):
    """
    Legacy placeholder. This function is not used in the threaded pipeline below.
    It exists for completeness but is overridden by process_fire_worker().
    """
    try:
        pass
    except Exception as e:
        return f"ERROR_INIT: {str(e)}"


# Worker used for parallel processing
def process_fire_worker(row, weather_ds, output_folder, input_crs):
    """
    Given a single aggregated Fire Event (one row from fire_events):
      - Extract the fire's start/end time and union geometry
      - Reproject geometry to the HRRR CRS
      - Compute a bounding box with BUFFER_METERS padding
      - Slice the HRRR weather dataset in space and time
      - Download selected variables and save them to a NetCDF file

    Returns a string status label for logging.
    """
    try:
        fire_id = int(row['fireID'])
        t_start = row['t_start']
        t_end   = row['t_end']

        # --- 1. GEOMETRY TRANSFORMATION ---
        # Construct a GeoSeries for the unioned fire footprint
        fire_gs = gpd.GeoSeries([row['geometry']], crs=input_crs)

        # Ensure weather dataset has an assigned CRS
        if weather_ds.rio.crs is None:
            return "ERROR_NO_WEATHER_CRS"

        # Reproject fire geometry into HRRR coordinate system
        fire_proj = fire_gs.to_crs(weather_ds.rio.crs)

        # Compute bounding box (with buffer)
        min_x, min_y, max_x, max_y = fire_proj.buffer(BUFFER_METERS).total_bounds

        # --- 2. SMART SLICING ---
        # First attempt: standard ascending y-direction slice
        subset = weather_ds.sel(
            x=slice(min_x, max_x),
            y=slice(min_y, max_y),
            init_time=slice(t_start, t_end)
        )

        # If the slice returns empty results, try reversed y-direction
        if subset.sizes['y'] == 0:
            subset = weather_ds.sel(
                x=slice(min_x, max_x),
                y=slice(max_y, min_y),
                init_time=slice(t_start, t_end)
            )

        # --- 3. VALIDATION ---
        # Ensure that spatial and temporal slices are non-empty
        if subset.sizes['x'] == 0 or subset.sizes['y'] == 0:
            return "SKIPPED_OUT_OF_BOUNDS"

        if subset.sizes['init_time'] == 0:
            return "SKIPPED_NO_TIME_MATCH"

        # --- 4. DOWNLOAD & SAVE ---
        # Keep only needed weather variables
        subset = subset[['temperature_2m', 'wind_u_10m', 'wind_v_10m']]

        # Trigger actual download from remote Zarr store
        subset.load()

        # Save to NetCDF using fireID in filename
        file_name = f"fireID_{fire_id}_weather.nc"
        save_path = os.path.join(output_folder, file_name)
        subset.to_netcdf(save_path)

        return "SUCCESS"

    except Exception as e:
        return f"ERROR: {str(e)}"


# --- MAIN PIPELINE ---

def main():
    print("--- PIPELINE STEP 1: WEATHER DOWNLOAD ---")

    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Load fire snapshot data from the Parquet file
    print("Aggregating snapshots into unique Fire Events...")
    data_path = Path(".") / "data" / "feds_western_us_2019_af_postprocessed.parquet"
    df = gpd.read_parquet(data_path)
    df_flat = df.reset_index()
    print(f"Loaded Fire Data. Rows: {len(df_flat)}")

    # 3. Aggregate per-fire snapshots into a single union geometry + start/end times
    fire_events = df_flat.groupby('fireID').agg({
        't': ['min', 'max'],
        'hull': combine_geoms
    }).reset_index()

    fire_events.columns = ['fireID', 't_start', 't_end', 'geometry']

    # Filter based on START_DATE
    valid_fires = fire_events[fire_events['t_start'] >= pd.to_datetime(START_DATE)].copy()
    global_crs = df_flat.crs
    print(f"Unique fires to download: {len(valid_fires)}")

    # 4. Connect to HRRR Zarr dataset
    print(f"Connecting to HRRR Zarr: {HRRR_URL}")
    ds = xr.open_zarr(HRRR_URL, decode_coords="all")

    # 5. Parallel processing of each fire event
    print(f"Starting parallel download with {N_JOBS} workers...")

    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(process_fire_worker)(row, ds, OUTPUT_DIR, global_crs)
        for index, row in tqdm(valid_fires.iterrows(), total=len(valid_fires))
    )

    # 6. Print summary of results
    print("\n" + "="*30)
    print("DOWNLOAD COMPLETE")
    print("="*30)
    counts = Counter(results)
    for status, count in counts.items():
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()