import yaml
import warnings
from collections import Counter
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import box
import rioxarray
import xarray as xr
import geopandas as gpd
import pandas as pd
import time
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from xrspatial import slope

# Suppress zarr v3 cosmetic warnings about consolidated metadata spec
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The data type.*does not have a Zarr V3 specification",
    category=UserWarning,
)

# Load up my configuration file
with open("pipeline/config.yaml", "r") as f:
    config = yaml.safe_load(f)

YEARS = config["YEARS"]
RAW_FIRE_DATA_DIR = config["RAW_FIRE_DATA_DIR"]
RAW_WEATHER_ZARR_DIR = config["RAW_WEATHER_ZARR_DIR"]
START_DATE = config["START_DATE"]
HRRR_URL = config["HRRR_URL"]
N_DOWNLOAD_WORKERS = config.get("N_DOWNLOAD_WORKERS", 4)
TEST_LIMIT = config["TEST_LIMIT"]
JSON_PATH = config["JSON_PATH"]

# --- Constants ---
FIXED_WINDOW_SIZE_METERS = config["FIXED_WINDOW_SIZE_METERS"]
HALF_WINDOW = FIXED_WINDOW_SIZE_METERS / 2

# --- Helper Functions ---

def combine_geoms(series):
    return unary_union(series)

def process_fire_worker(row, fire_rows, output_folder, input_crs, weather_crs, year, ds):
    MAX_RETRIES = 10
    BASE_DELAY = 5
    TARGET_RES = 1000

    fire_id = int(row['fireID'])
    year_output_folder = os.path.join(output_folder, str(year))
    os.makedirs(year_output_folder, exist_ok=True)
    save_path = os.path.join(year_output_folder, f"fireID_{fire_id}_weather_{year}.zarr")

    # Skip fires already downloaded
    if os.path.exists(save_path):
        return "SKIPPED_EXISTS"

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
            # Geometry Prep
            fire_gs = gpd.GeoSeries([row['geometry']], crs=input_crs)
            fire_proj = fire_gs.to_crs(weather_crs)

            centroid = fire_proj.geometry.iloc[0].centroid
            c_x, c_y = centroid.x, centroid.y

            min_x = c_x - HALF_WINDOW
            max_x = c_x + HALF_WINDOW
            min_y = c_y - HALF_WINDOW
            max_y = c_y + HALF_WINDOW

            subset_spatial = ds.sel(
                x=slice(min_x, max_x),
                y=slice(max_y, min_y)
            )

            subset = subset_spatial.sel(
                init_time=init_times,
                lead_time=forecast_deltas,
                method="nearest"
            )

            if subset.sizes['x'] == 0 or subset.sizes['y'] == 0:
                return "SKIPPED_OUT_OF_BOUNDS"

            if subset.sizes['init_time'] == 0:
                return "SKIPPED_NO_TIME_MATCH"

            vars_to_keep = [
                'temperature_2m', 'relative_humidity_2m', 'wind_u_10m', 'wind_v_10m',
                'precipitation_surface', 'total_cloud_cover_atmosphere',
                'downward_short_wave_radiation_flux_surface', 'wind_u_80m', 'wind_v_80m'
            ]
            available_vars = [v for v in vars_to_keep if v in subset.data_vars]
            subset = subset[available_vars]

            num_points = int(FIXED_WINDOW_SIZE_METERS / TARGET_RES)
            new_x = np.linspace(min_x, max_x, num_points, endpoint=False)
            new_y = np.linspace(max_y, min_y, num_points, endpoint=False)

            weather_1km = subset.interp(x=new_x, y=new_y, method="linear")

            transform = weather_1km.rio.transform()
            out_shape = (weather_1km.sizes['y'], weather_1km.sizes['x'])

            fire_rows_gdf = gpd.GeoDataFrame(fire_rows, geometry='hull', crs=input_crs)
            fire_rows_proj = fire_rows_gdf.to_crs(weather_crs)
            fire_rows_proj['t'] = pd.to_datetime(fire_rows_proj['t'])

            masks = []
            for t in weather_1km.init_time.values:
                ts = pd.to_datetime(t)
                valid_polys = fire_rows_proj[fire_rows_proj['t'] <= ts]

                if valid_polys.empty:
                    mask_arr = np.zeros(out_shape, dtype='uint8')
                else:
                    shapes = [(g, 1) for g in valid_polys.geometry]
                    mask_arr = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, all_touched=True, dtype='uint8')

                masks.append(mask_arr)

            weather_1km['fire_mask'] = (('init_time', 'y', 'x'), np.stack(masks))
            weather_1km.to_zarr(save_path, mode='w')

            return "SUCCESS"

        except Exception as e:
            error_msg = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (attempt + 1))
                continue
            else:
                return f"ERROR after {MAX_RETRIES} attempts: {error_msg}"

def main():
    print(f"--- PIPELINE STEP 1: WEATHER DOWNLOAD ({N_DOWNLOAD_WORKERS} workers) ---")

    for year in YEARS:
        print(f"Processing year {year}...")
        input_data = os.path.join(RAW_FIRE_DATA_DIR, f"feds_western_us_{year}_af_postprocessed.parquet")

        if not os.path.exists(input_data):
            print(f"  Skipping {year}: Input file not found.")
            continue

        df = gpd.read_parquet(input_data)
        df_flat = df.reset_index()
        df_flat['t'] = pd.to_datetime(df_flat['t'])

        fire_groups = df_flat.groupby('fireID')

        fire_events = df_flat.groupby('fireID').agg({
            't': ['min', 'max'],
            'hull': combine_geoms
        }).reset_index()
        fire_events.columns = ['fireID', 't_start', 't_end', 'geometry']

        valid_fires = gpd.GeoDataFrame(fire_events, geometry='geometry', crs=df.crs)
        valid_fires = valid_fires[valid_fires['t_start'] >= pd.to_datetime(START_DATE)].copy()

        # Filtering
        bounds = valid_fires.geometry.bounds
        valid_fires['width_m'] = bounds['maxx'] - bounds['minx']
        valid_fires['height_m'] = bounds['maxy'] - bounds['miny']
        valid_fires = valid_fires[
            (valid_fires['width_m'] <= FIXED_WINDOW_SIZE_METERS) &
            (valid_fires['height_m'] <= FIXED_WINDOW_SIZE_METERS)
        ].copy()

        valid_fires['area_m2'] = valid_fires.geometry.area
        valid_fires = valid_fires[valid_fires['area_m2'] >= 4_000_000].copy()

        if len(valid_fires) == 0:
            continue

        if TEST_LIMIT:
            valid_fires = valid_fires.head(TEST_LIMIT)

        # Get weather CRS from remote dataset
        weather_crs = None 
        for attempt in range(5):
            try:
                with xr.open_zarr(HRRR_URL, decode_coords="all", storage_options={'ssl': False}) as ds:
                    weather_crs = ds.rio.crs
                break
            except Exception:
                time.sleep(2)

        # Open the remote zarr once per year — reused for every fire
        print(f"  Opening remote HRRR zarr...")
        ds = xr.open_zarr(HRRR_URL, decode_coords="all", storage_options={'ssl': False}, chunks={})

        print(f"  Downloading {len(valid_fires)} fires (serial)...")
        results = []
        for _, row in tqdm(valid_fires.iterrows(), total=len(valid_fires), desc=f"Year {year}"):
            res = process_fire_worker(
                row,
                fire_groups.get_group(row['fireID']),
                RAW_WEATHER_ZARR_DIR,
                df.crs,
                weather_crs,
                year,
                ds,
            )
            results.append(res)

        print("\n" + "="*30)
        print(f"DOWNLOAD COMPLETE for {year}")
        print("="*30)

        counts = Counter(results)
        for status, count in counts.items():
            print(f"  {status}: {count}")

if __name__ == "__main__":
    main()
