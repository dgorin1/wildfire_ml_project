import xarray as xr
import rioxarray
import os
from dask.diagnostics import ProgressBar

# --- CONFIG ---
# OLD: Tight fit
# CO_BOUNDS = (-109.1, 36.9, -102.0, 41.1)

# NEW: Generous Buffer (Safety first!)
CO_BOUNDS = (-110.0, 36.0, -101.0, 42.0)
OUTPUT_PATH = "data/static/colorado_dem_copernicus.tif"

def main():
    print(f"1. Connecting to Copernicus DEM Zarr...")
    
    # Open Lazy (chunks={}) to avoid downloading the whole world
    ds = xr.open_dataset(
        "https://data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr",
        engine="zarr",
        chunks={}, # Explicitly use Dask (Lazy loading) initially
        storage_options={"client_kwargs":{"trust_env":True}},
        decode_coords="all",
        mask_and_scale=True 
    )

    var_name = list(ds.data_vars)[0]
    dem_da = ds[var_name]

    # Ensure CRS is set
    if dem_da.rio.crs is None:
        dem_da.rio.write_crs("EPSG:4326", inplace=True)

    print("2. Clipping to Colorado bounds (Metadata only)...")
    subset = dem_da.rio.clip_box(
        minx=CO_BOUNDS[0],
        miny=CO_BOUNDS[1],
        maxx=CO_BOUNDS[2],
        maxy=CO_BOUNDS[3]
    )

    # --- THE OPTIMIZATION ---
    print("3. Downloading subset to RAM (The 'Firehose' Step)...")
    print("   (This might take a moment, but it's faster than writing chunk-by-chunk)")
    
    # This triggers the actual network download in parallel
    with ProgressBar():
        subset = subset.load() 
    # ------------------------

    print(f"4. Writing to disk at {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Since 'subset' is now in RAM, this write will be nearly instant
    subset.rio.to_raster(
        OUTPUT_PATH,
        compress='LZW',
        tiled=True,
        windowed=True
    )
 
    print(f"SUCCESS! Process complete.")

if __name__ == "__main__":
    main()