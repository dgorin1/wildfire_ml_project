import os
import glob
import xarray as xr
import pandas as pd

def main():
    # Path to the 2018 weather Zarr files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data", "raw_weather_zarr", "2018")
    
    print(f"Looking for Zarr files in: {data_dir}")
    zarr_files = sorted(glob.glob(os.path.join(data_dir, "*.zarr")))
    
    if not zarr_files:
        print("No files found.")
        return
        
    print(f"Found {len(zarr_files)} files.\n")
    
    for z_path in zarr_files:
        file_name = os.path.basename(z_path)
        print(f"File: {file_name}")
        
        try:
            with xr.open_zarr(z_path, consolidated=False) as ds:
                if 'fire_mask' not in ds:
                    print("  No 'fire_mask' variable found.")
                    continue
                
                # Calculate sum over spatial dimensions (x, y)
                # fire_mask is expected to be (init_time, y, x)
                counts = ds['fire_mask'].sum(dim=['y', 'x'])
                
                # Bring to memory if dask array
                if hasattr(counts, 'compute'):
                    counts = counts.compute()
                
                # Print results for each time step
                for t, count in zip(ds.init_time.values, counts.values):
                    print(f"  {pd.to_datetime(t)}: {int(count)} pixels")
                    
        except Exception as e:
            print(f"  Error reading file: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()