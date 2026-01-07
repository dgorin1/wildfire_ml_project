import xarray as xr
import os
import sys
import glob

def main():
    # 1. Determine which file to open
    if len(sys.argv) > 1:
        zarr_path = sys.argv[1]
    else:
        # Try to find the specific requested file
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_weather_zarr")
        pattern = os.path.join(base_dir, "*", "fireID_58552_weather_2018.zarr")
        found_files = glob.glob(pattern)
        
        if found_files:
            zarr_path = found_files[0]
        else:
            print(f"Target file fireID_58552_weather_2018.zarr not found in {base_dir}")
            return

    if not os.path.exists(zarr_path):
        print(f"File path does not exist: {zarr_path}")
        return

    print(f"Opening: {zarr_path}")

    # 2. Open the dataset
    # consolidated=False is often needed for these specific HRRR downloads
    ds = xr.open_zarr(zarr_path, consolidated=False)

    print("\n--- Dataset Summary ---")
    print(ds)
    print("\nBreakpoint triggered. Inspect 'ds' now.")
    
    breakpoint()

if __name__ == "__main__":
    main()