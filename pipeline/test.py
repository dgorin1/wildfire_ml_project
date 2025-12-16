import os
import geopandas as gpd

def load_standardized_fire_data(file_path):
    """
    Loads a fire parquet file and ensures it has a standard 'geometry' column.
    
    Args:
        file_path (str): Path to the parquet file.
        
    Returns:
        gpd.GeoDataFrame: A standardized GeoDataFrame, or None if loading fails.
    """
    try:
        # Load the raw data
        gdf = gpd.read_parquet(file_path)
        
        # FIX 1: Locate the correct geometry column.
        # Your logs show 'hull' (polygon) and 'fline' (line). 
        # We want 'hull' for area/dimension calculations.
        if 'hull' in gdf.columns:
            gdf = gdf.set_geometry('hull')
        
        # FIX 2: Rename the active geometry to 'geometry'.
        # This is crucial! It ensures that when we stack (concat) multiple files,
        # the geometry columns align perfectly into one column named 'geometry'.
        try:
            gdf = gdf.rename_geometry('geometry')
        except AttributeError:
            # Fallback for older geopandas versions
            gdf = gdf.rename(columns={gdf.geometry.name: 'geometry'}).set_geometry('geometry')

        return gdf

    except Exception as e:
        print(f"Error loading {os.path.basename(file_path)}: {e}")
        return None
    


import os
import glob
import pandas as pd
import geopandas as gpd


# Path to the raw fire data directory
FIRE_DATA_DIR = "/Users/drew/Documents/wildfire_ml_project/data/raw_fire_data"

def main():
    # 1. Find all parquet files
    files = []
    years = range(2019, 2022)
    for year in years:
        search_path = os.path.join(FIRE_DATA_DIR, f"*{year}*.parquet")
        files.extend(glob.glob(search_path))
    
    print(f"Found {len(files)} fire data files in {FIRE_DATA_DIR} for years {list(years)}")
    
    gdfs = []
    
    # 2. Load and Standardize Data
    for file_path in sorted(files):
        print(f"Loading {os.path.basename(file_path)}...")
        gdf = load_standardized_fire_data(file_path)
        if gdf is not None:
            gdfs.append(gdf)

    # 3. Merge and Analyze
    if gdfs:
        print("\n--- Merging Data ---")
        merged_df = pd.concat(gdfs, ignore_index=True)
        
        # Re-construct GeoDataFrame
        merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=gdfs[0].crs)
        print(f"Merged Dimensions: {merged_gdf.shape}")
        
        # Create Global Unique ID (Year + MergeID)
        if 't_st' in merged_gdf.columns and 'mergeid' in merged_gdf.columns:
            print("Generating Global IDs...")
            merged_gdf['t_st'] = pd.to_datetime(merged_gdf['t_st'])
            merged_gdf['year'] = merged_gdf['t_st'].dt.year
            merged_gdf['global_id'] = (
                merged_gdf['year'].astype(str) + "_" + merged_gdf['mergeid'].astype(str)
            )
            
            print(f"Total Unique Fires (Global): {merged_gdf['global_id'].nunique()}")
            
            print("\n--- Calculating Fire Dimensions ---")
            
            # Calculate bounds
            bounds = merged_gdf.bounds
            bounds['global_id'] = merged_gdf['global_id']
            
            # Group by GLOBAL ID
            fire_extents = bounds.groupby('global_id').agg({
                'minx': 'min', 
                'miny': 'min', 
                'maxx': 'max', 
                'maxy': 'max'
            })
            
            fire_extents['width'] = fire_extents['maxx'] - fire_extents['minx']
            fire_extents['height'] = fire_extents['maxy'] - fire_extents['miny']
            
            # Get Top 10 by Width
            top_10 = fire_extents.sort_values('width', ascending=False).head(10)
            
            print("\n--- Top 10 Largest Fires by Width ---")
            print(f"{'Global ID':<20} | {'Width (km)':<12} | {'Height (km)':<12}")
            print("-" * 52)
            
            for fire_id, row in top_10.iterrows():
                w_km = row['width'] / 1000
                h_km = row['height'] / 1000
                print(f"{fire_id:<20} | {w_km:<12.2f} | {h_km:<12.2f}")

if __name__ == "__main__":
    main()