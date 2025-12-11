import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray  # needed for .rio.crs
from geocube.api.core import make_geocube
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DATA = "data/feds_western_us_2019_af_postprocessed.parquet"
EXAMPLE_WEATHER_LOCATION = "data/raw_weather_zarr/fireID_11_weather.zarr"
RASTER_RESOLUTION = (-500, 500)  # 500 m grid
OUTPUT_DIM = "snapshot"          # concatenation dimension name

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading fire data...")
fire_df = gpd.read_parquet(INPUT_DATA)
# 't' and 'fireID' are in the index; make them regular columns
fire_df = fire_df.reset_index()
print("Fire columns:", fire_df.columns)

print("Loading example weather dataset...")
weather_ds = xr.open_zarr(EXAMPLE_WEATHER_LOCATION)
weather_crs = weather_ds.rio.crs
print("Weather CRS:", weather_crs)

# -----------------------------
# GEOMETRY SETUP
# -----------------------------
# At load time, the active geometry was 'hull'. We want to:
# 1) explicitly set it as the active geometry
# 2) rename it to 'geometry' so geocube is happy

# Make sure 'hull' is the active geometry
fires = fire_df.set_geometry("hull")

# Rename the active geometry column from 'hull' -> 'geometry'
fires = fires.rename_geometry("geometry")

print("Active geometry column:", fires.geometry.name)
print("Original Fire CRS:", fires.crs)

# Reproject fire polygons into the HRRR CRS
fires = fires.to_crs(weather_crs)
print("Reprojected Fire CRS:", fires.crs)

# If you want to test on a subset while debugging:
# fires = fires.sample(500, random_state=0).reset_index(drop=True)

# -----------------------------
# RASTERIZE EACH SNAPSHOT
# -----------------------------
rasters = []

for idx in tqdm(range(len(fires)), desc="Rasterizing fire snapshots"):
    # Take a single-row GeoDataFrame (keeps CRS & geometry)
    snapshot_gdf = fires.iloc[[idx]]  # double brackets -> GeoDataFrame

    # Rasterize fireID for that snapshot
    raster = make_geocube(
        vector_data=snapshot_gdf,
        measurements=["fireID"],      # we rasterize the fireID value
        resolution=RASTER_RESOLUTION,
        output_crs=weather_crs,
    )

    rasters.append(raster)

# -----------------------------
# COMBINE ALL RASTERS
# -----------------------------
combined = xr.concat(rasters, dim=OUTPUT_DIM)

print("Final dataset shape (fireID):", combined["fireID"].shape)
print(combined)