# Wildfire Spread Prediction

![Status: Model Training](https://img.shields.io/badge/Status-Model_Training-blue)
![Focus: ML Experiment](https://img.shields.io/badge/Focus-ML_Feasibility-blueviolet)

**Work in Progress**: A deep learning pipeline to predict wildfire spread 12 hours ahead using high-resolution weather forecasts, terrain, and fuel data.

---

## Research Question

**Can we predict the future spread of a wildfire based on its current perimeter, forecasted weather, terrain, and fuel conditions?**

We train a U-Net CNN on historical VIIRS fire observations paired with NOAA HRRR weather forecasts, USGS 3DEP terrain, and LANDFIRE fuel data to predict binary fire masks 12 hours into the future. This work is loosely based on [Zou et al. (2023)](https://doi.org/10.3390/fire6080289).

---

## Current Status: Model Training

The full data pipeline is complete. We have assembled ~65,900 training samples across 1,678 fires (2018–2021) and are currently training a U-Net on Apple Silicon GPU (MPS).

| Script | Description | Status |
|--------|-------------|--------|
| `00_download_weather_data.py` | Download HRRR weather zarrs for each fire | Complete |
| `01_manage_veg_data.py` | Clip and reproject LANDFIRE fuel data to each fire | Complete |
| `02_add_terrain_data.py` | Fetch USGS 3DEP elevation, compute slope/aspect | Complete |
| `03_assemble_training_samples.py` | Build (16, 100, 100) input arrays and save as npz | Complete |
| `04_train_model.py` | Train U-Net with ComboLoss (BCE + Dice) | In progress |
| `05_evaluate.py` | Evaluate on held-out 2021 fires | Pending |

---

## Data Sources

- **Fire perimeters:** VIIRS FEDS fire-tracking dataset ([Orland et al., 2025](https://zenodo.org/records/15261836))
- **Weather:** NOAA HRRR 48-hour forecast zarr via [data.dynamical.org](https://data.dynamical.org)
- **Terrain:** USGS 3DEP 30m DEM via py3dep → slope, aspect
- **Fuel:** LANDFIRE FBFM13 fuel models mapped to fuel load, fuel bed depth, moisture extinction

---

## Model

- **Architecture:** U-Net (4 encoder stages, 64 base filters, ~31M parameters)
- **Input:** (16, 100, 100) — 16 channels × 100km × 100km at 1km resolution
- **Output:** (1, 100, 100) binary fire mask at T+12h
- **Loss:** ComboLoss = 0.5 × BCE(pos_weight) + 0.5 × Dice
- **Split:** Train 2018–2020, Val/Test 2021 (year-based to prevent temporal leakage)

### Input Channels
`fire_mask`, `temperature_2m`, `relative_humidity_2m`, `wind_u_10m`, `wind_v_10m`, `wind_u_80m`, `wind_v_80m`, `total_cloud_cover_atmosphere`, `downward_short_wave_radiation_flux_surface`, `fuel_load`, `fuel_depth`, `moisture_extinction`, `elevation`, `slope`, `aspect_sin`, `aspect_cos`

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires the FEDS fire event dataset (see [Orland et al., 2025](https://zenodo.org/records/15261836)).

---

##  Next Steps

1. **Complete training run** — U-Net training in progress on MPS (Apple Silicon GPU)
2. **Evaluate on 2021 held-out fires** — recall, precision, F1, IoU
3. **Switch to Focal Tversky Loss** — α=0.75, β=0.25 as in Zou et al. to better handle class imbalance
4. **Add canopy fuel channels** — CanopyCover, StandHeight, CanopyBulkDensity, CanopyBaseHeight from LANDFIRE (currently using surface fuels only)
5. **Add fire attention modules** — FirePolyAttn / FireLineAttn as described in Zou et al.

---

*Note: This codebase is experimental and subject to frequent changes as we iterate on the hypothesis.*