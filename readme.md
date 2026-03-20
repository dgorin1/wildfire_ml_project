# Wildfire Spread Prediction

![Status: Baseline Complete](https://img.shields.io/badge/Status-Baseline_Complete-green)
![Focus: ML Experiment](https://img.shields.io/badge/Focus-ML_Feasibility-blueviolet)

A deep learning pipeline to predict wildfire spread 12 hours ahead using high-resolution weather forecasts, terrain, and fuel data. This work is loosely based on [Zou et al. (2023)](https://doi.org/10.3390/fire6080289).

---

## Research Question

**Can we predict the future spread of a wildfire based on its current perimeter, forecasted weather, terrain, and fuel conditions?**

We train a U-Net CNN on historical VIIRS fire observations paired with NOAA HRRR weather forecasts, USGS 3DEP terrain, and LANDFIRE fuel data to predict binary fire masks 12 hours into the future.

---

## Pipeline Status

| Script | Description | Status |
|--------|-------------|--------|
| `00_download_weather_data.py` | Download HRRR weather zarrs for each fire | Complete |
| `01_manage_veg_data.py` | Clip and reproject LANDFIRE fuel data to each fire | Complete |
| `02_add_terrain_data.py` | Fetch USGS 3DEP elevation, compute slope/aspect | Complete |
| `03_assemble_training_samples.py` | Build (16, 100, 100) input arrays and save as npz | Complete |
| `04_train_model.py` | Train U-Net with ComboLoss (BCE + Dice) | Complete |
| `05_evaluate_model.py` | Evaluate on held-out 2021 fires | Complete |

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
- **Output:** (1, 100, 100) binary fire spread mask at T+12h
- **Loss:** ComboLoss = 0.5 × BCE(pos_weight=500) + 0.5 × Dice
- **Training target:** `Y_spread = Y_t+12h * (1 - fire_at_t)` — newly burned pixels only
- **Split:** Train 2018–2020, Val/Test 2021 (year-based to prevent temporal leakage)

### Input Channels
`fire_mask`, `temperature_2m`, `relative_humidity_2m`, `wind_u_10m`, `wind_v_10m`, `wind_u_80m`, `wind_v_80m`, `total_cloud_cover_atmosphere`, `downward_short_wave_radiation_flux_surface`, `fuel_load`, `fuel_depth`, `moisture_extinction`, `elevation`, `slope`, `aspect_sin`, `aspect_cos`

---

## Key Design Decisions

### Training on spread only, not the full fire mask
At 1km/12h resolution, ~97% of fire at T+12h is already burning at T. A model that simply copies the existing fire mask scores ~0.97 IoU trivially — a "persistence" solution that learns nothing about spread behavior. We instead train on `Y_spread`, the pixels that are *newly* on fire at T+12h. This forces the model to learn actual fire propagation dynamics rather than exploiting persistence.

### Wind representation
Rather than converting u/v components to speed + direction (as in Zou et al.), we pass raw u and v components at both 10m and 80m levels directly. This avoids the 0°/360° angular discontinuity problem and preserves directional information in a form CNNs handle naturally. The 80m level also captures vertical wind shear relevant to crown fire behavior.

### Evaluation metric: spread_iou
Standard IoU against the full fire mask is misleading when the model is trained to predict spread only. We report `spread_iou` — IoU of predictions against `Y_spread` on samples where fire actually spread — as the primary metric.

---

## Baseline Results

Evaluated on held-out 2021 fires (19,336 samples, 3,851 with active spread):

| Metric | Value |
|--------|-------|
| spread_iou (spreading samples only) | **0.102** |
| spread_recall | 0.213 |
| spread_precision | 0.066 |
| spread_F1 | 0.050 |

**Interpretation:** On samples where fire actually spread, the model achieves 10.2% IoU between predicted and actual newly burned pixels. The model correctly identifies roughly 21% of actual spread events with 6.6% precision. The low precision reflects the extreme class imbalance (~1 in 50,000 pixels are newly burned). The positive trend across training runs confirms the model is learning real signal rather than a trivial solution.

---

## Planned Improvements

1. **FireLineAttn module** (Zou et al.) — spatial attention focused on fire perimeter pixels, where spread actually occurs
2. **Data augmentation** — 90° rotations to 4x training set size, with consistent rotation of wind channels
3. **Canopy fuel channels** — CanopyCover, StandHeight, CanopyBulkDensity, CanopyBaseHeight from LANDFIRE (currently using surface fuels only)

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires the FEDS fire event dataset (see [Orland et al., 2025](https://zenodo.org/records/15261836)).

---

*This codebase is experimental and actively developed.*
