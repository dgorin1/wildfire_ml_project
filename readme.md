# Wildfire Prediction from Weather Data

![Status: Preliminary Research](https://img.shields.io/badge/Status-Preliminary_Research-yellow)
![Focus: ML Experiment](https://img.shields.io/badge/Focus-ML_Feasibility-blueviolet)

** Work in Progress**: This repository contains experimental code to investigate whether wildfire behavior can be accurately predicted using high-resolution weather forecasts.

---

## Project Goal

The primary objective of this project is to answer a specific research question:
**"Can we predict the future spread of a wildfire solely based on its current shape and the forecasted weather?"**

We are testing the hypothesis that local weather conditions (Wind, Temperature, etc. from HRRR models) are strong enough predictors of fire growth to train a Machine Learning model.

---

## Current Status: Data Collection

We are currently in the **Data Assembly Phase**. Before we can train any models, we need to align two massive, disparate datasets:
1.  **Fire Data:** Historical fire perimeters (VIIRS/MODIS).
2.  **Weather Data:** Historical weather forecasts (NOAA HRRR).

### What the code does right now
The current script (`00_download_fire_weather.py`) performs the initial data gathering.
* It takes a list of historical fires.
* It downloads the full history of weather forecasts for that specific area.


---

## Project Structure

* `01_download_fire_weather.py`: The main script currently in use. It connects to the NOAA weather archive and saves the relevant weather data for our target fires.
* `data/`: Stores the raw inputs and the resulting NetCDF weather files.
* `notebooks/`: Juypter notebooks exploring basic data structure and explaining thought behind modeling approach
---

##  How to Run

If you are collaborating on the data collection phase:

1.  Ensure you have the fire event dataframe (pickle file).
2.  Run the downloader:
    ```bash
    python 01_download_fire_weather.py
    ```
3.  The script will output `.nc` (NetCDF) files containing the weather history for each fire.

---

## 🔜 Next Steps

1.  **Finish Data Ingestion:** Complete the download of weather data for the target fires.
2.  **Data Pairing:** Create training samples that link a specific fire shape at Time A to the weather forecast for Time B.
3.  **Model Training:** Train a baseline ML model (likely a CNN or ConvLSTM) to see if it can learn the pattern.

---

*Note: This codebase is experimental and subject to frequent changes as we iterate on the hypothesis.*