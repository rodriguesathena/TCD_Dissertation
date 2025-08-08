# Dublin Tram Performance Analysis – MSc Dissertation

This repository contains the full codebase and documentation for my MSc dissertation at Trinity College Dublin. The project analyses operational performance on Dublin’s Luas tram system across four COVID-19 time periods: Pre-COVID, Lockdown, Recovery, and Post-COVID. The analysis focuses on three key metrics: headway regularity, travel time volatility, and journey duration.

## Repository Structure

`code/`  
Finalised Python scripts used to build journeys, compute metrics, and generate all appendix figures.

`data/`  
Processed journey-level and metrics-level CSVs, separated by phase. Due to size and licensing constraints, raw and intermediate data files are not included.

`figures/`  
All output figures for each metric (headway regularity, journey duration, travel time volatility), organised by type.

`documentation/`  
Contains the LaTeX source files for the final dissertation, including all chapter files and main compilation script.

`archive/`  
Early-stage notebooks, exploratory scripts, and legacy files used in the development of the final methodology. These are not used in the final analysis pipeline but are retained for transparency.

## Data Availability

The raw data used in this project was derived from Eoin O'Brien’s publicly available historical Luas real-time dataset, accessible at:

https://eoinobrien.ie/posts/historical-luas-real-time-data

Due to size limitations and licensing considerations, raw `.tsv` logs and large intermediate CSVs are not included in this repository. All figures are included under `figures/` for reference.

## How to Use

The repository includes two main scripts:

- `code/building_journeys/journey_processing.py`  
  Used to construct journey-level data from cleaned raw logs (already run).

- `code/metrics/metrics_processing.py`  
  Used to calculate final metrics (headways, volatility, durations) and produce all figures for the dissertation appendix.

Both scripts rely on a shared utility module located in `code/utils/all_functions.py`.

## Documentation

The full dissertation is typeset in LaTeX and located in the `documentation/` folder. To compile the paper, use the `main.tex` file along with the included chapter files.
