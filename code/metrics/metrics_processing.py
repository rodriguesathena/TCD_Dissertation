import pandas as pd
import numpy as np
import geopandas as gpd
import sys
import os

# Add utils folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from all_functions import (
    load_and_prepare_data,
    generate_final_segments,
    compute_and_filter_volatility,
    compute_journey_durations,
    compute_avg_peak_headways,
    plot_volatility,
    plot_durations,
    plot_headways
)

# Input files (full absolute paths)
period_configs = {
    "precovid": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\precovid_journeys.csv",
    "lockdown": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\lockdown_journeys.csv",
    "recovery": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\recovery_journeys.csv",
    "postcovid": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\postcovid_journeys.csv",
}

# Station coordinate file (full path)
mapping = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\station_coordinates\coordinates.geojson"

# Metrics output folders (full absolute paths)
metrics_output_paths = {
    "precovid": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\metrics\precovid",
    "lockdown": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\metrics\lockdown",
    "recovery": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\metrics\recovery",
    "postcovid": r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\metrics\postcovid",
}

# Figure output folders (relative paths)
fig_volatility = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\figures\travel_time_volatility"
fig_duration = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\figures\journey_duration"
fig_headways = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\figures\headway_regularity"

lines = ["Red", "Green"]
directions = ["Inbound", "Outbound"]
peak_periods = ["Morning", "Evening"]

# Run all metrics + plots
for period_name, input_file in period_configs.items():
    print(f"\nProcessing: {period_name.upper()}")

    # Use full output path for this period's metrics
    period_metrics_dir = metrics_output_paths[period_name]
    os.makedirs(period_metrics_dir, exist_ok=True)

    # Load + preprocess
    df, stops_gdf = load_and_prepare_data(input_file, mapping)
    analysisdf = generate_final_segments(df)

    # VOLATILITY
    volatilitydf = compute_and_filter_volatility(analysisdf)
    volatility_path = os.path.join(period_metrics_dir, f"volatilitydf_{period_name}.csv")
    volatilitydf.to_csv(volatility_path, index=False)
    print(f"Saved: {volatility_path}")

    for line in lines:
        for direction in directions:
            fname = f"volatility_{line.lower()}_{direction.lower()}_{period_name}.png"
            fpath = os.path.join(fig_volatility, fname)
            
            plot_volatility(volatilitydf, stops_gdf, line, [direction], max_annotations=3, save_path=fpath)

        fname = f"volatility_allperiods_{line.lower()}_combined_{period_name}.png"
        fpath = os.path.join(fig_volatility, fname)
        plot_volatility(volatilitydf, stops_gdf, line, ["Inbound", "Outbound"], max_annotations=3, save_path=fpath)

    # JOURNEY DURATIONS
    durationsdf = compute_journey_durations(analysisdf)
    durations_path = os.path.join(period_metrics_dir, f"durationsdf_{period_name}.csv")
    durationsdf.to_csv(durations_path, index=False)
    print(f"Saved: {durations_path}")

    fpath = os.path.join(fig_duration, f"duration_hourly_combined_{period_name}.png")
    plot_durations(durationsdf, by_period=False, save_path=fpath)

    fpath = os.path.join(fig_duration, f"duration_period_combined_{period_name}.png")
    plot_durations(durationsdf, by_period=True, save_path=fpath)

    # HEADWAYS
    headwaydf = compute_avg_peak_headways(analysisdf)
    headway_path = os.path.join(period_metrics_dir, f"headwaydf_{period_name}.csv")
    headwaydf.to_csv(headway_path, index=False)
    print(f"Saved: {headway_path}")

    for line in lines:
        for peak in peak_periods:
            fname = f"{peak.lower()}_headway_{line.lower()}_{period_name}.png"
            fpath = os.path.join(fig_headways, fname)
            plot_headways(headwaydf, stops_gdf, line, [peak], save_path=fpath)

        fname = f"headways_combined_{line.lower()}_{period_name}.png"
        fpath = os.path.join(fig_headways, fname)
        plot_headways(headwaydf, stops_gdf, line, save_path=fpath)

print("\nAll metrics and figures generated successfully.")
