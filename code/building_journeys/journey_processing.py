import pandas as pd
import pandas as pd
import numpy as np
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add utils folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from all_functions import (
     stitch_forecasts_by_station, 
     build_clean_tram_journeys, 
     stop_templates)

# --- CONFIG ---
#precovid
input_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\raw_phases\pre_covid.csv"
output_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\precovid_journeys.csv"

#lockdown
# input_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\raw_phases\covid_lockdown.csv"
# output_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\lockdown_journeys.csv"

#recovery
# input_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\raw_phases\covid_recovery.csv"
# output_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\recovery_journeys.csv"

#postcovid
# input_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\raw_phases\post_covid.csv"
# output_file = r"C:\Users\athen\Documents\GitHub\TCD_Dissertation\data\journeys\postcovid_journeys.csv"

max_workers = 3

# --- WRAPPER FUNCTION ---
def build_clean_tram_journeys_for_date(df_day, stop_templates, date):
    try:
        result = build_clean_tram_journeys(
            df_day,
            stop_templates=stop_templates,
            max_gap_min=6,
            allow_missing=1,
            min_stops=6,
            verbose=False
        )
        return result
    except Exception as e:
        print(f"Error on {date}: {e}")
        return None

# --- MAIN PIPELINE ---
def run_pipeline():
    start_time = time.time()
    print("Reading input file...")
    df = pd.read_csv(input_file, parse_dates=["DateTime", "ServiceDay"])
    print(f"Loaded {len(df):,} rows")

    print("\nStitching station journeys (sequential)...")
    stitched_df = stitch_forecasts_by_station(df)
    print(f"Stitching complete: {stitched_df['StationJourneyID'].nunique()} journeys stitched")

    print("\nSplitting by ServiceDay...")
    date_groups = {
        date: stitched_df[stitched_df["ServiceDay"] == date].copy()
        for date in stitched_df["ServiceDay"].dropna().unique()
    }
    print(f"Split into {len(date_groups)} daily groups")

    print("\nBuilding tram journeys per day (parallel)...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(build_clean_tram_journeys_for_date, df_day, stop_templates, date): date
            for date, df_day in date_groups.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing days"):
            result = future.result()
            if result is not None and not result.empty:
                results.append(result)

    if not results:
        print("No journeys were built.")
        return

    final_df = pd.concat(results, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(final_df):,} rows and {final_df['TramJourneyID'].nunique()} journeys to {output_file}")
    print(f"Done in {time.time() - start_time:.2f} seconds.")

# --- RUN ---
if __name__ == "__main__":
    run_pipeline()
