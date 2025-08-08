import numpy as np
import pandas as pd
import os

### STOP TEMPLATES ###
BRI_to_BRO = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES', 'OGP', 'OUP', 'DOM', 'BRD', 'GRA', 'PHI',
              'CAB', 'BRO']
SAN_to_BRO = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN', 'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES',
              'OGP', 'OUP', 'DOM', 'BRD', 'GRA', 'PHI', 'CAB', 'BRO']
BRI_to_PAR = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES', 'OGP', 'OUP', 'PAR']
SAN_to_PAR = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN', 'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS', 'DAW', 'WES',
              'OGP', 'OUP', 'PAR']
BRO_to_BRI = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM', 'PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE',
              'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL', 'STI', 'SAN', 'CPK', 'GLE', 'GAL', 'LEO', 'BAW', 'CCK', 'LAU',
              'CHE', 'BRI']
PAR_to_BRI = ['PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE', 'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL',
              'STI', 'SAN', 'CPK', 'GLE', 'GAL', 'LEO', 'BAW', 'CCK', 'LAU', 'CHE', 'BRI']
BRO_to_SAN = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM', 'PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE',
              'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL', 'STI', 'SAN']
PAR_to_SAN = ['PAR', 'MAR', 'TRY', 'DAW', 'STS', 'HAR', 'CHA', 'RAN', 'BEE', 'COW', 'MIL', 'WIN', 'DUN', 'BAL', 'KIL',
              'STI', 'SAN']
BRI_to_STS = ['BRI', 'CHE', 'LAU', 'CCK', 'BAW', 'LEO', 'GAL', 'GLE', 'CPK', 'SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN',
              'MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS']
SAN_to_STS = ['SAN', 'STI', 'KIL', 'BAL', 'DUN', 'WIN','MIL', 'COW', 'BEE', 'RAN', 'CHA', 'HAR', 'STS']
BRO_to_DOM = ['BRO', 'CAB', 'PHI', 'GRA', 'BRD', 'DOM']
SAG_to_CON = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA',
              'FAT', 'JAM', 'MUS', 'SMI', 'FOU', 'ABB', 'BUS', 'CON']
TAL_to_CON = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA', 'FAT', 'JAM',
              'MUS', 'SMI', 'FOU', 'ABB', 'BUS', 'CON']
SAG_to_TPT = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA',
              'FAT', 'JAM', 'MUS', 'SMI', 'FOU', 'ABB', 'GDK', 'MYS', 'SDK', 'TPT']
TAL_to_TPT = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA', 'DRI', 'GOL', 'SUI', 'RIA', 'FAT', 'JAM',
              'MUS', 'SMI', 'FOU', 'ABB', 'GDK', 'MYS', 'SDK', 'TPT']
CON_to_SAG = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED', 'KIN', 'BEL', 'FET', 'CVN', 'CIT', 'FOR', 'SAG']
TPT_to_SAG = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED', 'KIN', 'BEL', 'FET', 'CVN', 'CIT', 'FOR', 'SAG']
CON_to_TAL = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED', 'KIN', 'BEL', 'COO', 'HOS', 'TAL']
TPT_to_TAL = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED', 'KIN', 'BEL', 'COO', 'HOS', 'TAL']
CON_to_HEU = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU']
CON_to_RED = ['CON', 'BUS', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI', 'BLU', 'KYL',
              'RED']
SAG_to_BEL = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL']
SAG_to_BLA = ['SAG', 'FOR', 'CIT', 'CVN', 'FET', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA']
TAL_to_BEL = ['TAL', 'HOS', 'COO', 'BEL']
TAL_to_BLA = ['TAL', 'HOS', 'COO', 'BEL', 'KIN', 'RED', 'KYL', 'BLU', 'BLA']
TPT_to_HEU = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU']
TPT_to_RED = ['TPT', 'SDK', 'MYS', 'GDK', 'ABB', 'JER', 'FOU', 'SMI', 'MUS', 'HEU', 'JAM', 'RIA', 'SUI', 'GOL', 'DRI',
              'BLU', 'KYL', 'RED']
stop_templates = {
    ('BRI', 'BRO'): BRI_to_BRO, ('SAN', 'BRO'): SAN_to_BRO, ('BRI', 'PAR'): BRI_to_PAR, ('SAN', 'PAR'): SAN_to_PAR,
    ('BRO', 'BRI'): BRO_to_BRI, ('BRO', 'SAN'): BRO_to_SAN, ('PAR', 'BRI'): PAR_to_BRI, ('PAR', 'SAN'): PAR_to_SAN,
    ('BRI', 'STS'): BRI_to_STS,  ('SAN', 'STS'): SAN_to_STS,('BRO', 'DOM'): BRO_to_DOM,
    ('SAG', 'CON'): SAG_to_CON, ('TAL', 'CON'): TAL_to_CON, ('SAG', 'TPT'): SAG_to_TPT, ('TAL', 'TPT'): TAL_to_TPT,
    ('CON', 'SAG'): CON_to_SAG, ('TPT', 'SAG'): TPT_to_SAG, ('CON', 'TAL'): CON_to_TAL, ('TPT', 'TAL'): TPT_to_TAL,
    ('CON','HEU'):  CON_to_HEU, ('CON', 'RED'): CON_to_RED, ('SAG', 'BEL'): SAG_to_BEL, ('SAG','BLA'):  SAG_to_BLA, 
    ('TAL','BEL'):  TAL_to_BEL, ('TAL', 'BLA'): TAL_to_BLA, ('TPT', 'HEU'): TPT_to_HEU, ('TPT','RED'):  TPT_to_RED}

### STATION-LEVEL FORECAST ###
import pandas as pd
from tqdm import tqdm

def stitch_forecasts_by_station(df, max_gap_minutes=8, min_logs=5):
    """
    Assigns unique StationJourneyIDs to sequences of forecast logs that appear to belong to the same journey

    Parameters:
        df: Input pandemic phase dataset containing columns like 'Origin', 'ServiceDay', 'Destination', 'DateTime', and 'Minutes'
        max_gap_minutes (int): Maximum allowed time gap (in minutes) between successive logs in the same journey
        min_logs (int): Minimum number of logs required to consider a sequence a valid journey

    Returns:
        pd.DataFrame: Modified dataframe with 'StationJourneyIDs' and clear station-level forecast logs
    """
    
    #sort inputs for chronological processing
    df = df.sort_values(['Origin', 'ServiceDay', 'Destination', 'DateTime']).reset_index(drop=True)
    df['StationJourneyID'] = pd.NA

    journey_id = 1 #unique counter for StationJourneyID
    grouped = df.groupby(['Origin', 'ServiceDay', 'Destination']) #group by route
    total_groups = len(grouped)

    print(f"Stitching {total_groups} station groups...")
    for (station, day, dest), group in tqdm(grouped, total=total_groups, desc="Stitching station journeys"):
        group = group.sort_values("DateTime")
        active_journeys = [] #active journey indices
        journey_meta = [] #descriptive info for last row of an active journey
        journey_timestamps = [] #timestamps already used in each journey
        journey_min_minutes = [] #minimum MInutes seen in each journey

        for idx, row in group.iterrows():
            assigned = False
            to_remove = []
            
            #attempt to assign current row in loop to an existing journey
            for i in range(len(active_journeys)):
                last_idx = active_journeys[i][-1]
                last_row = group.loc[last_idx]

                #calculate time gaps and changes in forecasted Minutes
                time_diff = (row['DateTime'] - last_row['DateTime']).total_seconds() / 60
                minute_diff = last_row['Minutes'] - row['Minutes'] if pd.notna(row['Minutes']) and pd.notna(last_row['Minutes']) else 0
                forecast_jump = row['Minutes'] - last_row['Minutes'] if pd.notna(row['Minutes']) and pd.notna(last_row['Minutes']) else 0
                min_seen = journey_min_minutes[i]

                #ensure timestamps are not reused in journeys
                if row['DateTime'] in journey_timestamps[i]:
                    continue
                
                #requirement to add logs to an active journey
                if ( 0 <= time_diff <= max_gap_minutes and
                    -2 <= minute_diff <= 5 and
                    forecast_jump <= 3 and
                    not (min_seen <= 1 and forecast_jump > 3)):
                    active_journeys[i].append(idx)
                    journey_meta[i] = row
                    journey_timestamps[i].add(row['DateTime'])
                    journey_min_minutes[i] = min(min_seen, row['Minutes']) if pd.notna(row['Minutes']) else min_seen
                    assigned = True
                    break
                #mark unfit journeys for removal
                elif time_diff > max_gap_minutes or (min_seen <= 1 and forecast_jump > 3):
                    to_remove.append(i)

            #finalize no longer extendable journeys
            for i in sorted(to_remove, reverse=True):
                journey = active_journeys.pop(i)
                if len(journey) >= min_logs:
                    for row_idx in journey:
                        df.at[row_idx, 'StationJourneyID'] = journey_id
                    journey_id += 1
                journey_meta.pop(i)
                journey_timestamps.pop(i)
                journey_min_minutes.pop(i)

            #if row is not assigned to an active journey, starts a new journey
            if not assigned:
                active_journeys.append([idx])
                journey_meta.append(row)
                journey_timestamps.append({row['DateTime']})
                journey_min_minutes.append(row['Minutes'] if pd.notna(row['Minutes']) else float('inf'))
        
        #final run through assigning StationJourneyID to remaining active journeys
        for journey in active_journeys:
            if len(journey) >= min_logs:
                for row_idx in journey:
                    df.at[row_idx, 'StationJourneyID'] = journey_id
                journey_id += 1

    return df


### CREATING JOURNEYS ###
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def build_clean_tram_journeys(df, stop_templates, max_gap_min=6, allow_missing=1, min_stops=6, verbose=True):
    """
    Builds full tram journeys by matching sequences of StationJourneyIDs to known stop templates.

    Parameters:
        df: Input dataframe created from stitch_forecasts_by_station function with station-level forecasts
        stop_templates: Dictionary where keys are (start, end) tuples and values are ordered stop lists for that route
        max_gap_min: Maximum allowed time gap between consecutive station arrivals within a journey
        allow_missing: Number of stops that can be missing while still considering a journey valid
        min_stops: Minimum number of matched stops required to register a tram journey
        verbose: Boolean constraint; if True, prints progress and info about built journeys.

    Returns:
        pd.DataFrame: DataFrame of stitched tram journeys with 'TramJourneyID' column; returns empty DataFrame if no journeys are found.
    """
    
    df = df.copy()
    
    #estimate actual arrival times
    df["EstimatedArrival"] = df["DateTime"] + pd.to_timedelta(df["Minutes"], unit="m")
    df["ServiceDate"] = df["DateTime"].dt.date

    #summarise and group station-level journeys
    grouped = df.groupby("StationJourneyID")
    sjid_info = []
    for sjid, group in grouped:
        dest = group["Destination"].iloc[0]
        station = group["Origin"].iloc[0]
        eta = group["EstimatedArrival"].max()
        date = group["ServiceDate"].iloc[0]
        sjid_info.append({
            "StationJourneyID": sjid,
            "Station": station,
            "Destination": dest,
            "EstimatedArrival": eta,
            "ServiceDate": date})
    sjid_df = pd.DataFrame(sjid_info)
    
    used_ids = set() #track used StationJourneyIDs
    journeys = [] #list completed tram journeys
    journey_counts = defaultdict(lambda: defaultdict(int)) # TramJourneyID counter for naming

    route_templates = list(stop_templates.items())
    print(f"Processing {len(route_templates)} route templates...")
    for (start, end), template in tqdm(route_templates, desc="Route templates"):
        for date in sjid_df["ServiceDate"].unique():
            
            #potential journeys on the selected date and destination
            candidates = sjid_df[
                (sjid_df["Destination"] == end) &
                (sjid_df["ServiceDate"] == date)]
            
            #find valid starting points for parameters above
            starters = candidates[candidates["Station"] == start].sort_values("EstimatedArrival")

            for _, start_row in starters.iterrows():
                if start_row["StationJourneyID"] in used_ids:
                    continue

                block = [start_row["StationJourneyID"]]
                current_time = start_row["EstimatedArrival"]
                missing = 0
                valid = True

                for stop in template[1:]:
                    possible = candidates[
                        (candidates["Station"] == stop) &
                        (~candidates["StationJourneyID"].isin(used_ids))]
                    possible = possible[possible["EstimatedArrival"].between(
                        current_time, current_time + pd.Timedelta(minutes=max_gap_min))].sort_values("EstimatedArrival")

                    if possible.empty: #setting limit on number of missing stops
                        missing += 1
                        if missing > allow_missing:
                            valid = False
                            break
                    else: #adding next matched stop
                        next_row = possible.iloc[0]
                        block.append(next_row["StationJourneyID"])
                        current_time = next_row["EstimatedArrival"]

                #final journey validation check
                if valid and len(block) >= min_stops:
                    journey_counts[(start, end)][date] += 1
                    jid = f"{start}_{end}{journey_counts[(start, end)][date]:02d}_{date.year}_{date.month:02d}_{date.day:02d}"
                    matched_rows = df[df["StationJourneyID"].isin(block)].copy()
                    matched_rows["TramJourneyID"] = jid
                    journeys.append(matched_rows)
                    used_ids.update(block)

                    if verbose:
                        print(f"Built journey {jid} with {len(block)} stops")

    if not journeys:
        return pd.DataFrame()

    return pd.concat(journeys, ignore_index=True)

### METRICS ###

## Processing ##
def load_and_prepare_data(journey_path, mapping_path):
    """
    Loads journey data and stop location mappings, preparing them for spatial and temporal analysis

    Parameters:
        journey_path (str): Path to CSV file containing journey-level data
        mapping_path (str): Path to GIS file containing stop geometries

    Returns:
        tuple: (DataFrame of journey data, GeoDataFrame of stops with cleaned column names)
    """
    import pandas as pd
    import geopandas as gpd
    
    #load journey data frames and parse datetime columns
    df = pd.read_csv(journey_path, parse_dates=["DateTime", "ServiceDay"])
    df["EstimatedArrival"] = pd.to_datetime(df["EstimatedArrival"], errors="coerce")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    #load GIS mappings, remove unnecessary columns, rename for consistency 
    stops_gdf = gpd.read_file(mapping_path)
    stops_gdf = stops_gdf.drop(columns=["Latitude", "Longitude"], errors="ignore")
    stops_gdf = stops_gdf.rename(columns={"StopAbbreviation": "Stop"})

    return df, stops_gdf

## Final Journey df ##
def generate_final_segments(journey_df):
    """
    Constructs journey segments from forecasted arrival data that save the final logs for each station in a journey

    Parameters:
        journey_df: DataFrame of processed station-level forecasts

    Returns:
        pd.DataFrame: Journey segments with origin, destination, travel time, and time-of-day attributes
    """
    
    df = journey_df.copy()

    #check that required columns are present
    required_columns = [
        "EstimatedArrival", "DateTime", "ServiceDay",
        "StationJourneyID", "TramJourneyID", "Origin",
        "Direction", "Line"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    #parse timestamps again for safety
    df["EstimatedArrival"] = pd.to_datetime(df["EstimatedArrival"], errors="coerce")
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df["ServiceDay"] = pd.to_datetime(df["ServiceDay"], errors="coerce")
    
    #warn if parsing failed
    if df["EstimatedArrival"].isna().sum() > 0:
        print("Warning: Some EstimatedArrival values could not be parsed into timestamps.")
    if df["DateTime"].isna().sum() > 0:
        print("Warning: Some DateTime values could not be parsed into timestamps.")

    #Step 1: get last forecasted arrival for each stop
    last_forecasts = df.sort_values("DateTime").groupby("StationJourneyID").tail(1).copy()
    last_forecasts.rename(columns={"Origin": "Stop", "EstimatedArrival": "ArrivalTime"}, inplace=True)

    #Step 2: link each stop with the next one within the same tram journey
    last_forecasts.sort_values(["TramJourneyID", "ArrivalTime"], inplace=True)
    last_forecasts["NextStop"] = last_forecasts.groupby("TramJourneyID")["Stop"].shift(-1)
    last_forecasts["NextArrival"] = last_forecasts.groupby("TramJourneyID")["ArrivalTime"].shift(-1)

    #Step 3: clean and calculate travel time between stop pairs
    journey_segments = last_forecasts.dropna(subset=["NextStop", "NextArrival"]).copy()
    journey_segments["ArrivalTime"] = pd.to_datetime(journey_segments["ArrivalTime"], errors="coerce")
    journey_segments["NextArrival"] = pd.to_datetime(journey_segments["NextArrival"], errors="coerce")
    journey_segments["TravelTimeMinutes"] = (
        (journey_segments["NextArrival"] - journey_segments["ArrivalTime"]).dt.total_seconds() / 60)

    #Step 4: contextual information
    journey_segments["hour_of_day"] = journey_segments["ArrivalTime"].dt.hour
    journey_segments["Direction"] = journey_segments["Direction"].str.strip().str.capitalize()
    journey_segments["Line"] = journey_segments["Line"].str.strip().str.capitalize()

    return journey_segments


## Volatility ##
def compute_and_filter_volatility(journey_segments):
    """
    Calculates the volatility of travel times between stop pairs and summarizes it by time period

    Parameters:
        journey_segments: DataFrame with travel time segments and metadata

    Returns:
        pd.DataFrame: Mean volatility per stop pair, direction, line, and time period
    """
    
    #subset function that assigns time periods based on hour
    def assign_period(hour):
        if 5 <= hour < 11:
            return "Morning"
        elif 11 <= hour < 17:
            return "Midday"
        elif 17 <= hour < 21:
            return "Evening"
        elif 21 <= hour <= 23:
            return "Night"
        else:
            return "Other"

    #Step 1: compute volatility at the hour level
    volatility_df = (journey_segments
        .groupby(["Stop", "NextStop", "hour_of_day", "Direction", "Line"], observed=True)
        .agg(VolatilityMinutes=("TravelTimeMinutes", "std"))
        .reset_index()
        .dropna(subset=["VolatilityMinutes"]))

    #Step 2: assign periods and compute average volatility per period
    volatility_df["Period"] = volatility_df["hour_of_day"].apply(assign_period)
    period_volatility = (volatility_df[volatility_df["Period"] != "Other"]
        .groupby(["Stop", "NextStop", "Direction", "Line", "Period"], observed=True)
        .agg(AvgVolatilityMinutes=("VolatilityMinutes", "mean"))
        .reset_index()
        .sort_values("AvgVolatilityMinutes", ascending=False))  #sort by volatility 

    return period_volatility

def plot_volatility(
    volatility_df,
    stops_gdf,
    line,
    directions=["Inbound", "Outbound"],
    interchanges=None,
    max_annotations=7,
    save_path=None):
    
    """
    Plots stop-level average volatility for each time period and direction on a map of the specified tram line

    Parameters:
        volatility_df (pd.DataFrame): Output from compute_and_filter_volatility()
        stops_gdf (GeoDataFrame): Stop geometries with 'Stop' as unique ID
        line: Tram line to plot ('Red' or 'Green')
        directions: Directions to include (['Inbound', 'Outbound'])
        interchanges: Optional list of stop names to highlight as interchanges
        max_annotations: Max number of high-volatility stops to annotate
        save_path: Optional path to save the output figure

    Returns:
        plot: 2x5 plot of Inbound and Outbound volatility across the specified line for all periods
    """
    import matplotlib.pyplot as plt
    import contextily as ctx
    import matplotlib.patheffects as pe
    import numpy as np

    interchanges = interchanges or []
    cmap = "Reds" if line.lower() == "red" else "Greens"
    periods = ["Morning", "Midday", "Evening", "Night"]

    df = volatility_df.copy()
    df["Line"] = df["Line"].str.strip().str.capitalize()
    df["Direction"] = df["Direction"].str.strip().str.replace(" to", "", regex=False).str.capitalize()
    df["Period"] = df["Period"].str.strip().str.capitalize()
    df = df[df["AvgVolatilityMinutes"] > 0]

    nrows, ncols = len(directions), len(periods)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 6 * nrows), sharex=True, sharey=True)

    #ensures 2D array of axes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    #loops through each direction and period to generate plots
    for row_idx, direction in enumerate(directions):
        for col_idx, period in enumerate(periods):
            ax = axes[row_idx][col_idx]
            subset = df[
                (df["Line"] == line.capitalize()) &
                (df["Direction"] == direction.capitalize()) &
                (df["Period"] == period)]

            if subset.empty:
                ax.set_title(f"{direction} — {period}\n(No Data)", fontsize=11)
                ax.axis("off")
                continue
            
            #aggregate volatility per stop
            stop_vol = (
                subset.groupby("Stop", observed=True)
                .agg(AvgVolatility=("AvgVolatilityMinutes", "mean"))
                .reset_index())

            merged = stops_gdf.merge(stop_vol, on="Stop", how="left").to_crs(epsg=3857)
            background = stops_gdf.to_crs(epsg=3857)

            #if no data is matched
            if merged["AvgVolatility"].isna().all():
                ax.set_title(f"{direction} — {period}\n(No Volatility Data)", fontsize=11)
                ax.axis("off")
                continue

            background.plot(ax=ax, color="#dddddd", markersize=5, alpha=0.3)

            #base map and volatiliy sized dots
            merged.plot(
                ax=ax,
                column="AvgVolatility",
                cmap=cmap,
                markersize=80,
                edgecolor="black",
                linewidth=0.5,
                legend=(row_idx == 0 and col_idx == ncols - 1),
                legend_kwds={"label": "Mean Journey Time Volatility (minutes)"},)

            if interchanges:
                merged[merged["Stop"].isin(interchanges)].plot(
                    ax=ax,
                    markersize=160,
                    color="none",
                    edgecolor="black",
                    linewidth=2,)

            #annotate top volatile stops
            annotated = merged[merged["AvgVolatility"] > 1.2]
            annotated = annotated.sort_values("AvgVolatility", ascending=False).head(max_annotations)
            for _, row in annotated.iterrows():
                ax.text(
                    row.geometry.x, row.geometry.y + 30, row["Stop"],
                    fontsize=10, ha="center", weight="bold",
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            ax.set_title(f"{direction} — {period}", fontsize=13)
            ax.axis("off")

    #titleand save
    #fig.suptitle(f"Average Travel Time Volatility on the {line.capitalize()} Line", fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.close()

## Journey Duration ##
def compute_journey_durations(journey_segments):
    """
    Aggregates stop-level travel times into total journey durations and filters outliers

    Parameters:
        journey_segments: Segmented journey data with stop-to-stop travel times

    Returns:
        pd.DataFrame: Journey-level durations with peak period and basic metadata, with outliers removed via IQR
    """
    
    #if not already done, compute TravelTimeMinutes
    if "TravelTimeMinutes" not in journey_segments.columns:
        journey_segments["TravelTimeMinutes"] = (
            journey_segments["ArrivalTime"] - journey_segments["DepartureTime"]
        ).dt.total_seconds() / 60

    #sum travel times by TramJourneyID
    journey_durations = (
        journey_segments.groupby("TramJourneyID")
        .agg({
            "TravelTimeMinutes": "sum",
            "Line": "first",
            "Direction": "first",
            "hour_of_day": "first",
            "ServiceDay": "first"}).reset_index())

    #assign peak periods
    def assign_period(hour):
        if 5 <= hour < 11:
            return "Morning"
        elif 11 <= hour < 17:
            return "Midday"
        elif 17 <= hour < 21:
            return "Evening"
        elif 21 <= hour <= 23 or 0 <= hour < 5:
            return "Night"
        else:
            return "Unknown"  

    journey_durations["Period"] = journey_durations["hour_of_day"].apply(assign_period)

    #IQR outlier filtering
    Q1 = journey_durations["TravelTimeMinutes"].quantile(0.25)
    Q3 = journey_durations["TravelTimeMinutes"].quantile(0.75)
    IQR = Q3 - Q1
    mask = (journey_durations["TravelTimeMinutes"] >= Q1 - 1.5 * IQR) & (
        journey_durations["TravelTimeMinutes"] <= Q3 + 1.5 * IQR)

    return journey_durations[mask]

def plot_durations(journey_df, by_period=True, save_path=None):
    """
    Plots average journey durations by time period or hour for both tram lines and directions

    Parameters:
        journey_df: DataFrame with journey durations and period labels
        by_period: Boolean; if True, group by Period and if False, group by hour_of_day
        save_path: Optional path to save the figure

    Returns:
        plot: 2x2 plots showing journey duration by line nad direction over period and/or hour
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)

    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    lines = ["Red", "Green"]
    directions = ["Inbound", "Outbound"]
    colors = {"Red": "#990000", "Green": "#006600"}

    group_col = "Period" if by_period else "hour_of_day"
    x_label = "Time Period" if by_period else "Hour of Day"

    df = journey_df.copy()
    df["Line"] = df["Line"].str.strip().str.capitalize()
    df["Direction"] = df["Direction"].str.replace(" to", "", regex=False).str.strip().str.title()
    
    #categorical ordering for period plots
    if by_period and "Period" in df.columns:
        ordered_periods = ["Morning", "Midday", "Evening", "Night"]
        df["Period"] = pd.Categorical(df["Period"], categories=ordered_periods, ordered=True)

    y_min = df["TravelTimeMinutes"].min()
    y_max = df["TravelTimeMinutes"].max()

    for i, line in enumerate(lines):
        for j, direction in enumerate(directions):
            ax = axes[i][j]
            subset = df[(df["Line"] == line) & (df["Direction"] == direction)]

            if subset.empty:
                ax.set_title(f"{line} Line — {direction}\n(No Data)", fontsize=11)
                ax.axis("off")
                continue

            sns.lineplot(
                data=subset,
                x=group_col,
                y="TravelTimeMinutes",
                marker="o",
                errorbar="sd",
                color=colors[line],
                ax=ax)

            ax.set_ylim(y_min, y_max)
            ax.set_title(f"{line} Line — {direction}", fontsize=13)
            ax.set_xlabel(x_label)
            if j == 0:
                ax.set_ylabel("Mean Duration (minutes)")
            else:
                ax.set_ylabel("")

    #fig.suptitle(f"Average Journey Duration on the Luas Red and Green Lines\nby Direction and {x_label}",fontsize=16,weight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.close()

## Headways ##

import pandas as pd
import numpy as np

def compute_avg_peak_headways(journey_segments):
    """
    Calculates average headways at each stop during peak periods by line and direction

    Parameters:
        journey_segments: Stop-to-stop journey segments with arrival times

    Returns:
        pd.DataFrame: Mean headway in minutes per stop and direction during peaks
    """
    
    df = journey_segments.copy()

    #clean Direction field
    df["Direction"] = df["Direction"].str.strip().str.replace(" to", "", regex=False).str.capitalize()
    df["ArrivalTime"] = pd.to_datetime(df["ArrivalTime"])

    #compute decimal hour
    df["hour"] = df["ArrivalTime"].dt.hour + df["ArrivalTime"].dt.minute / 60

    #define peak periods
    df["PeakPeriod"] = pd.cut(
        df["hour"],
        bins=[-1, 6, 10, 15, 19, 24],
        labels=[np.nan, "Morning", np.nan, "Evening", np.nan],
        ordered=False,
        include_lowest=True)

    df = df[df["PeakPeriod"].notna()].copy()

    #sort for time difference
    df = df.sort_values(["Line", "Direction", "Stop", "ArrivalTime"])

    #compute headways
    df["HeadwayMinutes"] = (
        df.groupby(["Line", "Direction", "Stop", "PeakPeriod"], observed=True)["ArrivalTime"]
        .diff()
        .dt.total_seconds() / 60)

    df = df.dropna(subset=["HeadwayMinutes"])

    #IQR filtering: remove extreme headways
    Q1 = df["HeadwayMinutes"].quantile(0.25)
    Q3 = df["HeadwayMinutes"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df["HeadwayMinutes"].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]

    #compute average headway per Stop / Direction / PeakPeriod
    summary = (
        df.groupby(["Line", "Direction", "Stop", "PeakPeriod"], observed=True)["HeadwayMinutes"]
        .mean()
        .reset_index(name="AvgHeadwayMinutes"))

    return summary

def plot_headways(summary_df, stops_gdf, line, peak_periods=["Morning", "Evening"], annotate_top_n=3, save_path=None):
    """
    Plots stop-level average headways during peak periods on a map for a given line.

    Parameters:
        summary_df (pd.DataFrame): Output from compute_avg_peak_headways()
        stops_gdf (GeoDataFrame): GIS stop point data
        line (str): Line to visualize ('Red' or 'Green')
        peak_periods (list): List of peak periods to include (["Morning", "Evening"])
        annotate_top_n (int): Number of most delayed stops to label
        save_path (str): Optional path to save the figure

    Returns:
        plot: 2x2 plot showing morning and evening headways for the chosen line in both directions
    """
    import matplotlib.pyplot as plt
    import contextily as ctx
    import matplotlib.patheffects as pe
    import numpy as np

    directions = ["Inbound", "Outbound"]
    cmap = "Greens" if line.lower() == "green" else "Reds"

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), squeeze=False, sharex=True, sharey=True)

    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for row_idx, period in enumerate(peak_periods):
        for col_idx, direction in enumerate(directions):
            ax = axes[row_idx][col_idx]

            subset = summary_df[
                (summary_df["Line"].str.capitalize() == line.capitalize()) &
                (summary_df["Direction"] == direction) &
                (summary_df["PeakPeriod"] == period)]

            if subset.empty:
                ax.set_title(f"{line} Line — {direction}, {period} Peak\n(No Data)", fontsize=11)
                ax.axis("off")
                continue

            merged = stops_gdf.merge(subset, on="Stop", how="inner").to_crs(epsg=3857)
            background = stops_gdf.to_crs(epsg=3857)
            background.plot(ax=ax, color="#dddddd", markersize=5, alpha=0.3)

            merged.plot(
                ax=ax,
                column="AvgHeadwayMinutes",
                cmap=cmap,
                markersize=merged["AvgHeadwayMinutes"] * 5,
                legend=(row_idx == 0 and col_idx == 1),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.9,
                legend_kwds={"label": "Mean Headway (minutes)"})

            # Annotate low-performing stops
            if annotate_top_n > 0:
                annotated = merged.nlargest(annotate_top_n, "AvgHeadwayMinutes")
                for _, row in annotated.iterrows():
                    ax.text(
                        row.geometry.x,
                        row.geometry.y + 20,
                        row["Stop"],
                        fontsize=9,
                        weight="bold",
                        ha="center",
                        color="black",
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
                    )

            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            ax.set_title(f"{line.title()} Line — {direction}, {period} Peak", fontsize=13)
            ax.set_axis_off()
            ax.set_aspect("equal")

    #fig.suptitle(f"Average Peak Period Headways on the {line.title()} Line", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
