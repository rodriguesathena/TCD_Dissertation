
### STATION-LEVEL FORECAST ###
import pandas as pd
import numpy as np
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
