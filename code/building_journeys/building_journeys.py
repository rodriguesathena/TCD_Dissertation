
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
