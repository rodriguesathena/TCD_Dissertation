
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
    fig.suptitle(f"Average Travel Time Volatility on the {line.capitalize()} Line", fontsize=18, weight="bold")
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

    fig.suptitle(f"Average Journey Duration on the Luas Red and Green Lines\nby Direction and {x_label}",
        fontsize=16,
        weight="bold")

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
        None. Saves a figure to the specified path if provided.
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

    fig.suptitle(f"Average Peak Period Headways on the {line.title()} Line", fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f" Saved: {save_path}")
    plt.close()
