"""
Feature engineering utilities for the wec_analytics ML layer.

Transforms cleaned Phase 1 lap DataFrames into feature matrices
suitable for model training and prediction. All functions follow
the non-destructive pattern established in Phase 1 -- original
columns are preserved and engineered features are appended as
new columns.

Every function returns the full DataFrame so that car, lap, and
session identifiers travel with the features for debugging and
evaluation.
"""

import pandas as pd
import numpy as np


LAP_CLEAN_FLAGS = ["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]


def build_lap_features(laps: pd.DataFrame, rolling_window: int = 5,) -> pd.DataFrame:
    """Build a per-lap feature matrix from a cleaned session DataFrame.

    Appends engineered feature columns to the input DataFrame and
    returns the full result. The original columns are never modified
    or dropped, so boolean flags from Phase 1 (is_outlier,
    is_in_lap, is_out_lap, is_traffic_lap) are available alongside
    the new features for filtering decisions downstream.

    Parameters
    ----------
    laps : pd.DataFrame
        Cleaned lap DataFrame produced by models.py. Must contain
        is_outlier, is_in_lap, is_out_lap, is_traffic_lap columns.
    rolling_window : int, optional
        Number of previous laps used to compute rolling median pace.
        Default is 5, appropriate for 6-hour WEC sessions. Increase
        for 24-hour races with longer stints.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with engineered feature columns appended.

    Raises
    ------
    KeyError
        If any required Phase 1 boolean columns are missing.
    """
    # guard clause: verify required Phase 1 boolean columns exist
    required_columns = ["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]
    missing_columns = [col for col in required_columns if col not in laps.columns]

    if missing_columns:
        raise KeyError(
            f"Missing required Phase 1 columns from input DataFrame: {missing_columns}. "
            f"Ensure the data has been processed by the cleaning layer first."
        )

    # work on a copy to preserve non-destructive behaviour
    df = laps.copy()

    # guarantee chronological order per car before cumulative tracking
    df = df.sort_values(by=["car_number", "lap_number"]).reset_index(drop=True)

    # create unique stint identifier per car
    # add 1 so that the first stint of the session is always going to be 1
    # even if the first lap isn't actually flagged as a pit exit out-lap
    df["stint_id"] = df.groupby("car_number")["is_out_lap"].cumsum() + 1

    # calculate the stint age using sequential counting within the stint group
    df["stint_age"] = df.groupby(["car_number", "stint_id"]).cumcount() + 1

    # calculate robust rolling pace (median of previous clean laps only)
    # closed='left' prevents target leakage: the window never includes the current lap.
    # we mask dirty laps (outliers, pit laps, traffic) to NaN before rolling so that
    # safety-car periods don't contaminate the rolling baseline for subsequent clean laps.
    # dirty-lap rows still receive a rolling_pace value (forward-filled from last clean window)
    # so they are not lost, but their rolling_pace reflects only clean-lap history.
    flag_cols = [c for c in ["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"] if c in df.columns]
    clean_lap_time = df["lap_time"].where(~df[flag_cols].any(axis=1)) if flag_cols else df["lap_time"]

    df["rolling_pace"] = (
        clean_lap_time.groupby([df["car_number"], df["stint_id"]])
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1, closed='left').median())
    )

    # Step D: Reference Pace & Class Delta Features
    # We mask outlier lap times to NaN so they are ignored by .transform('min').
    # Because the index remains intact, the clean minimum is broadcast back to
    # ALL laps — including outliers — so even a safety car lap can show how far
    # off the clean class pace it was. Any lap-number/class combination where
    # every car was an outlier will produce NaN, which HistGradientBoostingRegressor
    # handles natively downstream.
    clean_lap_time = df["lap_time"].where(~df["is_outlier"])

    df["class_best_lap"] = (
        clean_lap_time
        .groupby([df["car_class"], df["lap_number"]])
        .transform("min")
    )

    # even an outlier lap can see how far off the clean class pace it was
    df["class_pace_delta"] = df["lap_time"] - df["class_best_lap"]

    return df


def enrich_with_norm_stint_age(df: pd.DataFrame) -> pd.DataFrame:
    """Add norm_stint_age: stint_age divided by the car's rolling median completed stint length.

    Makes stint position circuit-agnostic. stint_age=10 means different things at
    Le Mans (11-lap stints) vs Spa (25-lap stints); norm_stint_age ~1.0 means
    "approaching typical pit window" regardless of circuit.

    Uses only COMPLETED stints prior to the current one (shift + expanding median),
    so there is no future data leakage. The first stint of each car falls back to
    the class-level median stint length computed from the session.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_lap_features. Must contain car_number, car_class,
        stint_id, stint_age.

    Returns
    -------
    pd.DataFrame
        Copy of df with norm_stint_age column appended.
    """
    required = ["car_number", "car_class", "stint_id", "stint_age"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. Run build_lap_features first.")

    out = df.copy()

    # completed length of each (car, stint) = max stint_age in that group
    stint_lengths = (
        out.groupby(["car_number", "stint_id"])["stint_age"]
        .max()
        .rename("stint_length")
        .reset_index()
        .sort_values(["car_number", "stint_id"])
    )

    # rolling median of PREVIOUS completed stints per car (shift excludes current)
    stint_lengths["rolling_median_stint"] = (
        stint_lengths
        .groupby("car_number")["stint_length"]
        .transform(lambda x: x.shift(1).expanding().median())
    )

    # first stint has no prior data: fall back to class-level median from this session
    car_class_map = out.groupby("car_number")["car_class"].first()
    stint_lengths["car_class"] = stint_lengths["car_number"].map(car_class_map)
    class_medians = stint_lengths.groupby("car_class")["stint_length"].median()
    mask = stint_lengths["rolling_median_stint"].isna()
    stint_lengths.loc[mask, "rolling_median_stint"] = (
        stint_lengths.loc[mask, "car_class"].map(class_medians)
    )
    # final fallback if class median is also NaN (single-car class with no data)
    stint_lengths["rolling_median_stint"] = stint_lengths["rolling_median_stint"].fillna(20.0)

    out = out.merge(
        stint_lengths[["car_number", "stint_id", "rolling_median_stint"]],
        on=["car_number", "stint_id"],
        how="left",
    )
    out["norm_stint_age"] = out["stint_age"] / out["rolling_median_stint"].clip(lower=1.0)
    out = out.drop(columns=["rolling_median_stint"])

    return out


def build_stint_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-lap feature matrix down to one row per stint.

    Takes the output of `build_lap_features` (which includes stint_id,
    stint_age, and all per-lap engineered columns) and groups by
    car and stint. Returns a DataFrame with one row per stint,
    suitable for clustering stint types and degradation milestone
    detection.

    Parameters
    ----------
    laps : pd.DataFrame
        DataFrame from `build_lap_features`. Must contain columns:
        car_number, car_class, stint_id, lap_number, lap_time,
        rolling_pace, class_pace_delta, is_traffic_lap, is_outlier,
        is_in_lap, is_out_lap.

    Returns
    -------
    pd.DataFrame
        Stint-level feature matrix with columns:
        - car_number, car_class, stint_id (identifiers)
        - stint_length (number of laps in stint)
        - lap_number_first (starting lap of stint)
        - lap_time_median, lap_time_min, lap_time_max, lap_time_std
        - rolling_pace_median
        - class_pace_delta_median, class_pace_delta_min, class_pace_delta_max
        - traffic_ratio, outlier_ratio, in_lap_ratio, out_lap_ratio

    Notes
    -----
    - Boolean columns are aggregated to _ratio via mean (proportion of laps).
    - For class_pace_delta, a smaller (or negative) delta means faster
      relative to class. Therefore `class_pace_delta_min` is the *best*
      competitive performance in the stint, and `_max` is the worst.
    - `lap_time_std` measures within-stint consistency; high values
      often indicate traffic, incidents, or a steep degradation curve.
    - `lap_number_first` uses .first(), not .min(), to explicitly
      capture the chronological start of the stint.
    """
    required_cols = [
        "car_number", "car_class", "stint_id", "lap_number", "lap_time",
        "rolling_pace", "class_pace_delta", "is_traffic_lap", "is_outlier",
        "is_in_lap", "is_out_lap"
    ]
    missing = [c for c in required_cols if c not in laps.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Ensure input is from build_lap_features.")

    # work on a copy to preserve non-destructive behaviour
    df = laps.copy()
    df = df.sort_values(["car_number", "stint_id", "lap_number"])

    # define aggregation dictionary
    agg_dict = {
        # identifiers (constant within group)
        "car_class": [("car_class", "first")],

        # lap number aggregations
        "lap_number": [
            ("lap_number_first", "first"),
            ("stint_length", "count"),
        ],

        # lap time multi-aggregation
        "lap_time": [
            ("lap_time_median", "median"),
            ("lap_time_min", "min"),
            ("lap_time_max", "max"),
            ("lap_time_std", "std")
        ],

        # rolling pace
        "rolling_pace": [("rolling_pace_median", "median")],

        # class pace delta (smaller = faster relative to class)
        "class_pace_delta": [
            ("class_pace_delta_median", "median"),
            ("class_pace_delta_min", "min"),
            ("class_pace_delta_max", "max")
        ],

        # boolean flags -> ratio via mean
        "is_traffic_lap": [("traffic_ratio", "mean")],
        "is_outlier": [("outlier_ratio", "mean")],
        "is_in_lap": [("in_lap_ratio", "mean")],
        "is_out_lap": [("out_lap_ratio", "mean")],
    }

    # group and aggregate
    grouped = df.groupby(["car_number", "stint_id"])
    stint_df = grouped.agg(agg_dict)

    # flatten MultiIndex columns — no special cases needed
    stint_df.columns = [col[1] if isinstance(col, tuple) else col for col in stint_df.columns]

    # reset index to bring car_number and stint_id back as columns
    stint_df = stint_df.reset_index()

    # reorder columns for clarity
    ordered_cols = [
        "car_number", "car_class", "stint_id", "stint_length", "lap_number_first",
        "lap_time_median", "lap_time_min", "lap_time_max", "lap_time_std",
        "rolling_pace_median",
        "class_pace_delta_median", "class_pace_delta_min", "class_pace_delta_max",
        "traffic_ratio", "outlier_ratio", "in_lap_ratio", "out_lap_ratio"
    ]
    stint_df = stint_df[ordered_cols]

    return stint_df