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

    # garuntee chronological order per car before cumulative tracking
    df = df.sort_values(by=["car_number", "lap_number"]).reset_index(drop=True)

    # create unique stint identifier per car
    # add 1 so that the first stint of the session is always going to be 1
    # even if the first lap inst actaully flagged as a pit exit out-lap
    df["stint_id"] = df.groupby("car_number")["is_out_lap"].cumsum() + 1
    
    # calculate the stint age using sequential counting within the stin group
    df["stint_age"] = df.groupby(["car_number", "stint_id"]).cumcount() + 1

    # calculate robust rolling pace (median)
    # closed = 'left' ensures the rolling window strictly uses previous laps,
    # preventing target leakage for downstream predictive models
    df["rolling_pace"] = (
        df.groupby(["car_number", "stint_id"])["lap_time"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1, closed='left').median())
    )

    # reference pace and class features
    # we mask outlier lap times to NaN so they are ignored by .transform('min')
    # because the index remains intact, the clean minimum is broadcasted
    # back to ALL laps including the outliers also
    clean_lap_time = df["lap_time"].where(~df["is_outlier"])

    df["class_best_lap"] = (
        clean_lap_time
        .groupby([df["car_class"], df["lap_number"]])
        .transform("min")
)
    # if an outlier lap can see how far off the clean class pace it was
    df["class_pace_delta"] = df["lap_time"] - df["class_best_lap"]
    return df