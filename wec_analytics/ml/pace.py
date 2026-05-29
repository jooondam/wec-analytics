"""
Pace regression model for wec_analytics ML layer.

Predicts expected lap time for a car given its current stint context.
The residual between predicted and actual pace is the quantity of
interest — underperforming the model means slower than expected,
overperforming means faster.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


DIRTY_FLAG_COLS = ["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]


def prepare_pace_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target for pace regression.

    Parameters
    ----------
    laps : pd.DataFrame
        Output of build_lap_features. Must contain columns:
        stint_age, rolling_pace, lap_number, car_class, lap_time.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X : DataFrame with columns [stint_age, rolling_pace, lap_number, car_class]
        y : Series of lap_time

    Notes
    -----
    Caller is responsible for filtering to clean racing laps (is_outlier=False,
    is_in_lap=False, is_out_lap=False, is_traffic_lap=False) before calling.
    Categorical encoding of car_class should be handled by an sklearn pipeline
    (e.g., ColumnTransformer with OneHotEncoder) outside this function.
    """
    required = ["stint_age", "rolling_pace", "lap_number", "car_class", "lap_time"]
    missing = [col for col in required if col not in laps.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Ensure input is from build_lap_features.")

    X = laps[["stint_age", "rolling_pace", "lap_number", "car_class"]].copy()
    y = laps["lap_time"].copy()

    return X, y


def train_pace_model(laps: pd.DataFrame) -> Pipeline:
    """Train a pace regression model on clean lap data.

    Parameters
    ----------
    laps : pd.DataFrame
        Lap-level DataFrame, typically filtered to clean racing laps
        (is_outlier=False, is_in_lap=False, is_out_lap=False, is_traffic_lap=False).
        Must contain columns required by prepare_pace_features.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline with ColumnTransformer and LinearRegression.
        The pipeline can be used to predict expected lap times on new data.
    """
    # check for dirty laps and report all offending columns in one error
    violators = {}
    for col in DIRTY_FLAG_COLS:
        if col in laps.columns:
            n = laps[col].sum()
            if n > 0:
                violators[col] = n
    if violators:
        parts = [f"{col} ({n} laps)" for col, n in violators.items()]
        offending_cols = list(violators.keys())
        raise ValueError(
            "train_pace_model requires completely clean laps, but the following columns "
            "still contain True values: " + "; ".join(parts) + ".\n"
            "Remove all outlier, in-lap, out-lap, and traffic laps before training.\n"
            f"Example: df_clean = df[(df[{offending_cols}] == False).all(axis=1)]"
        )

    # prepare feature matrix and target
    X, y = prepare_pace_features(laps)

    # encode car_class with one-hot, pass numeric columns through unchanged
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first"), ["car_class"])
        ],
        remainder="passthrough"
    )

    # build and fit pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    pipeline.fit(X, y)
    return pipeline