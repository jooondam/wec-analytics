"""
Pace regression model for wec_analytics ML layer.

Predicts expected lap time for a car given its current stint context.
The residual between predicted and actual pace is the quantity of
interest — underperforming the model means slower than expected,
overperforming means faster.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wec_analytics.ml.persistence import load_model


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


def predict_pace(
    model_path: str | Path,
    laps: pd.DataFrame,
) -> pd.Series:
    """
    Predict expected lap times for every row in laps using a saved model.

    Returns a Series of predicted lap times aligned to the input DataFrame's
    index, so it can be assigned back as a column: laps["predicted_pace"] = ...
    The residual (laps["lap_time"] - predicted) is usually more interesting
    than the prediction itself — negative means faster than expected.

    Parameters
    ----------
    model_path : str or Path
        Path to a joblib model file saved by save_model.
    laps : pd.DataFrame
        Feature-engineered lap DataFrame. Does not need to be filtered to
        clean laps — the caller controls what gets predicted on.

    Returns
    -------
    pd.Series
        Predicted lap times, same index as laps.
    """
    model, _ = load_model(model_path)

    # prepare_pace_features expects lap_time to exist for the y extraction,
    # but we may be predicting on laps where lap_time is unknown — add a
    # dummy column if missing so the feature selector doesn't raise
    laps = laps.copy()
    if "lap_time" not in laps.columns:
        laps["lap_time"] = float("nan")

    X, _ = prepare_pace_features(laps)
    predictions = model.predict(X)

    return pd.Series(predictions, index=laps.index, name="predicted_pace")


def predict_pace_session(
    model_path: str | Path,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict expected lap times for a full session and return an annotated DataFrame.

    Adds two columns to a copy of the input: 'predicted_pace' and 'pace_residual'.
    A positive residual means the car was slower than the model expected;
    negative means faster — the sign convention matches how a race engineer
    would think about it ("we're losing half a second to the model").

    Parameters
    ----------
    model_path : str or Path
        Path to a joblib model file saved by save_model.
    laps : pd.DataFrame
        Full session lap DataFrame from build_lap_features. All laps are
        predicted on; filtering to clean laps is the caller's decision.

    Returns
    -------
    pd.DataFrame
        Copy of laps with 'predicted_pace' and 'pace_residual' columns added.
    """
    result = laps.copy()
    result["predicted_pace"] = predict_pace(model_path, laps)

    # residual = actual - predicted:
    # positive → slower than model expected (bad)
    # negative → faster than model expected (good or suspicious)
    result["pace_residual"] = result["lap_time"] - result["predicted_pace"]

    return result