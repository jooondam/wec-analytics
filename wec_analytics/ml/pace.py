"""
Pace regression model for wec_analytics ML layer.

Predicts the deviation of each lap from the car's recent rolling pace,
then adds rolling_pace back to produce an absolute predicted lap time.

Targeting the deviation (lap_time - rolling_pace) rather than absolute
lap_time makes the model track-agnostic: a 5s positive deviation at
Le Mans and a 5s positive deviation at Spa mean the same thing to a
race engineer, even though the absolute lap times differ by 100s.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from wec_analytics.ml.features import LAP_CLEAN_FLAGS

DIRTY_FLAG_COLS = LAP_CLEAN_FLAGS


def prepare_pace_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features and target for pace regression.

    Parameters
    ----------
    laps : pd.DataFrame
        Output of build_lap_features enriched with enrich_with_deg_slope.
        Must contain columns:
        stint_age, lap_number, car_class, deg_slope, rolling_pace, lap_time.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X : DataFrame with columns [stint_age, lap_number, car_class, deg_slope]
        y : Series of (lap_time - rolling_pace), the deviation from recent pace

    Notes
    -----
    The target is lap_time minus rolling_pace so the model learns circuit-agnostic
    deviations rather than absolute lap times. Callers must add rolling_pace back
    to the model output to recover an absolute predicted lap time.

    Caller is responsible for filtering to clean racing laps before calling.
    Categorical encoding of car_class is handled by the sklearn Pipeline in
    train_pace_model via ColumnTransformer with OneHotEncoder.
    """
    required = ["stint_age", "lap_number", "car_class", "deg_slope", "rolling_pace", "lap_time"]
    missing = [col for col in required if col not in laps.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Ensure input is from build_lap_features and enrich_with_deg_slope.")

    X = laps[["stint_age", "lap_number", "car_class", "deg_slope"]].copy()
    y = (laps["lap_time"] - laps["rolling_pace"]).copy()

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
    model: Pipeline,
    laps: pd.DataFrame,
) -> pd.Series:
    """
    Predict expected lap times for every row in laps.

    Returns a Series of predicted lap times aligned to the input DataFrame's
    index, so it can be assigned back as a column: laps["predicted_pace"] = ...
    The residual (laps["lap_time"] - predicted) is usually more interesting
    than the prediction itself — negative means faster than expected.

    Parameters
    ----------
    model : Pipeline
        Fitted sklearn Pipeline returned by train_pace_model or loaded via
        load_model.
    laps : pd.DataFrame
        Feature-engineered lap DataFrame. Does not need to be filtered to
        clean laps — the caller controls what gets predicted on.

    Returns
    -------
    pd.Series
        Predicted lap times, same index as laps.
    """
    laps = laps.copy()
    if "lap_time" not in laps.columns:
        laps["lap_time"] = float("nan")

    X, _ = prepare_pace_features(laps)

    # predict only on rows where all features are present; return NaN for the
    # rest (e.g. the first lap of each stint where rolling_pace is undefined)
    valid = X.notna().all(axis=1) & laps["rolling_pace"].notna()
    result = pd.Series(float("nan"), index=laps.index, name="predicted_pace")
    if valid.any():
        # model predicts deviation from rolling_pace; add it back for absolute time
        deviation = model.predict(X.loc[valid])
        result.loc[valid] = deviation + laps.loc[valid, "rolling_pace"].to_numpy()

    return result


def predict_pace_session(
    model: Pipeline,
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
    model : Pipeline
        Fitted sklearn Pipeline returned by train_pace_model or loaded via
        load_model.
    laps : pd.DataFrame
        Full session lap DataFrame from build_lap_features. All laps are
        predicted on; filtering to clean laps is the caller's decision.

    Returns
    -------
    pd.DataFrame
        Copy of laps with 'predicted_pace' and 'pace_residual' columns added.
    """
    result = laps.copy()
    result["predicted_pace"] = predict_pace(model, laps)

    # residual = actual - predicted:
    # positive → slower than model expected (bad)
    # negative → faster than model expected (good or suspicious)
    result["pace_residual"] = result["lap_time"] - result["predicted_pace"]

    return result