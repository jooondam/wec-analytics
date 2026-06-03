"""
Tests for wec_analytics.ml.pit_window.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wec_analytics.ml.pit_window import (
    FORBIDDEN,
    predict_pit_curve,
    predict_pit_window,
    prepare_pit_features,
)


@pytest.fixture
def synthetic_laps():
    """Minimal lap DataFrame with clean features and a binary is_in_lap label."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "stint_age":    rng.integers(1, 30, size=n),
        "rolling_pace": rng.uniform(90.0, 110.0, size=n),
        "lap_number":   np.arange(1, n + 1),
        "car_class":    rng.choice(["Hypercar", "LMP2"], size=n),
        "race_id":      rng.choice(["SPA_2019_RACE", "FUJI_2018_RACE"], size=n),
        "is_in_lap":    rng.choice([True, False], size=n, p=[0.08, 0.92]),
    })
    return df


@pytest.fixture
def fitted_pipeline(synthetic_laps):
    """Logistic regression pipeline fitted on synthetic numeric features."""
    feature_columns = ["stint_age", "rolling_pace", "lap_number"]
    X = synthetic_laps[feature_columns]
    y = synthetic_laps["is_in_lap"].astype(bool)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500)),
    ])
    pipeline.fit(X, y)
    return pipeline, feature_columns


def test_forbidden_columns_are_rejected(synthetic_laps):
    """prepare_pit_features must raise ValueError for any column in FORBIDDEN."""
    forbidden_col = next(iter(FORBIDDEN))
    # inject the forbidden column into the DataFrame so the missing-column
    # check does not fire first -- we want to reach the leakage guard
    synthetic_laps[forbidden_col] = False
    with pytest.raises(ValueError, match="forbidden"):
        prepare_pit_features(synthetic_laps, feature_columns=["stint_age", forbidden_col])


def test_missing_column_raises(synthetic_laps):
    """prepare_pit_features must raise ValueError for columns absent from df."""
    with pytest.raises(ValueError, match="not present"):
        prepare_pit_features(synthetic_laps, feature_columns=["nonexistent_column"])


def test_predict_pit_curve_probabilities_in_unit_interval(synthetic_laps, fitted_pipeline):
    """predict_pit_curve must return values in [0, 1] for every lap."""
    pipeline, feature_columns = fitted_pipeline
    curve = predict_pit_curve(pipeline, synthetic_laps, feature_columns)
    assert (curve >= 0.0).all() and (curve <= 1.0).all(), (
        f"Probabilities outside [0, 1]: min={curve.min():.4f}, max={curve.max():.4f}"
    )


def test_predict_pit_window_returns_first_lap_above_threshold(synthetic_laps, fitted_pipeline):
    """predict_pit_window must return the lap_number of the first lap >= threshold, as int."""
    pipeline, feature_columns = fitted_pipeline
    # use threshold=0.0 so every lap qualifies -- the answer must be the first lap_number
    result = predict_pit_window(pipeline, synthetic_laps, feature_columns, threshold=0.0)
    assert result is not None
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result == int(synthetic_laps["lap_number"].iloc[0])


def test_predict_pit_window_returns_none_when_threshold_unreachable(synthetic_laps, fitted_pipeline):
    """predict_pit_window must return None when no lap exceeds the threshold."""
    pipeline, feature_columns = fitted_pipeline
    # threshold=1.1 is outside [0, 1] so no probability can ever reach it
    result = predict_pit_window(pipeline, synthetic_laps, feature_columns, threshold=1.1)
    assert result is None
