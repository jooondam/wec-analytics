"""
Tests for wec_analytics.ml.persistence and the predict_pace API.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from wec_analytics.ml.pace import predict_pace, predict_pace_session
from wec_analytics.ml.persistence import load_model, make_versioned_path, save_model


@pytest.fixture
def simple_pipeline(tmp_path):
    """Fit a minimal pipeline on synthetic data and return it alongside the data."""
    rng = np.random.default_rng(42)
    # synthetic feature matrix with the four columns prepare_pace_features expects
    X = pd.DataFrame({
        "stint_age": rng.integers(1, 20, size=100),
        "rolling_pace": rng.uniform(90.0, 110.0, size=100),
        "lap_number": rng.integers(1, 50, size=100),
        "car_class": rng.choice(["Hypercar", "LMP2", "LMGT3"], size=100),
    })
    y = pd.Series(rng.uniform(90.0, 110.0, size=100), name="lap_time")

    # minimal pipeline — no ColumnTransformer here since we're testing
    # persistence, not the full pace pipeline; LinearRegression on numerics only
    pipeline = Pipeline([("regressor", LinearRegression())])
    pipeline.fit(X.select_dtypes(include="number"), y)

    return pipeline, X.select_dtypes(include="number"), y


def test_save_creates_model_and_sidecar(simple_pipeline, tmp_path):
    """save_model should create both the .joblib file and the .json sidecar."""
    pipeline, _, _ = simple_pipeline
    filepath = tmp_path / "test_model.joblib"

    save_model(pipeline, filepath, metadata={"rmse_mean": 1.23})

    assert filepath.exists(), "joblib file not created"
    assert filepath.with_suffix(".json").exists(), "json sidecar not created"


def test_sidecar_contains_expected_keys(simple_pipeline, tmp_path):
    """sidecar should always contain save_timestamp and sklearn_version."""
    pipeline, _, _ = simple_pipeline
    filepath = tmp_path / "test_model.joblib"

    save_model(pipeline, filepath, metadata={"rmse_mean": 0.85})
    _, metadata = load_model(filepath)

    assert "save_timestamp" in metadata
    assert "sklearn_version" in metadata
    assert metadata["rmse_mean"] == 0.85


def test_load_produces_identical_predictions(simple_pipeline, tmp_path):
    """A loaded model must produce bit-identical predictions to the original."""
    pipeline, X, _ = simple_pipeline
    filepath = tmp_path / "test_model.joblib"

    predictions_before = pipeline.predict(X)
    save_model(pipeline, filepath)
    loaded_model, _ = load_model(filepath)
    predictions_after = loaded_model.predict(X)

    # np.testing.assert_array_equal checks for exact equality, not just
    # approximate — serialisation must not alter any coefficient values
    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
        err_msg="Loaded model predictions differ from original — serialisation error.",
    )


def test_load_raises_on_missing_file(tmp_path):
    """load_model should raise FileNotFoundError with an actionable message."""
    with pytest.raises(FileNotFoundError, match="No model file found"):
        load_model(tmp_path / "nonexistent.joblib")


def test_make_versioned_path_is_unique(tmp_path):
    """Two calls to make_versioned_path should not produce the same filename."""
    import time
    path_a = make_versioned_path("pace_linear", directory=tmp_path)
    time.sleep(1.1)  # ensure the timestamp second ticks over
    path_b = make_versioned_path("pace_linear", directory=tmp_path)
    assert path_a != path_b
