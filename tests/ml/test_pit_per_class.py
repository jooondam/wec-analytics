"""Tests for train_pit_models_per_class and predict_pit_curve_per_class."""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.pit_window import (
    predict_pit_curve_per_class,
    train_pit_models_per_class,
)

FEATURE_COLS = ["stint_age", "rolling_pace", "lap_number"]
NUMERIC_FEATURES = FEATURE_COLS


def _make_df(n_cars: int = 4, laps_per_car: int = 30, n_races: int = 3) -> pd.DataFrame:
    """Synthetic lap dataset with two car classes across multiple races."""
    rng = np.random.default_rng(0)
    rows = []
    race_ids = [f"RACE_{i}" for i in range(n_races)]
    classes = ["LMP2", "GTE_Am"]
    for race_idx, race_id in enumerate(race_ids):
        for car in range(n_cars):
            car_class = classes[car % len(classes)]
            for lap in range(1, laps_per_car + 1):
                rows.append({
                    "car_number": f"{race_idx}_{car}",
                    "car_class": car_class,
                    "race_id": race_id,
                    "lap_number": lap,
                    "stint_age": lap,
                    "rolling_pace": 90.0 + rng.normal(0, 0.3),
                    "norm_stint_age": lap / 20.0,
                    "pace_trend": rng.normal(0, 0.1),
                    "is_in_lap": lap == laps_per_car,
                    "is_out_lap": lap == 1,
                })
    return pd.DataFrame(rows)


def test_returns_dict_with_expected_keys():
    df = _make_df()
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    assert set(result.keys()) == {"models", "cv_metrics", "classes"}


def test_models_keyed_by_car_class():
    df = _make_df()
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    assert set(result["models"].keys()) == {"LMP2", "GTE_Am"}


def test_each_model_is_fitted_pipeline():
    from sklearn.pipeline import Pipeline
    df = _make_df()
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    for cls, model in result["models"].items():
        assert isinstance(model, Pipeline), f"{cls} model is not a Pipeline"
        assert hasattr(model.named_steps["classifier"], "n_iter_"), (
            f"{cls} model classifier does not appear fitted"
        )


def test_cv_metrics_present_per_class():
    df = _make_df()
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    for cls in result["models"]:
        metrics = result["cv_metrics"][cls]
        assert set(metrics.keys()) >= {"f1", "recall", "precision", "brier"}


def test_classes_list_matches_model_keys():
    df = _make_df()
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    assert sorted(result["classes"]) == sorted(result["models"].keys())


def test_single_race_class_is_skipped():
    df = _make_df(n_races=3)
    # Force one class to appear in only one race
    df = df.copy()
    mask = (df["car_class"] == "GTE_Am") & (df["race_id"] != "RACE_0")
    df = df[~mask].reset_index(drop=True)
    result = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    # GTE_Am now has only 1 race, should be skipped
    assert "GTE_Am" not in result["models"]
    assert "LMP2" in result["models"]


def test_predict_per_class_returns_series_aligned_to_index():
    df = _make_df()
    trained = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    subset = df[df["race_id"] == "RACE_0"].copy()
    proba = predict_pit_curve_per_class(trained["models"], subset, FEATURE_COLS)
    assert isinstance(proba, pd.Series)
    assert list(proba.index) == list(subset.index)


def test_predict_per_class_values_in_unit_interval():
    df = _make_df()
    trained = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    proba = predict_pit_curve_per_class(trained["models"], df, FEATURE_COLS)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_unknown_class_defaults_to_half():
    df = _make_df()
    trained = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    unknown = df[df["race_id"] == "RACE_0"].copy()
    unknown["car_class"] = "HYPERCAR"
    proba = predict_pit_curve_per_class(trained["models"], unknown, FEATURE_COLS)
    assert (proba == 0.5).all()


def test_predict_mixed_classes_routes_correctly():
    df = _make_df()
    trained = train_pit_models_per_class(df, FEATURE_COLS, NUMERIC_FEATURES)
    subset = df[df["race_id"] == "RACE_0"].copy()

    lmp2_mask = subset["car_class"] == "LMP2"
    gte_mask = subset["car_class"] == "GTE_Am"

    proba_full = predict_pit_curve_per_class(trained["models"], subset, FEATURE_COLS)
    proba_lmp2 = predict_pit_curve_per_class(trained["models"], subset[lmp2_mask], FEATURE_COLS)
    proba_gte = predict_pit_curve_per_class(trained["models"], subset[gte_mask], FEATURE_COLS)

    pd.testing.assert_series_equal(proba_full[lmp2_mask], proba_lmp2)
    pd.testing.assert_series_equal(proba_full[gte_mask], proba_gte)
