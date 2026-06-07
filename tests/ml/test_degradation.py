"""
Tests for wec_analytics.ml.degradation.
"""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.degradation import (
    MIN_STINT_LAPS,
    compare_degradation,
    fit_degradation_curve,
)


@pytest.fixture
def monotone_stint():
    """15-lap stint with a clean linear trend at ~0.2 s/lap."""
    rng = np.random.default_rng(0)
    ages = np.arange(1, 16, dtype=float)
    return pd.DataFrame({
        "stint_age": ages,
        "lap_time": 90.0 + 0.2 * ages + rng.normal(scale=0.05, size=len(ages)),
    })


@pytest.fixture
def quadratic_stint():
    """20-lap stint with a clear U-shaped (quadratic) degradation profile."""
    rng = np.random.default_rng(1)
    ages = np.arange(1, 21, dtype=float)
    return pd.DataFrame({
        "stint_age": ages,
        "lap_time": 92.0 - 0.3 * ages + 0.04 * ages ** 2 + rng.normal(scale=0.05, size=len(ages)),
    })


@pytest.fixture
def multi_car_session():
    """Two cars, two stints each, with different degradation slopes."""
    rng = np.random.default_rng(42)
    rows = []
    flag_cols = {"is_outlier": False, "is_in_lap": False, "is_out_lap": False, "is_traffic_lap": False}

    for car_number, slope in [(8, 0.1), (50, 0.4)]:
        for stint_id in [1, 2]:
            ages = np.arange(1, 16, dtype=float)
            for age in ages:
                rows.append({
                    "car_number": car_number,
                    "car_class": "Hypercar",
                    "stint_id": stint_id,
                    "stint_age": age,
                    "lap_time": 90.0 + slope * age + rng.normal(scale=0.05),
                    **flag_cols,
                })

    return pd.DataFrame(rows)


def test_slope_matches_known(monotone_stint):
    result = fit_degradation_curve(monotone_stint)
    assert abs(result["linear_slope"] - 0.2) < 0.05


def test_r2_high_for_clean_trend(monotone_stint):
    result = fit_degradation_curve(monotone_stint)
    assert result["linear_r2"] > 0.9


def test_best_model_linear_for_linear_data(monotone_stint):
    result = fit_degradation_curve(monotone_stint)
    assert result["best_model"] == "linear"


def test_best_model_quadratic_for_curved_data(quadratic_stint):
    result = fit_degradation_curve(quadratic_stint)
    assert result["best_model"] == "quadratic"


def test_too_short_returns_empty(monotone_stint):
    short = monotone_stint.head(MIN_STINT_LAPS - 1)
    assert fit_degradation_curve(short) == {}


def test_missing_column_raises():
    df = pd.DataFrame({"stint_age": [1, 2, 3, 4, 5]})
    with pytest.raises(KeyError):
        fit_degradation_curve(df)


def test_deg_slope_matches_best_model(monotone_stint, quadratic_stint):
    lin_result = fit_degradation_curve(monotone_stint)
    assert lin_result["deg_slope"] == lin_result["linear_slope"]

    quad_result = fit_degradation_curve(quadratic_stint)
    assert quad_result["deg_slope"] == quad_result["quadratic_b"]


def test_compare_degradation_shape(multi_car_session):
    summary = compare_degradation(multi_car_session)
    assert len(summary) == 2
    expected_cols = {
        "car_number", "car_class", "n_stints",
        "deg_slope_mean", "deg_slope_median", "deg_slope_std",
        "best_r2_mean", "deg_rank",
    }
    assert expected_cols.issubset(summary.columns)


def test_deg_rank_lower_is_better(multi_car_session):
    summary = compare_degradation(multi_car_session)
    ranked = summary.set_index("car_number")
    # car 8 has slope 0.1, car 50 has slope 0.4 -- car 8 should rank 1
    assert ranked.loc[8, "deg_rank"] < ranked.loc[50, "deg_rank"]


def test_compare_degradation_empty_when_no_qualifying_stints():
    df = pd.DataFrame({
        "car_number": [1] * 3,
        "car_class": ["Hypercar"] * 3,
        "stint_id": [1] * 3,
        "stint_age": [1, 2, 3],
        "lap_time": [90.0, 90.5, 91.0],
    })
    result = compare_degradation(df)
    assert result.empty
