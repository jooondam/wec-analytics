"""
Tests for wec_analytics.ml.anomaly.
"""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.anomaly import (
    ANOMALY_FEATURES,
    DEFAULT_CONTAMINATION,
    MIN_LAPS_TO_SCORE,
    VALID_METHODS,
    compare_with_iqr,
    detect_lap_anomalies,
)


def _make_session(
    n_cars: int = 6,
    laps_per_car: int = 30,
    n_stints: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Synthetic session with all required columns."""
    if rng is None:
        rng = np.random.default_rng(0)

    rows = []
    laps_per_stint = laps_per_car // n_stints

    for car_num in range(1, n_cars + 1):
        car_class = "Hypercar" if car_num <= 3 else "LMP2"
        base_time = 100.0 if car_class == "Hypercar" else 115.0
        lap_global = 1

        for stint in range(1, n_stints + 1):
            for age in range(1, laps_per_stint + 1):
                lap_time = base_time + age * 0.1 + rng.normal(scale=0.5)
                rows.append({
                    "car_number": car_num,
                    "car_class": car_class,
                    "stint_id": stint,
                    "stint_age": age,
                    "lap_number": lap_global,
                    "lap_time": lap_time,
                    "rolling_pace": lap_time + rng.normal(scale=0.3),
                    "class_pace_delta": rng.uniform(-1.0, 1.0),
                    "class_best_lap": base_time,
                    "is_outlier": False,
                    "is_in_lap": False,
                    "is_out_lap": False,
                    "is_traffic_lap": False,
                    "race_id": "TEST",
                })
                lap_global += 1

    return pd.DataFrame(rows)


@pytest.fixture
def session():
    return _make_session(rng=np.random.default_rng(42))


@pytest.fixture
def session_with_injected_anomaly(session):
    """Session where car 1's last 3 laps are made obviously anomalous."""
    out = session.copy()
    mask = (out["car_number"] == 1) & (out["stint_age"] >= 8)
    out.loc[mask, "class_pace_delta"] = 25.0
    return out


# ---------------------------------------------------------------------------
# Output shape and columns
# ---------------------------------------------------------------------------

def test_output_has_anomaly_score_column(session):
    out = detect_lap_anomalies(session)
    assert "anomaly_score" in out.columns


def test_output_has_is_lap_anomaly_column(session):
    out = detect_lap_anomalies(session)
    assert "is_lap_anomaly" in out.columns


def test_output_row_count_unchanged(session):
    out = detect_lap_anomalies(session)
    assert len(out) == len(session)


def test_output_preserves_original_columns(session):
    original_cols = set(session.columns)
    out = detect_lap_anomalies(session)
    assert original_cols.issubset(set(out.columns))


def test_is_lap_anomaly_is_boolean(session):
    out = detect_lap_anomalies(session)
    assert out["is_lap_anomaly"].dtype == bool


def test_anomaly_score_is_numeric(session):
    out = detect_lap_anomalies(session)
    assert pd.api.types.is_float_dtype(out["anomaly_score"])


# ---------------------------------------------------------------------------
# Both methods run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["isolation_forest", "lof"])
def test_method_runs_without_error(session, method):
    out = detect_lap_anomalies(session, method=method)
    assert not out["anomaly_score"].isna().any()


@pytest.mark.parametrize("method", ["isolation_forest", "lof"])
def test_flag_rate_near_contamination(session, method):
    contamination = 0.10
    out = detect_lap_anomalies(session, method=method, contamination=contamination)
    actual_rate = out["is_lap_anomaly"].mean()
    # rank-based flagging: rate should be exactly contamination
    assert abs(actual_rate - contamination) < 0.01


# ---------------------------------------------------------------------------
# Score ordering
# ---------------------------------------------------------------------------

def test_injected_anomalies_score_higher(session_with_injected_anomaly):
    """Laps with injected extreme class_pace_delta should score above normal."""
    out = detect_lap_anomalies(session_with_injected_anomaly, method="isolation_forest")
    anomalous_scores = out[
        (out["car_number"] == 1) & (out["stint_age"] >= 8)
    ]["anomaly_score"]
    normal_scores = out[
        (out["car_number"] != 1) & (out["stint_age"] < 8)
    ]["anomaly_score"]
    assert anomalous_scores.mean() > normal_scores.mean()


# ---------------------------------------------------------------------------
# Threshold parameter
# ---------------------------------------------------------------------------

def test_explicit_threshold_respected(session):
    out = detect_lap_anomalies(session, threshold=1e6)
    assert not out["is_lap_anomaly"].any()


def test_zero_threshold_flags_all(session):
    # score_samples outputs are always > 0 after negation for IF
    out = detect_lap_anomalies(session, threshold=0.0)
    assert out["is_lap_anomaly"].all()


# ---------------------------------------------------------------------------
# Small class guard
# ---------------------------------------------------------------------------

def test_tiny_class_gets_zero_score():
    """A class with fewer than MIN_LAPS_TO_SCORE laps gets score = 0 and no flags."""
    rng = np.random.default_rng(7)
    # one big class (Hypercar, 40 laps) and one tiny class (LMP2, 3 laps)
    big = _make_session(n_cars=4, laps_per_car=20, rng=rng)
    big["car_class"] = "Hypercar"

    tiny_rows = []
    for car in [10, 11]:
        for age in range(1, 4):
            tiny_rows.append({
                "car_number": car,
                "car_class": "LMP2",
                "stint_id": 1,
                "stint_age": age,
                "lap_number": age,
                "lap_time": 115.0,
                "rolling_pace": 115.0,
                "class_pace_delta": 0.0,
                "class_best_lap": 115.0,
                "is_outlier": False,
                "is_in_lap": False,
                "is_out_lap": False,
                "is_traffic_lap": False,
                "race_id": "TEST",
            })
    tiny = pd.DataFrame(tiny_rows)
    session = pd.concat([big, tiny], ignore_index=True)

    out = detect_lap_anomalies(session)
    lmp2_out = out[out["car_class"] == "LMP2"]
    assert (lmp2_out["anomaly_score"] == 0.0).all()
    assert not lmp2_out["is_lap_anomaly"].any()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_method_raises(session):
    with pytest.raises(ValueError, match="method"):
        detect_lap_anomalies(session, method="autoencoder")


def test_invalid_contamination_raises(session):
    with pytest.raises(ValueError, match="contamination"):
        detect_lap_anomalies(session, contamination=0.6)


def test_missing_feature_column_raises():
    df = pd.DataFrame({"car_class": ["Hypercar"] * 20, "lap_time": [100.0] * 20})
    with pytest.raises(KeyError):
        detect_lap_anomalies(df)


# ---------------------------------------------------------------------------
# compare_with_iqr
# ---------------------------------------------------------------------------

def test_compare_returns_four_rows(session):
    result = compare_with_iqr(session)
    assert len(result) == 4


def test_compare_agreement_types(session):
    result = compare_with_iqr(session)
    expected = {"both_flagged", "iqr_only", "ml_only", "neither_flagged"}
    assert set(result["agreement_type"]) == expected


def test_compare_n_laps_sums_to_total(session):
    result = compare_with_iqr(session)
    assert result["n_laps"].sum() == len(session)


def test_compare_pct_sums_to_100(session):
    result = compare_with_iqr(session)
    assert abs(result["pct_laps"].sum() - 100.0) < 0.01


def test_compare_raises_without_is_outlier(session):
    no_flag = session.drop(columns=["is_outlier"])
    with pytest.raises(KeyError, match="is_outlier"):
        compare_with_iqr(no_flag)
