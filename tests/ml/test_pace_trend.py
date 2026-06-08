"""Tests for the pace_trend feature in build_lap_features."""

import pandas as pd
import pytest

from wec_analytics.ml.features import build_lap_features


def _make_session(lap_times: list[float], car: str = "1", car_class: str = "LMP2") -> pd.DataFrame:
    n = len(lap_times)
    return pd.DataFrame({
        "car_number": car,
        "car_class": car_class,
        "lap_number": range(1, n + 1),
        "lap_time": lap_times,
        "is_outlier": False,
        "is_in_lap": False,
        "is_out_lap": False,
        "is_traffic_lap": False,
    })


def test_pace_trend_column_added():
    df = build_lap_features(_make_session([90.0] * 10))
    assert "pace_trend" in df.columns


def test_first_three_laps_are_nan():
    # diff(3) requires 4 values, so laps 1-3 should be NaN
    df = build_lap_features(_make_session([90.0] * 10))
    assert df["pace_trend"].iloc[:3].isna().all()


def test_lap_five_is_not_nan():
    # rolling_pace is NaN on lap 1 (closed='left', no prior laps),
    # so diff(3) at lap 4 subtracts that NaN -- first valid value is lap 5
    df = build_lap_features(_make_session([90.0] * 10))
    assert pd.notna(df["pace_trend"].iloc[4])


def test_flat_pace_gives_zero_trend():
    # constant lap times -> rolling_pace is constant -> diff(3) = 0
    df = build_lap_features(_make_session([90.0] * 10))
    assert (df["pace_trend"].dropna() == 0.0).all()


def test_degrading_pace_gives_positive_trend():
    # lap times increase linearly -> rolling_pace rises -> pace_trend > 0
    lap_times = [90.0 + i * 0.5 for i in range(12)]
    df = build_lap_features(_make_session(lap_times))
    assert (df["pace_trend"].dropna() > 0).all()


def test_improving_pace_gives_negative_trend():
    # lap times decrease -> pace_trend < 0
    lap_times = [100.0 - i * 0.5 for i in range(12)]
    df = build_lap_features(_make_session(lap_times))
    assert (df["pace_trend"].dropna() < 0).all()


def test_pace_trend_resets_after_pit():
    # two stints: if pace_trend bleeds across stint boundary it would be wrong
    rows = []
    for lap in range(1, 11):
        rows.append({
            "car_number": "1", "car_class": "LMP2",
            "lap_number": lap, "lap_time": 90.0,
            "is_outlier": False,
            "is_in_lap": lap == 10,
            "is_out_lap": False,
            "is_traffic_lap": False,
        })
    for lap in range(11, 21):
        rows.append({
            "car_number": "1", "car_class": "LMP2",
            "lap_number": lap, "lap_time": 90.0,
            "is_outlier": False,
            "is_in_lap": False,
            "is_out_lap": lap == 11,
            "is_traffic_lap": False,
        })
    df = build_lap_features(pd.DataFrame(rows))
    # first 3 laps of second stint (stint_id==2) must be NaN
    stint2 = df[df["stint_id"] == 2].sort_values("stint_age")
    assert stint2["pace_trend"].iloc[:3].isna().all()


def test_pace_trend_independent_per_car():
    # two cars with different trends should each have correct signs
    fast = _make_session([90.0 - i * 0.5 for i in range(12)], car="A")
    slow = _make_session([90.0 + i * 0.5 for i in range(12)], car="B")
    combined = pd.concat([fast, slow], ignore_index=True)
    df = build_lap_features(combined)

    a_trend = df[df["car_number"] == "A"]["pace_trend"].dropna()
    b_trend = df[df["car_number"] == "B"]["pace_trend"].dropna()
    assert (a_trend < 0).all()
    assert (b_trend > 0).all()
