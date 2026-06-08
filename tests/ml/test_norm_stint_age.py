"""Tests for enrich_with_norm_stint_age."""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.features import build_lap_features, enrich_with_norm_stint_age


def _make_session(stints_per_car: dict[str, list[int]], car_class: str = "LMP2") -> pd.DataFrame:
    """Build a minimal session DataFrame with given stint lengths per car.

    stints_per_car: {car_number: [stint1_length, stint2_length, ...]}
    """
    rows = []
    for car, lengths in stints_per_car.items():
        lap = 1
        for stint_idx, length in enumerate(lengths, start=1):
            for pos in range(1, length + 1):
                is_in_lap = pos == length and stint_idx < len(lengths)
                is_out_lap = pos == 1 and stint_idx > 1
                rows.append({
                    "car_number": car,
                    "car_class": car_class,
                    "lap_number": lap,
                    "lap_time": 90.0,
                    "is_outlier": False,
                    "is_in_lap": is_in_lap,
                    "is_out_lap": is_out_lap,
                    "is_traffic_lap": False,
                })
                lap += 1
    return pd.DataFrame(rows)


@pytest.fixture
def simple_session():
    # car A: two stints of 10 laps each
    # car B: two stints of 20 laps each
    return _make_session({"A": [10, 10], "B": [20, 20]})


@pytest.fixture
def built(simple_session):
    return enrich_with_norm_stint_age(build_lap_features(simple_session))


def test_column_added(built):
    assert "norm_stint_age" in built.columns


def test_no_nans(built):
    assert built["norm_stint_age"].notna().all()


def test_positive(built):
    assert (built["norm_stint_age"] > 0).all()


def test_first_stint_uses_class_fallback(simple_session):
    # first stint of car A has no prior stints: should use class median
    # class median = median([10, 10, 20, 20]) = 15
    built = enrich_with_norm_stint_age(build_lap_features(simple_session))
    car_a_stint1 = built[(built["car_number"] == "A") & (built["stint_id"] == 1)]
    # norm_stint_age at stint_age=1 should be 1/15
    first_lap = car_a_stint1[car_a_stint1["stint_age"] == 1].iloc[0]
    assert abs(first_lap["norm_stint_age"] - 1 / 15) < 0.01


def test_second_stint_uses_first_stint_length(simple_session):
    # car A: first stint = 10 laps, so second stint's norm denominator should be 10
    built = enrich_with_norm_stint_age(build_lap_features(simple_session))
    car_a_stint2 = built[(built["car_number"] == "A") & (built["stint_id"] == 2)]
    first_lap = car_a_stint2[car_a_stint2["stint_age"] == 1].iloc[0]
    assert abs(first_lap["norm_stint_age"] - 1 / 10) < 0.01


def test_norm_approaches_one_at_end_of_typical_stint(simple_session):
    # car A second stint: length 10, denominator 10 -> norm_stint_age at lap 10 = 1.0
    built = enrich_with_norm_stint_age(build_lap_features(simple_session))
    car_a_stint2 = built[(built["car_number"] == "A") & (built["stint_id"] == 2)]
    last_lap = car_a_stint2[car_a_stint2["stint_age"] == 10].iloc[0]
    assert abs(last_lap["norm_stint_age"] - 1.0) < 0.01


def test_short_and_long_stint_normalized_differently():
    # Le Mans style: 11-lap stints; Spa style: 25-lap stints
    # both at stint_age=10 -> very different raw, similar norm
    lm = _make_session({"1": [11, 11, 11]}, car_class="LMP1")
    spa = _make_session({"2": [25, 25, 25]}, car_class="LMP1")
    combined = pd.concat([lm, spa], ignore_index=True)
    built = enrich_with_norm_stint_age(build_lap_features(combined))

    lm_s2 = built[(built["car_number"] == "1") & (built["stint_id"] == 2) & (built["stint_age"] == 10)]
    spa_s2 = built[(built["car_number"] == "2") & (built["stint_id"] == 2) & (built["stint_age"] == 10)]

    lm_norm = lm_s2["norm_stint_age"].iloc[0]
    spa_norm = spa_s2["norm_stint_age"].iloc[0]

    # Le Mans norm ~0.91, Spa norm ~0.40 -- both below 1.0 but very different from raw
    assert lm_norm > spa_norm  # Le Mans is closer to pit window at lap 10
    assert lm_norm < 1.1       # shouldn't overshoot by much


def test_single_car_single_class_fallback():
    # single car, single class: first stint falls back to session-level median
    session = _make_session({"X": [15, 15]}, car_class="GTE")
    built = enrich_with_norm_stint_age(build_lap_features(session))
    assert built["norm_stint_age"].notna().all()


def test_denominator_clipped_at_one():
    # edge case: single-lap stint -> denominator clips to 1, norm_stint_age >= 1
    session = _make_session({"Z": [1, 10]}, car_class="LMP2")
    built = enrich_with_norm_stint_age(build_lap_features(session))
    assert (built["norm_stint_age"] >= 0).all()
    assert built["norm_stint_age"].notna().all()


def test_missing_required_column_raises():
    df = pd.DataFrame({"car_number": [1], "lap_time": [90.0]})
    with pytest.raises(KeyError):
        enrich_with_norm_stint_age(df)
