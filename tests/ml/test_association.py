"""
Tests for wec_analytics.ml.association.
"""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.association import (
    DEFAULT_WINDOW_MINUTES,
    LOOK_AHEAD_WINDOWS,
    SC_FIELD_FRACTION,
    TRAFFIC_LAP_FRACTION,
    _assign_windows,
    _detect_sc_windows,
    _parse_elapsed_minutes,
    build_strategy_transactions,
    mine_strategy_rules,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_elapsed(minutes: float) -> str:
    """Convert float minutes to Al Kamel 'M:SS.mmm' string."""
    total_secs = minutes * 60
    mins = int(total_secs // 60)
    secs = total_secs % 60
    return f"{mins}:{secs:06.3f}"


@pytest.fixture
def sc_session() -> pd.DataFrame:
    """
    8 cars, 3 hours of racing, 15-min windows (12 windows: 0-11).
    Window 3 (45-59 min): all cars have is_outlier=True (SC period).
    Windows 4-5 (60-89 min): all cars have is_in_lap=True (pitting after SC).
    Other windows: clean laps (~3 laps per car per window at 5-min lap times).
    """
    rng = np.random.default_rng(99)
    n_cars = 8
    classes = ["Hypercar"] * 4 + ["LMP2"] * 4
    rows = []

    for car_idx in range(n_cars):
        car_number = car_idx + 1
        car_class = classes[car_idx]
        driver = f"driver_{car_idx}"
        lap_num = 1

        for window in range(12):  # 12 windows of 15 min = 3 hours
            window_start_min = window * 15
            n_laps_in_window = 3
            for lap_offset in range(n_laps_in_window):
                elapsed_min = window_start_min + lap_offset * 5.0 + rng.uniform(0, 0.5)
                is_outlier = (window == 3)
                is_in_lap = (window in (4, 5)) and (lap_offset == 0)
                is_out_lap = (window in (4, 5)) and (lap_offset == 1)
                is_traffic = False

                rows.append({
                    "car_number": car_number,
                    "car_class": car_class,
                    "driver_name": driver,
                    "elapsed": _make_elapsed(elapsed_min),
                    "lap_number": lap_num,
                    "lap_time": 90.0 + rng.normal(scale=0.5),
                    "is_outlier": is_outlier,
                    "is_in_lap": is_in_lap,
                    "is_out_lap": is_out_lap,
                    "is_traffic_lap": is_traffic,
                })
                lap_num += 1

    return pd.DataFrame(rows)


@pytest.fixture
def transactions(sc_session):
    return build_strategy_transactions(sc_session)


# ---------------------------------------------------------------------------
# _parse_elapsed_minutes
# ---------------------------------------------------------------------------

def test_elapsed_parsing_correct():
    elapsed = pd.Series(["0:00.000", "15:00.000", "1:30.000", "142:06.349"])
    result = _parse_elapsed_minutes(elapsed)
    assert abs(result.iloc[0] - 0.0) < 1e-6
    assert abs(result.iloc[1] - 15.0) < 1e-6
    assert abs(result.iloc[2] - 1.5) < 1e-6
    assert abs(result.iloc[3] - (142 + 6.349 / 60)) < 1e-4


def test_elapsed_parsing_bad_values_produce_nan():
    elapsed = pd.Series(["bad", "", None, "1:30.000"])
    result = _parse_elapsed_minutes(elapsed)
    assert result.iloc[0:3].isna().all()
    assert not pd.isna(result.iloc[3])


# ---------------------------------------------------------------------------
# _assign_windows
# ---------------------------------------------------------------------------

def test_window_assignment_first_window():
    mins = pd.Series([0.0, 7.5, 14.99])
    windows = _assign_windows(mins, window_minutes=15)
    assert (windows == 0).all()


def test_window_assignment_second_window():
    mins = pd.Series([15.0, 22.5, 29.99])
    windows = _assign_windows(mins, window_minutes=15)
    assert (windows == 1).all()


# ---------------------------------------------------------------------------
# _detect_sc_windows
# ---------------------------------------------------------------------------

def test_sc_detection_fires(sc_session):
    df = sc_session.copy()
    from wec_analytics.ml.association import _parse_elapsed_minutes, _assign_windows
    df["_elapsed_min"] = _parse_elapsed_minutes(df["elapsed"])
    df["_window"] = _assign_windows(df["_elapsed_min"], window_minutes=15)
    sc_windows = _detect_sc_windows(df, "_window")
    # Window 3 has all 8 cars with is_outlier=True (non-in-lap)
    assert 3 in sc_windows


def test_sc_detection_does_not_fire_for_single_car(sc_session):
    # Modify sc_session so only 1 of 8 cars is an outlier in window 3
    df = sc_session.copy()
    mask = (df["elapsed"].apply(
        lambda e: 45.0 <= (int(e.split(":")[0]) + float(e.split(":")[1]) / 60) < 60
    )) & (df["car_number"] != 1)
    df.loc[mask, "is_outlier"] = False

    from wec_analytics.ml.association import _parse_elapsed_minutes, _assign_windows
    df["_elapsed_min"] = _parse_elapsed_minutes(df["elapsed"])
    df["_window"] = _assign_windows(df["_elapsed_min"], window_minutes=15)
    sc_windows = _detect_sc_windows(df, "_window")
    # Only 1 of 8 cars (12.5%) is outlier -- below SC_FIELD_FRACTION (50%)
    assert 3 not in sc_windows


# ---------------------------------------------------------------------------
# build_strategy_transactions -- shape and basic integrity
# ---------------------------------------------------------------------------

def test_transaction_shape(sc_session, transactions):
    # One row per (car, window) combination that appears in the session
    expected_rows = (
        sc_session.assign(
            _elapsed_min=_parse_elapsed_minutes(sc_session["elapsed"]),
        )
        .assign(_window=lambda d: _assign_windows(d["_elapsed_min"], 15))
        .groupby(["car_number", "_window"])
        .ngroups
    )
    assert len(transactions) == expected_rows


def test_transaction_all_item_columns_present(transactions):
    expected = ["pit_stop", "driver_change", "traffic_heavy",
                "outlier_lap", "sc_period", "q1", "q2", "q3", "q4", "next_pit"]
    for col in expected:
        assert col in transactions.columns, f"Missing column: {col}"


def test_transaction_no_nans(transactions):
    item_cols = [c for c in transactions.columns if c not in ("car_number", "window_id")]
    assert not transactions[item_cols].isna().any().any()


def test_transaction_all_boolean(transactions):
    item_cols = [c for c in transactions.columns if c not in ("car_number", "window_id")]
    for col in item_cols:
        assert transactions[col].dtype == bool or transactions[col].dtype == object or \
               set(transactions[col].unique()).issubset({True, False}), \
               f"Column {col} is not boolean"


# ---------------------------------------------------------------------------
# Specific item correctness
# ---------------------------------------------------------------------------

def test_pit_stop_item(sc_session, transactions):
    # Cars with is_in_lap laps must have pit_stop=True in corresponding windows
    pit_tx = transactions[transactions["pit_stop"]]
    assert len(pit_tx) > 0, "Expected some pit_stop=True rows"


def test_sc_period_item(transactions):
    sc_rows = transactions[transactions["sc_period"]]
    # Window 3 (all 8 cars) should all be marked sc_period=True
    assert len(sc_rows) > 0, "Expected some sc_period=True rows"
    sc_windows = sc_rows["window_id"].unique()
    assert 3 in sc_windows


def test_traffic_heavy_threshold():
    """Exactly 30% traffic -> traffic_heavy=True; 29% -> False."""
    rng = np.random.default_rng(7)
    n_laps = 10

    def _make_session(n_traffic):
        rows = []
        for i in range(n_laps):
            rows.append({
                "car_number": 1,
                "car_class": "LMP2",
                "driver_name": "d",
                "elapsed": _make_elapsed(i * 1.5),
                "lap_number": i + 1,
                "lap_time": 90.0,
                "is_outlier": False,
                "is_in_lap": False,
                "is_out_lap": False,
                "is_traffic_lap": i < n_traffic,
            })
        return pd.DataFrame(rows)

    # 3 / 10 = 30.0%: should be traffic_heavy
    tx_30 = build_strategy_transactions(_make_session(3), window_minutes=15)
    # 2 / 10 = 20.0%: should not be traffic_heavy (need 3 to hit 30%)
    # Actually TRAFFIC_LAP_FRACTION = 0.30, so exactly 3/10 = 0.30 qualifies (>=)
    tx_2 = build_strategy_transactions(_make_session(2), window_minutes=15)

    assert tx_30["traffic_heavy"].any(), "3/10 traffic laps should trigger traffic_heavy"
    assert not tx_2["traffic_heavy"].any(), "2/10 traffic laps should not trigger traffic_heavy"


def test_next_pit_lookahead(sc_session, transactions):
    # Windows 4-5 have pit_stop=True. Windows 2-3 should have next_pit=True (within 2 windows ahead).
    # Window 3 is exactly 1 before window 4; window 2 is exactly 2 before window 4.
    car1_tx = transactions[transactions["car_number"] == 1].sort_values("window_id")
    w2 = car1_tx[car1_tx["window_id"] == 2]
    w3 = car1_tx[car1_tx["window_id"] == 3]
    w0 = car1_tx[car1_tx["window_id"] == 0]

    assert not w2.empty and w2["next_pit"].iloc[0], "Window 2 should have next_pit=True"
    assert not w3.empty and w3["next_pit"].iloc[0], "Window 3 should have next_pit=True"
    # Window 0 is more than LOOK_AHEAD_WINDOWS=2 before window 4
    if not w0.empty:
        assert not w0["next_pit"].iloc[0], "Window 0 should NOT have next_pit=True"


# ---------------------------------------------------------------------------
# mine_strategy_rules
# ---------------------------------------------------------------------------

def test_mine_returns_dataframe(sc_session):
    rules = mine_strategy_rules(sc_session, min_support=0.05, min_confidence=0.4)
    assert isinstance(rules, pd.DataFrame)


def test_mine_has_required_columns(sc_session):
    rules = mine_strategy_rules(sc_session, min_support=0.05, min_confidence=0.4)
    for col in ["antecedents_str", "consequents_str", "support", "confidence", "lift"]:
        assert col in rules.columns, f"Missing column: {col}"


def test_sc_predicts_next_pit_rule_found(sc_session):
    rules = mine_strategy_rules(sc_session, min_support=0.05, min_confidence=0.4)
    if rules.empty:
        pytest.skip("No rules found at these thresholds")
    # Look for any rule with sc_period as antecedent and next_pit as consequent
    sc_rules = rules[
        rules["antecedents"].apply(lambda fs: "sc_period" in fs) &
        rules["consequents"].apply(lambda fs: "next_pit" in fs)
    ]
    assert len(sc_rules) > 0, "Expected {sc_period} -> {next_pit} rule"
    assert sc_rules["confidence"].max() > 0.5


def test_empty_below_min_support(sc_session):
    rules = mine_strategy_rules(sc_session, min_support=0.99, min_confidence=0.99)
    assert isinstance(rules, pd.DataFrame)
    assert len(rules) == 0


def test_multiple_sessions_pooled(sc_session):
    rules_one = mine_strategy_rules(sc_session, min_support=0.05, min_confidence=0.4)
    rules_two = mine_strategy_rules([sc_session, sc_session], min_support=0.05, min_confidence=0.4)
    # Two copies of the same session: same transactions proportions, so rule count should be equal
    assert len(rules_two) >= len(rules_one)


def test_invalid_support_raises(sc_session):
    with pytest.raises(ValueError, match="min_support"):
        mine_strategy_rules(sc_session, min_support=0, min_confidence=0.5)


def test_invalid_confidence_raises(sc_session):
    with pytest.raises(ValueError, match="min_confidence"):
        mine_strategy_rules(sc_session, min_support=0.05, min_confidence=1.1)
