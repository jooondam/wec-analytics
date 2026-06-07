"""
Tests for wec_analytics.ml.pace and wec_analytics.ml.degradation.enrich_with_deg_slope.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from wec_analytics.ml.degradation import enrich_with_deg_slope
from wec_analytics.ml.pace import prepare_pace_features, train_pace_model


def _make_clean_laps(n_laps: int = 40, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Synthetic clean lap DataFrame with all required columns."""
    if rng is None:
        rng = np.random.default_rng(0)

    n_cars = 4
    rows = []
    for car_num in range(1, n_cars + 1):
        for stint_id in range(1, 3):
            for stint_age in range(1, n_laps // (n_cars * 2) + 1):
                rows.append({
                    "car_number": car_num,
                    "car_class": "LMP2" if car_num <= 2 else "Hypercar",
                    "stint_id": stint_id,
                    "stint_age": stint_age,
                    "lap_number": len(rows) + 1,
                    "lap_time": 90.0 + stint_age * 0.2 + rng.normal(scale=0.5),
                    "rolling_pace": 90.0 + rng.normal(scale=0.3),
                    "class_pace_delta": rng.uniform(-2.0, 2.0),
                    "is_outlier": False,
                    "is_in_lap": False,
                    "is_out_lap": False,
                    "is_traffic_lap": False,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def clean_laps():
    return _make_clean_laps(rng=np.random.default_rng(42))


@pytest.fixture
def enriched_laps(clean_laps):
    return enrich_with_deg_slope(clean_laps)


# ---------------------------------------------------------------------------
# enrich_with_deg_slope
# ---------------------------------------------------------------------------

def test_enrich_adds_deg_slope_column(clean_laps):
    out = enrich_with_deg_slope(clean_laps)
    assert "deg_slope" in out.columns


def test_enrich_preserves_row_count(clean_laps):
    out = enrich_with_deg_slope(clean_laps)
    assert len(out) == len(clean_laps)


def test_enrich_preserves_all_original_columns(clean_laps):
    original_cols = set(clean_laps.columns)
    out = enrich_with_deg_slope(clean_laps)
    assert original_cols.issubset(set(out.columns))


def test_enrich_no_nans_in_deg_slope(clean_laps):
    out = enrich_with_deg_slope(clean_laps)
    assert not out["deg_slope"].isna().any()


def test_enrich_short_stints_get_zero(clean_laps):
    # Remove all but 2 laps per stint so no stint qualifies (< MIN_STINT_LAPS)
    tiny = clean_laps[clean_laps["stint_age"] <= 2].copy()
    out = enrich_with_deg_slope(tiny)
    assert (out["deg_slope"] == 0.0).all()


def test_enrich_slope_constant_within_stint(enriched_laps):
    # All laps in the same (car, stint) must carry the same slope value
    for (car, stint), group in enriched_laps.groupby(["car_number", "stint_id"]):
        assert group["deg_slope"].nunique() == 1, (
            f"car {car} stint {stint} has non-uniform deg_slope"
        )


# ---------------------------------------------------------------------------
# prepare_pace_features
# ---------------------------------------------------------------------------

def test_prepare_returns_x_and_y(enriched_laps):
    X, y = prepare_pace_features(enriched_laps)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_prepare_feature_columns(enriched_laps):
    X, _ = prepare_pace_features(enriched_laps)
    assert list(X.columns) == ["stint_age", "lap_number", "car_class", "deg_slope"]


def test_prepare_y_is_deviation_from_rolling_pace(enriched_laps):
    _, y = prepare_pace_features(enriched_laps)
    expected = (enriched_laps["lap_time"] - enriched_laps["rolling_pace"]).reset_index(drop=True)
    pd.testing.assert_series_equal(y.reset_index(drop=True), expected)


def test_prepare_raises_if_deg_slope_missing(clean_laps):
    with pytest.raises(KeyError, match="deg_slope"):
        prepare_pace_features(clean_laps)


# ---------------------------------------------------------------------------
# train_pace_model
# ---------------------------------------------------------------------------

def test_train_returns_pipeline(enriched_laps):
    model = train_pace_model(enriched_laps)
    assert isinstance(model, Pipeline)


def test_train_model_can_predict(enriched_laps):
    model = train_pace_model(enriched_laps)
    X, _ = prepare_pace_features(enriched_laps)
    preds = model.predict(X)
    assert len(preds) == len(enriched_laps)
    assert not np.isnan(preds).any()


def test_train_rejects_dirty_laps(enriched_laps):
    dirty = enriched_laps.copy()
    dirty.loc[dirty.index[0], "is_outlier"] = True
    with pytest.raises(ValueError, match="is_outlier"):
        train_pace_model(dirty)
