"""
Tests for wec_analytics.ml.clustering.
"""

import numpy as np
import pandas as pd
import pytest

from wec_analytics.ml.clustering import (
    CLUSTER_FEATURES,
    MIN_CARS_TO_CLUSTER,
    _extract_strategy_features,
    cluster_strategies,
)


def _make_car_stints(
    car_number: int,
    car_class: str,
    n_stints: int,
    laps_per_stint: int,
    lap_time_base: float,
    rng: np.random.Generator,
    lap_offset: int = 0,
) -> list[dict]:
    """Build synthetic lap rows for one car."""
    rows = []
    lap_num = 1 + lap_offset
    for stint_id in range(1, n_stints + 1):
        for stint_age in range(1, laps_per_stint + 1):
            rows.append({
                "car_number": car_number,
                "car_class": car_class,
                "stint_id": stint_id,
                "stint_age": stint_age,
                "lap_number": lap_num,
                "lap_time": lap_time_base + rng.normal(scale=0.3),
                "rolling_pace": lap_time_base + rng.normal(scale=0.5),
                "class_pace_delta": rng.uniform(-2.0, 2.0),
                "is_outlier": False,
                "is_in_lap": False,
                "is_out_lap": False,
                "is_traffic_lap": False,
            })
            lap_num += 1
    return rows


@pytest.fixture
def strategy_session():
    """6 cars, two very distinct strategy archetypes.

    Cars 1-3: 6 stints of 3 laps each -> frequent stoppers.
    Cars 4-6: 2 stints of 22 laps each -> long-stint strategists.
    Both n_stints and stint_length_mean differ strongly, giving KMeans a clear
    signal even in the presence of noise in other features.
    """
    rng = np.random.default_rng(7)
    rows = []
    for car_num in range(1, 4):
        rows.extend(_make_car_stints(car_num, "LMP2", 6, 3, 120.0, rng))
    for car_num in range(4, 7):
        rows.extend(_make_car_stints(car_num, "LMP2", 2, 22, 120.0, rng))
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_session(strategy_session):
    """Fewer than MIN_CARS_TO_CLUSTER cars."""
    return strategy_session[strategy_session["car_number"] <= 2].copy()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def test_extract_features_shape(strategy_session):
    features = _extract_strategy_features(strategy_session)
    assert len(features) == 6
    for col in CLUSTER_FEATURES:
        assert col in features.columns, f"Missing feature column: {col}"


def test_extract_features_no_nans(strategy_session):
    features = _extract_strategy_features(strategy_session)
    assert not features[CLUSTER_FEATURES].isna().any().any()


def test_single_stint_car_gets_zero_std():
    rng = np.random.default_rng(0)
    rows = _make_car_stints(1, "Hypercar", 1, 10, 100.0, rng)
    rows += _make_car_stints(2, "Hypercar", 1, 10, 100.0, rng)
    rows += _make_car_stints(3, "Hypercar", 1, 10, 100.0, rng)
    rows += _make_car_stints(4, "Hypercar", 1, 10, 100.0, rng)
    df = pd.DataFrame(rows)
    features = _extract_strategy_features(df)
    assert (features["stint_length_std"] == 0.0).all()


# ---------------------------------------------------------------------------
# KMeans
# ---------------------------------------------------------------------------

def test_kmeans_assigns_all_cars(strategy_session):
    labels_df, _ = cluster_strategies(strategy_session, method="kmeans")
    assert not labels_df["cluster_label"].isna().any()
    assert len(labels_df) == 6


def test_explicit_k_respected(strategy_session):
    labels_df, meta = cluster_strategies(strategy_session, n_clusters=2, method="kmeans")
    assert meta["n_clusters"] == 2
    assert set(labels_df["cluster_label"]).issubset({0, 1})


def test_auto_k_in_range(strategy_session):
    _, meta = cluster_strategies(strategy_session, n_clusters="auto", method="kmeans")
    assert 2 <= meta["n_clusters"] <= 8


def test_auto_k_populates_k_scores(strategy_session):
    _, meta = cluster_strategies(strategy_session, n_clusters="auto", method="kmeans")
    assert meta["k_scores"] is not None
    assert len(meta["k_scores"]) >= 1


def test_short_long_stints_in_different_clusters(strategy_session):
    labels_df, _ = cluster_strategies(strategy_session, n_clusters=2, method="kmeans")
    short_labels = set(labels_df[labels_df["car_number"] <= 3]["cluster_label"])
    long_labels = set(labels_df[labels_df["car_number"] >= 4]["cluster_label"])
    assert short_labels.isdisjoint(long_labels), (
        "Short-stint and long-stint cars should be in different clusters"
    )


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def test_dbscan_runs_without_error(strategy_session):
    labels_df, meta = cluster_strategies(strategy_session, method="dbscan")
    assert "cluster_label" in labels_df.columns
    assert meta["method"] == "dbscan"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_keys_present(strategy_session):
    _, meta = cluster_strategies(strategy_session, method="kmeans")
    required = {
        "method", "n_clusters", "silhouette", "inertia",
        "k_scores", "k_inertias", "pca_coords",
        "pca_explained_variance", "feature_names",
    }
    assert required.issubset(meta.keys())


def test_pca_coords_shape(strategy_session):
    _, meta = cluster_strategies(strategy_session, method="kmeans")
    pca_df = meta["pca_coords"]
    assert list(pca_df.columns) == ["car_number", "PC1", "PC2"]
    assert len(pca_df) == 6


def test_pca_explained_variance_sums_to_at_most_one(strategy_session):
    _, meta = cluster_strategies(strategy_session, method="kmeans")
    assert meta["pca_explained_variance"].sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_too_few_cars_raises(tiny_session):
    with pytest.raises(ValueError, match=str(MIN_CARS_TO_CLUSTER)):
        cluster_strategies(tiny_session, method="kmeans")


def test_invalid_method_raises(strategy_session):
    with pytest.raises(ValueError, match="method"):
        cluster_strategies(strategy_session, method="spectral")
