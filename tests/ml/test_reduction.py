"""
Tests for wec_analytics.ml.reduction.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from wec_analytics.ml.reduction import (
    VALID_METHODS,
    explained_variance_ratio,
    reduce_to_2d,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((30, 6)),
        columns=["n_stints", "stint_length_mean", "stint_length_std",
                 "class_delta_mean", "consistency_mean", "deg_slope_mean"],
    )


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(rng.standard_normal((5, 4)), columns=list("abcd"))


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------

def test_pca_output_shape(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="pca")
    assert coords.shape == (len(feature_df), 2)


def test_umap_output_shape(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="umap")
    assert coords.shape == (len(feature_df), 2)


def test_output_columns_are_x_y(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="pca")
    assert list(coords.columns) == ["x", "y"]


def test_umap_output_columns(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="umap")
    assert list(coords.columns) == ["x", "y"]


def test_returns_pipeline(feature_df):
    _, transformer = reduce_to_2d(feature_df, method="pca")
    assert isinstance(transformer, Pipeline)
    assert "scaler" in transformer.named_steps
    assert "reducer" in transformer.named_steps


def test_output_has_no_nans(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="pca")
    assert not coords.isnull().any().any()


def test_umap_output_has_no_nans(feature_df):
    coords, _ = reduce_to_2d(feature_df, method="umap")
    assert not coords.isnull().any().any()


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------

def test_index_preserved(feature_df):
    indexed = feature_df.copy()
    indexed.index = range(100, 100 + len(feature_df))
    coords, _ = reduce_to_2d(indexed, method="pca")
    assert list(coords.index) == list(indexed.index)


# ---------------------------------------------------------------------------
# PCA explained variance
# ---------------------------------------------------------------------------

def test_pca_explained_variance_accessible(feature_df):
    _, transformer = reduce_to_2d(feature_df, method="pca")
    ev = explained_variance_ratio(transformer)
    assert ev is not None
    assert len(ev) == 2
    assert all(0.0 <= v <= 1.0 for v in ev)
    assert sum(ev) <= 1.0 + 1e-9


def test_umap_explained_variance_is_none(feature_df):
    _, transformer = reduce_to_2d(feature_df, method="umap")
    ev = explained_variance_ratio(transformer)
    assert ev is None


# ---------------------------------------------------------------------------
# Method differences and reproducibility
# ---------------------------------------------------------------------------

def test_pca_and_umap_produce_different_coords(feature_df):
    pca_coords, _ = reduce_to_2d(feature_df, method="pca")
    umap_coords, _ = reduce_to_2d(feature_df, method="umap")
    # They should not be identical (different algorithms)
    assert not np.allclose(pca_coords.values, umap_coords.values, atol=1e-3)


def test_pca_deterministic(feature_df):
    c1, _ = reduce_to_2d(feature_df, method="pca", random_state=7)
    c2, _ = reduce_to_2d(feature_df, method="pca", random_state=7)
    pd.testing.assert_frame_equal(c1, c2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_invalid_method_raises(feature_df):
    with pytest.raises(ValueError, match="method"):
        reduce_to_2d(feature_df, method="tsne")


def test_nan_input_raises():
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": [2.0, 3.0]})
    with pytest.raises(ValueError, match="NaN"):
        reduce_to_2d(df, method="pca")


def test_small_dataset_umap_safe_n_neighbors(small_df):
    # 5 samples: default n_neighbors=15 would crash without clamping
    coords, _ = reduce_to_2d(small_df, method="umap")
    assert coords.shape == (5, 2)


# ---------------------------------------------------------------------------
# Integration: clustering still works after refactor
# ---------------------------------------------------------------------------

def _make_cluster_session() -> pd.DataFrame:
    """Minimal session with all columns required by build_stint_features."""
    rng = np.random.default_rng(0)
    n_cars = 6
    rows = []
    for car_num in range(1, n_cars + 1):
        n_stints = 1 if car_num <= 3 else 5
        for stint_id in range(1, n_stints + 1):
            n_laps = 5 if car_num <= 3 else 18
            for stint_age in range(1, n_laps + 1):
                lap_time = 90.0 + rng.normal(scale=0.5)
                rows.append({
                    "car_number": car_num,
                    "car_class": "LMP2" if car_num <= 3 else "Hypercar",
                    "stint_id": stint_id,
                    "stint_age": stint_age,
                    "lap_number": len(rows) + 1,
                    "lap_time": lap_time,
                    "rolling_pace": lap_time + rng.normal(scale=0.1),
                    "class_pace_delta": rng.uniform(-1.0, 1.0),
                    "is_outlier": False,
                    "is_in_lap": False,
                    "is_out_lap": stint_age == n_laps and stint_id < n_stints,
                    "is_traffic_lap": False,
                })
    return pd.DataFrame(rows)


def test_clustering_with_pca_projection():
    from wec_analytics.ml.clustering import cluster_strategies

    session = _make_cluster_session()
    n_cars = session["car_number"].nunique()

    labels_df, meta = cluster_strategies(session, n_clusters=2, projection="pca")
    assert "pca_coords" in meta
    assert meta["pca_coords"].shape == (n_cars, 3)  # car_number, PC1, PC2
    ev = meta["pca_explained_variance"]
    assert ev is not None and len(ev) == 2


def test_clustering_with_umap_projection():
    from wec_analytics.ml.clustering import cluster_strategies

    session = _make_cluster_session()
    n_cars = session["car_number"].nunique()

    labels_df, meta = cluster_strategies(session, n_clusters=2, projection="umap")
    assert meta["projection"] == "umap"
    assert meta["pca_coords"].shape == (n_cars, 3)
    assert meta["pca_explained_variance"] is None  # UMAP doesn't have this
