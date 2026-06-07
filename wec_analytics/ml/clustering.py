"""
Strategy clustering for wec_analytics ML layer.

Groups cars into strategy archetypes (early stopper, late stopper, even-split,
push-and-conserve) using per-car features derived from stint structure and tyre
degradation. Uses class-relative pace features so the clustering finds strategy
archetypes rather than the class hierarchy (Hypercar vs LMP2 vs GTE).

This is unsupervised learning: there are no ground-truth labels. Validation is
interpretability — do the clusters make sense to a race strategist?
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from wec_analytics.ml.degradation import compare_degradation
from wec_analytics.ml.features import build_stint_features


CLUSTER_FEATURES = [
    "n_stints",           # number of pit stops + 1 (primary strategy signal)
    "stint_length_mean",  # average laps per stint
    "stint_length_std",   # regularity of stop timing (0 = perfectly even)
    "class_delta_mean",   # pace relative to class best (effort level)
    "consistency_mean",   # mean intra-stint lap_time_std
    "deg_slope_mean",     # tyre degradation rate from Milestone 4
]

MIN_CARS_TO_CLUSTER = 4


def _extract_strategy_features(session: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session laps into one per-car row of strategy features.

    Parameters
    ----------
    session : pd.DataFrame
        Full session lap DataFrame from build_lap_features. Must contain all
        columns required by build_stint_features.

    Returns
    -------
    pd.DataFrame
        One row per car with columns: car_number, car_class, and all six
        CLUSTER_FEATURES. NaN values for stint_length_std (single-stint cars)
        and deg_slope_mean (cars with no qualifying degradation stints) are
        filled with 0.
    """
    stint_df = build_stint_features(session)

    per_car = (
        stint_df.groupby(["car_number", "car_class"])
        .agg(
            n_stints=("stint_length", "count"),
            stint_length_mean=("stint_length", "mean"),
            stint_length_std=("stint_length", "std"),
            consistency_mean=("lap_time_std", "mean"),
            class_delta_mean=("class_pace_delta_median", "mean"),
        )
        .reset_index()
    )

    deg = compare_degradation(session)
    if not deg.empty:
        per_car = per_car.merge(
            deg[["car_number", "deg_slope_mean"]], on="car_number", how="left"
        )
    else:
        per_car["deg_slope_mean"] = 0.0

    per_car["stint_length_std"] = per_car["stint_length_std"].fillna(0.0)
    per_car["deg_slope_mean"] = per_car["deg_slope_mean"].fillna(0.0)

    return per_car


def _choose_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
) -> tuple[int, dict, dict]:
    """Select k for KMeans by maximising silhouette score.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Candidate values of k to evaluate.

    Returns
    -------
    tuple[int, dict, dict]
        (best_k, silhouette_scores, inertias) where both dicts map k -> value.
    """
    sil_scores: dict[int, float] = {}
    inertias: dict[int, float] = {}
    n = len(X_scaled)

    for k in k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias[k] = float(km.inertia_)
        if len(set(labels)) > 1:
            sil_scores[k] = float(silhouette_score(X_scaled, labels))

    if not sil_scores:
        best_k = min(k_range[0], n - 1)
    else:
        best_k = max(sil_scores, key=sil_scores.__getitem__)

    return best_k, sil_scores, inertias


def _auto_eps(X_scaled: np.ndarray, min_samples: int = 3) -> float:
    """Estimate DBSCAN epsilon from the k-NN distance elbow.

    Fits a k-nearest-neighbour model (k=min_samples), sorts the distances to
    the k-th neighbour for each point, then returns the distance at the point
    of maximum curvature (the elbow) as the epsilon estimate.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    min_samples : int
        DBSCAN min_samples parameter; also the k for kNN.

    Returns
    -------
    float
        Estimated epsilon.
    """
    k = min(min_samples, len(X_scaled) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    knn_dists = np.sort(distances[:, -1])

    if len(knn_dists) < 3:
        return float(knn_dists[-1])

    second_diff = np.diff(np.diff(knn_dists))
    elbow_idx = int(np.argmax(second_diff)) + 1
    return float(knn_dists[elbow_idx])


def cluster_strategies(
    session: pd.DataFrame,
    n_clusters: int | str = "auto",
    method: str = "kmeans",
) -> tuple[pd.DataFrame, dict]:
    """Cluster cars in a session into strategy archetypes.

    Parameters
    ----------
    session : pd.DataFrame
        Full session lap DataFrame from build_lap_features.
    n_clusters : int or 'auto'
        Number of clusters for KMeans. When 'auto', selects k in range(2, 9)
        by maximising silhouette score. Ignored for DBSCAN.
    method : 'kmeans' or 'dbscan'
        Clustering algorithm. DBSCAN auto-tunes epsilon from k-NN distances
        and labels outlier cars as cluster -1.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        labels_df : DataFrame with one row per car containing all
            CLUSTER_FEATURES and a ``cluster_label`` column.
        metadata : dict with keys:
            - method (str)
            - n_clusters (int)
            - silhouette (float or None)
            - inertia (float or None, KMeans only)
            - k_scores (dict or None, KMeans auto mode only: k -> silhouette)
            - k_inertias (dict or None, KMeans auto mode only: k -> inertia)
            - pca_coords (DataFrame: car_number, PC1, PC2)
            - pca_explained_variance (np.ndarray)
            - feature_names (list[str])

    Raises
    ------
    ValueError
        If fewer than MIN_CARS_TO_CLUSTER cars are present in the session.
    ValueError
        If method is not 'kmeans' or 'dbscan'.
    """
    if method not in ("kmeans", "dbscan"):
        raise ValueError(f"method must be 'kmeans' or 'dbscan', got {method!r}")

    features = _extract_strategy_features(session)

    if len(features) < MIN_CARS_TO_CLUSTER:
        raise ValueError(
            f"Need at least {MIN_CARS_TO_CLUSTER} cars to cluster, "
            f"got {len(features)}."
        )

    X = features[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pca_coords_arr = pca.fit_transform(X_scaled)
    pca_coords = pd.DataFrame(
        pca_coords_arr, columns=["PC1", "PC2"]
    )
    pca_coords.insert(0, "car_number", features["car_number"].values)

    k_scores = None
    k_inertias = None
    inertia = None
    sil = None

    if method == "kmeans":
        if n_clusters == "auto":
            chosen_k, k_scores, k_inertias = _choose_k(X_scaled)
        else:
            chosen_k = int(n_clusters)

        km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        n_clusters_actual = chosen_k

    else:  # dbscan
        eps = _auto_eps(X_scaled, min_samples=3)
        db = DBSCAN(eps=eps, min_samples=3)
        labels = db.fit_predict(X_scaled)
        n_clusters_actual = int(len(set(labels) - {-1}))

    if len(set(labels)) > 1 and -1 not in set(labels):
        sil = float(silhouette_score(X_scaled, labels))
    elif len(set(labels) - {-1}) > 1:
        mask = labels != -1
        if mask.sum() > 1:
            sil = float(silhouette_score(X_scaled[mask], labels[mask]))

    labels_df = features.copy()
    labels_df["cluster_label"] = labels

    metadata = {
        "method": method,
        "n_clusters": n_clusters_actual,
        "silhouette": sil,
        "inertia": inertia,
        "k_scores": k_scores,
        "k_inertias": k_inertias,
        "pca_coords": pca_coords,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "feature_names": CLUSTER_FEATURES,
    }

    return labels_df, metadata
