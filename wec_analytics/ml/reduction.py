"""
Dimensionality reduction utilities for wec_analytics.

Projects high-dimensional feature matrices to 2D for visualisation and
downstream use. Provides two methods:

PCA (linear baseline):
    Fast, deterministic, interpretable. The returned transformer exposes
    explained_variance_ratio_ so you can see how compressible your feature
    space is. High values mean features are correlated and some could be
    dropped without much loss.

UMAP (non-linear):
    Preserves local and global neighbourhood structure better than PCA or
    t-SNE. Slower and non-deterministic (fix random_state for reproducibility).
    Particularly useful when clusters are non-linearly separable.

Typical usage:
    coords, transformer = reduce_to_2d(feature_df, method='pca')
    # coords has columns ['x', 'y'], same index as feature_df
    # transformer.named_steps['reducer'].explained_variance_ratio_  # PCA only
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

VALID_METHODS = ("pca", "umap")


def reduce_to_2d(
    features: pd.DataFrame,
    method: str = "pca",
    random_state: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, Pipeline]:
    """Project a numeric feature matrix to 2D.

    Applies StandardScaler then the chosen reducer. The scaler and reducer
    are wrapped in an sklearn Pipeline so the full transform can be saved and
    applied to new data with a single call.

    Parameters
    ----------
    features : pd.DataFrame
        Numeric feature matrix. Each row is an observation; columns are
        features. Must not contain NaN values.
    method : 'pca' or 'umap'
        Reduction algorithm.
        - 'pca': linear, fast, explained variance accessible.
        - 'umap': non-linear, slower, better at preserving cluster structure.
    random_state : int
        Random seed for reproducibility (used by PCA and UMAP).
    **kwargs
        Passed through to the underlying reducer constructor (e.g.
        n_neighbors, min_dist for UMAP).

    Returns
    -------
    tuple[pd.DataFrame, Pipeline]
        coords : DataFrame with columns ['x', 'y'], same index as features.
        transformer : fitted sklearn Pipeline with steps 'scaler' and 'reducer'.
            Access explained variance via
            transformer.named_steps['reducer'].explained_variance_ratio_
            (PCA only; UMAP does not expose this).

    Raises
    ------
    ValueError
        If method is not 'pca' or 'umap', or if features contain NaN.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got {method!r}"
        )

    if features.isnull().any().any():
        raise ValueError(
            "features contains NaN values. Fill or drop them before calling reduce_to_2d."
        )

    n_samples = len(features)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        import umap as _umap  # deferred to avoid import cost when not used

        # n_neighbors must be < n_samples; default 15 crashes on tiny datasets
        safe_n_neighbors = min(kwargs.pop("n_neighbors", 15), max(n_samples - 1, 1))
        reducer = _umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=safe_n_neighbors,
            **kwargs,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reducer", reducer),
    ])

    coords_arr = pipeline.fit_transform(features.to_numpy(dtype=float))
    coords = pd.DataFrame(coords_arr, columns=["x", "y"], index=features.index)

    return coords, pipeline


def explained_variance_ratio(transformer: Pipeline) -> np.ndarray | None:
    """Return the PCA explained variance ratio, or None for UMAP.

    Parameters
    ----------
    transformer : Pipeline
        Fitted pipeline returned by reduce_to_2d.

    Returns
    -------
    np.ndarray or None
        Array of shape (2,) for PCA; None for UMAP.
    """
    reducer = transformer.named_steps.get("reducer")
    if reducer is None:
        return None
    return getattr(reducer, "explained_variance_ratio_", None)
