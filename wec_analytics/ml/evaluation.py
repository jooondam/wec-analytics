"""
Shared evaluation utilities for the wec_analytics ML layer.

Owns cross-validation, baseline comparison, and metric aggregation.
These helpers are model-agnostic — pace.py, pit_window.py, and future
milestone modules all import from here rather than duplicating GroupKFold
logic across files.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from wec_analytics.ml.pace import prepare_pace_features


def cross_validate_pace(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator,
    cv: int = 5,
) -> dict:
    """
    Evaluate a pace model using GroupKFold with race_id as groups.

    Groups by race rather than session so that all CSV segments belonging
    to the same race weekend (e.g. Hour 1, Hour 2) land in the same fold.
    This prevents the model from absorbing a race's track layout, weather
    profile, and BoP settings in training and then predicting on the same
    event in test — which would inflate performance scores dishonestly.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (clean laps only).
    y : pd.Series
        Target lap times.
    groups : pd.Series
        race_id values aligned to X and y rows.
    estimator : sklearn-compatible estimator
        Unfitted pipeline or regressor. Cloned fresh for every fold so
        fitted state never leaks between splits.
    cv : int, optional
        Number of folds. Must be <= number of unique race_ids. Default 5.

    Returns
    -------
    dict
        Mean and std of MAE, RMSE, R² across folds, plus raw per-fold
        lists so individual races can be inspected when one fold's RMSE
        is suspiciously high.
    """
    gkf = GroupKFold(n_splits=cv)
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # clone gives a fresh unfitted copy with the same hyperparameters —
        # without this, each fold's fit() overwrites the previous fold's
        # stored encoder categories and regression coefficients
        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        # root_mean_squared_error replaces the deprecated squared=False
        # argument removed in sklearn 1.6
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        # raw per-fold scores for diagnosing which races the model
        # generalises to poorly — a high-RMSE fold often points to a
        # specific track or weather condition not well represented in training
        "mae_per_fold": mae_scores,
        "rmse_per_fold": rmse_scores,
        "r2_per_fold": r2_scores,
    }


def baseline_cross_validate(
    y: pd.Series,
    groups: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Evaluate a mean-prediction baseline using the same GroupKFold splits.

    The baseline predicts the training-set mean lap time for every lap in
    the test fold. It intentionally ignores all features — its RMSE is the
    floor that any real model must beat to be considered useful. A model
    that only narrowly outperforms this baseline is not learning meaningful
    structure from the features.

    Parameters
    ----------
    y : pd.Series
        Target lap times (clean laps only, aligned to groups).
    groups : pd.Series
        race_id values aligned to y.
    cv : int, optional
        Number of folds. Must be <= number of unique race_ids. Default 5.

    Returns
    -------
    dict
        Same structure as cross_validate_pace for direct comparison.
    """
    # DummyRegressor(strategy="mean") fits by storing y_train.mean() and
    # returns that single number for every row in X_test — no features used
    dummy = DummyRegressor(strategy="mean")
    gkf = GroupKFold(n_splits=cv)

    # dummy still needs an X to satisfy sklearn's API, but its values are
    # never used — a single constant column is enough to satisfy the contract
    X_dummy = pd.DataFrame({"_dummy": np.ones(len(y))})

    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, test_idx in gkf.split(X_dummy, y, groups):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train = X_dummy.iloc[train_idx]
        X_test = X_dummy.iloc[test_idx]

        dummy_fold = clone(dummy)
        dummy_fold.fit(X_train, y_train)
        y_pred = dummy_fold.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "mae_per_fold": mae_scores,
        "rmse_per_fold": rmse_scores,
        "r2_per_fold": r2_scores,
    }


def run_pace_evaluation(
    clean_laps: pd.DataFrame,
    pipeline,
    cv: int = 5,
) -> dict:
    """
    Compare the pace model against the mean-prediction baseline using
    GroupKFold cross-validation grouped by race_id.

    Parameters
    ----------
    clean_laps : pd.DataFrame
        Clean laps only (all outlier/in/out/traffic laps already removed).
        Must contain 'lap_time', 'race_id', and all feature columns needed
        by prepare_pace_features.
    pipeline : sklearn.pipeline.Pipeline or compatible estimator
        Unfitted pace regression pipeline — will be cloned and fitted fresh
        on each fold so the caller's object is never mutated.
    cv : int, optional
        Number of cross-validation folds (default 5).

    Returns
    -------
    dict
        Nested dict with keys 'baseline' and 'model', each containing the
        output of baseline_cross_validate and cross_validate_pace respectively.
        Compare result['baseline']['rmse_mean'] against result['model']['rmse_mean']
        to assess whether the model is learning anything useful.
    """
    if "race_id" not in clean_laps.columns:
        raise ValueError(
            "clean_laps is missing a 'race_id' column. "
            "Run attach_race_id() on the DataFrame before calling run_pace_evaluation."
        )

    # extract groups separately so prepare_pace_features stays single-purpose
    X, y = prepare_pace_features(clean_laps)
    groups = clean_laps["race_id"]

    # baseline first — establishes the floor before the model is evaluated
    baseline_results = baseline_cross_validate(y, groups, cv=cv)
    model_results = cross_validate_pace(X, y, groups, estimator=pipeline, cv=cv)

    return {
        "baseline": baseline_results,
        "model": model_results,
    }


def print_pace_comparison(eval_results: dict) -> None:
    """
    Print a human-readable RMSE comparison between the baseline and pace model.

    Parameters
    ----------
    eval_results : dict
        Output of run_pace_evaluation — expects nested keys
        'baseline' and 'model', each containing 'rmse_mean'.
    """
    # fail loudly if the dict shape is wrong rather than printing a
    # misleading "missing values" message that hides the real bug
    try:
        baseline_rmse = eval_results["baseline"]["rmse_mean"]
        model_rmse = eval_results["model"]["rmse_mean"]
    except KeyError as e:
        raise KeyError(
            f"Expected key {e} not found in eval_results. "
            "Pass the output of run_pace_evaluation directly."
        ) from None

    improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100

    print("\n=== Pace Model Evaluation ===")
    print(f"Baseline (mean predictor):  {baseline_rmse:.4f}s")
    print(f"Model (linear regression):  {model_rmse:.4f}s")
    print(f"Improvement:                {improvement:+.1f}%  ({baseline_rmse - model_rmse:+.4f}s)")
    print("=============================")
