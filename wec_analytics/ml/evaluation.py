import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold


def cross_validate_pace(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    estimator,
    cv: int = 5,
) -> dict:
    """
    Evaluate the pace model using GroupKFold with race_id as groups.

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
        Mean and standard deviation of MAE, RMSE, and R² across folds,
        plus the raw per-fold lists so individual races can be inspected.
    """
    gkf = GroupKFold(n_splits=cv)

    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # clone gives a fresh, unfitted copy with the same hyperparameters —
        # without this, each fold's fit() would overwrite the previous fold's
        # stored encoder categories and regression coefficients on the same object
        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred))
        # root_mean_squared_error replaces the deprecated squared=False argument
        # removed in sklearn 1.6
        rmse_scores.append(root_mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    return {
        # summary statistics for quick model comparison
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std": np.std(rmse_scores),
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        # raw per-fold scores so you can trace which races the model
        # struggles on — a high-RMSE fold often points to a specific
        # track or weather condition the model hasn't generalised to
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
    Evaluate the mean-prediction baseline using the same GroupKFold split.

    For every fold, predicts the training-set mean lap time for every test lap —
    no features, no model, just a single constant. This is the floor any useful
    model must clear: if cross_validate_pace can't beat these numbers, the
    features are contributing nothing beyond knowing the average pace.

    Parameters
    ----------
    y : pd.Series
        Target lap times (same series passed to cross_validate_pace).
    groups : pd.Series
        race_id values aligned to y rows (same series passed to cross_validate_pace).
    cv : int, optional
        Number of folds. Must match the value used in cross_validate_pace. Default 5.

    Returns
    -------
    dict
        Same key schema as cross_validate_pace so results can be compared directly.
    """
    # X is unused by DummyRegressor but GroupKFold.split() requires it —
    # a zero-column DataFrame with the right index is the lightest possible stand-in
    X_dummy = pd.DataFrame(index=y.index)
    dummy = DummyRegressor(strategy="mean")
    return cross_validate_pace(X_dummy, y, groups, estimator=dummy, cv=cv)
