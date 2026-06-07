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
    print(f"Model (HistGBT regressor):  {model_rmse:.4f}s")
    print(f"Improvement:                {improvement:+.1f}%  ({baseline_rmse - model_rmse:+.4f}s)")
    print("=============================")


def cross_validate_pit(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "is_in_lap",
    groups: np.ndarray | pd.Series | None = None,
    estimator=None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Cross-validate a pit classifier using GroupKFold.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 1 output containing features and target.
    feature_columns : list[str]
        Columns to use as features (must already be pre-selected, no leakage).
    target_column : str, default='is_in_lap'
        Name of the binary pit label column.
    groups : array-like, optional
        Group labels for GroupKFold. If None, attempts to use df['race_id'].
        Passing groups explicitly is preferred -- it makes the grouping visible
        at the call site rather than hidden inside this function.
    estimator : sklearn estimator or Pipeline, optional
        If provided, used as the classifier. Must expose predict_proba.
        If None, a default LogisticRegression with balanced class weight is used.
        Note: if the estimator is a Pipeline with a ColumnTransformer, X is
        passed as a DataFrame (with column names intact) rather than a numpy
        array, so the transformer can reference columns by name.
    n_splits : int, default=5
        Number of folds for GroupKFold. Requires at least n_splits unique groups.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        'model_metrics'    : dict with precision, recall, f1, roc_auc, brier
        'dummy_metrics'    : same keys for DummyClassifier(strategy='most_frequent')
        'model_predictions': array of cross-validated predicted probabilities
        'dummy_predictions': array of dummy predicted probabilities
        'true_labels'      : array of true y values
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score

    # keep X as a DataFrame so that Pipeline + ColumnTransformer can reference
    # columns by name. converting to numpy here would silently break any
    # estimator that uses a ColumnTransformer internally.
    X = df[feature_columns].copy()
    y = df[target_column].astype(bool).values

    if groups is None:
        if "race_id" not in df.columns:
            raise KeyError(
                "No groups provided and 'race_id' column not found in DataFrame. "
                "Pass groups explicitly or ensure the DataFrame has a 'race_id' column."
            )
        groups = df["race_id"].values
    else:
        groups = np.asarray(groups)

    if estimator is None:
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        )

    dummy = DummyClassifier(strategy="most_frequent", random_state=random_state)

    gkf = GroupKFold(n_splits=n_splits)

    # collect out-of-fold predicted probabilities by iterating folds manually.
    # cross_val_predict would be cleaner but does not accept a groups argument
    # in all sklearn versions when method='predict_proba', so we loop explicitly
    # to be safe and to keep the two estimators on identical fold splits.
    model_proba = np.zeros(len(X))
    dummy_proba = np.zeros(len(X))

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y[train_idx]

        estimator.fit(X_train, y_train)
        dummy.fit(X_train, y_train)

        model_proba[val_idx] = estimator.predict_proba(X_val)[:, 1]
        dummy_proba[val_idx] = dummy.predict_proba(X_val)[:, 1]

    # threshold at 0.5 for precision/recall/f1 -- note that this threshold
    # choice matters a lot for imbalanced data. the roc_auc and brier scores
    # below are threshold-independent and are therefore the more honest
    # summary of model quality.
    model_pred = model_proba >= 0.5
    dummy_pred = dummy_proba >= 0.5

    model_prec, model_rec, model_f1, _ = precision_recall_fscore_support(
        y, model_pred, average="binary", zero_division=0
    )
    dummy_prec, dummy_rec, dummy_f1, _ = precision_recall_fscore_support(
        y, dummy_pred, average="binary", zero_division=0
    )

    model_auc = roc_auc_score(y, model_proba)

    # DummyClassifier(strategy='most_frequent') always predicts the same class,
    # so dummy_proba is a constant array. roc_auc_score requires discriminative
    # variation to produce a meaningful value -- we assign 0.5 explicitly since
    # random chance is the correct characterisation of a non-discriminating baseline.
    try:
        dummy_auc = roc_auc_score(y, dummy_proba)
    except ValueError:
        dummy_auc = 0.5

    model_brier = brier_score_loss(y, model_proba)
    dummy_brier = brier_score_loss(y, dummy_proba)

    return {
        "model_metrics": {
            "precision": float(model_prec),
            "recall":    float(model_rec),
            "f1":        float(model_f1),
            "roc_auc":   float(model_auc),
            "brier":     float(model_brier),
        },
        "dummy_metrics": {
            "precision": float(dummy_prec),
            "recall":    float(dummy_rec),
            "f1":        float(dummy_f1),
            "roc_auc":   float(dummy_auc),
            "brier":     float(dummy_brier),
        },
        "model_predictions": model_proba,
        "dummy_predictions": dummy_proba,
        "true_labels":       y,
    }


def print_pit_comparison(results: dict) -> None:
    """
    Print a side-by-side comparison of model vs dummy classifier for pit prediction.

    Parameters
    ----------
    results : dict
        Return value from cross_validate_pit, containing 'model_metrics' and
        'dummy_metrics' keys.
    """
    model = results["model_metrics"]
    dummy = results["dummy_metrics"]

    # F1 improvement -- higher is better. when the dummy f1 is zero (which it
    # always will be for most_frequent on an imbalanced dataset), we cannot
    # compute a percentage so we cap the display string instead of printing
    # "infinite" mid-sentence, which reads awkwardly.
    if dummy["f1"] == 0:
        f1_str = ">999%" if model["f1"] > 0 else "+0.0%"
    else:
        f1_improvement = (model["f1"] - dummy["f1"]) / dummy["f1"] * 100
        f1_str = f"+{f1_improvement:.1f}%"

    # Brier improvement -- lower is better, so improvement is reduction relative
    # to dummy. the sign convention is deliberately inverted compared to f1:
    # a positive improvement means the model's brier is smaller than the dummy's.
    if dummy["brier"] == 0:
        brier_str = "+0.0%"
    else:
        brier_improvement = (dummy["brier"] - model["brier"]) / dummy["brier"] * 100
        brier_str = f"+{brier_improvement:.1f}%"

    print("\n" + "=" * 70)
    print("Pit Prediction: Model vs Dummy Classifier (majority class)")
    print("=" * 70)
    print(f"{'Metric':<15} {'Model':>12} {'Dummy':>12} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Precision':<15} {model['precision']:>12.3f} {dummy['precision']:>12.3f} {'':>15}")
    print(f"{'Recall':<15} {model['recall']:>12.3f} {dummy['recall']:>12.3f} {'':>15}")
    print(f"{'F1':<15} {model['f1']:>12.3f} {dummy['f1']:>12.3f} {f1_str:>15}")
    print(f"{'ROC AUC':<15} {model['roc_auc']:>12.3f} {dummy['roc_auc']:>12.3f} {'':>15}")
    print(f"{'Brier':<15} {model['brier']:>12.3f} {dummy['brier']:>12.3f} {brier_str:>15}")
    print("=" * 70)
    print("Interpretation:")
    print(f"  - F1 improvement of {f1_str} over dummy means the model actually detects pit laps.")
    print(f"  - Brier improvement of {brier_str} means probability estimates are better calibrated.")
    if model["recall"] < 0.1:
        print("  - Warning: recall is very low -- the model is missing most pit laps.")
    elif model["precision"] < 0.3:
        print("  - Warning: precision is low -- too many false alarms.")
    else:
        print("  - Model provides useful pit probability estimates.")
    print("=" * 70)
