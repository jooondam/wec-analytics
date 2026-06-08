"""
Pit window prediction model for the wec_analytics ML layer.

Predicts the probability that a given lap is followed by a pit stop.
The target is df['is_in_lap'] -- True means the car pits at the end of
this lap. Outputs are calibrated probabilities rather than hard labels
so that downstream strategy tools can apply their own decision thresholds.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# columns that must never appear as features -- either they are the label,
# directly imply the label, or encode information that only exists after
# the lap in question has ended.
FORBIDDEN = {
    "is_in_lap",               # the label itself
    "is_out_lap",              # true only on the lap after a pit -- leaks the label one row earlier
    "pit_stop_flag",           # alias for is_in_lap in most encodings
    "pit_occurred",            # same
    "laps_remaining_in_stint", # requires knowing when the stint ends -- pure future knowledge
    "stint_end_flag",          # structurally identical to is_in_lap
    "next_stint_id",           # references the stint that begins after the pit
    "next_compound",           # references the tyre fitted after the pit
    "out_lap_flag",            # alias for is_out_lap
    # lap_time_seconds is excluded not because it is unknown at prediction time
    # (it is known once the lap completes) but because it is a proxy for is_in_lap:
    # in-laps are systematically slower due to pit-entry traffic avoidance.
    # the rolling median features in build_lap_features already capture the
    # lagged pace signal without this confound.
    "lap_time_seconds",
    # race_id is a grouping key for GroupKFold, not a feature. if included,
    # the model would learn race-specific pit patterns ("at Spa, lap 25 is
    # a common pit lap") that cannot generalise to unseen races.
    "race_id",
}


def prepare_pit_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and binary target y for pit window prediction.

    The target y = df['is_in_lap'] (True if this lap is followed by a pit stop).

    Unlike the auto-selection path that existed previously, this function now
    requires an explicit feature_columns list. The caller (train_pit_model) is
    responsible for constructing that list deliberately, so that every column
    included has a documented reason for being there.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 1 output (laps + stints + strategy). Must contain 'is_in_lap'.
    feature_columns : list[str]
        Explicit list of columns to use as features. Must not contain any
        column in the module-level FORBIDDEN set. race_id must not appear
        here -- it is a grouping key for cross-validation, not a feature.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing only the requested columns.
    y : pd.Series
        Binary pit label as bool dtype (True = pit stop follows this lap).

    Raises
    ------
    ValueError
        If 'is_in_lap' is missing from df.
        If feature_columns contains any forbidden column.
        If any requested column is missing from df.
    """
    if "is_in_lap" not in df.columns:
        raise ValueError(
            "'is_in_lap' column is missing from the DataFrame. "
            "Run stints.detect_stints() before calling prepare_pit_features()."
        )

    forbidden_requested = set(feature_columns) & FORBIDDEN
    if forbidden_requested:
        raise ValueError(
            "The following columns are forbidden as features (leakage or label): "
            f"{sorted(forbidden_requested)}. Remove them from feature_columns."
        )

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"The following requested feature columns are not present in the DataFrame: "
            f"{sorted(missing)}."
        )

    X = df[feature_columns].copy()
    y = df["is_in_lap"].astype(bool).copy()

    return X, y


def train_pit_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    estimator_name: str = "logistic_regression",
    group_kfold_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train a pit prediction model with grouped k-fold cross-validation.

    Groups are taken from df['race_id'] so that all laps from a given race
    appear in either the train set or the validation set, never both. This
    prevents the model from learning race-specific pit patterns that would
    not generalise to unseen races.

    Parameters
    ----------
    df : pd.DataFrame
        Phase 1 output. Must contain 'is_in_lap' and all columns in
        feature_columns, plus 'race_id' as a grouping key.
    feature_columns : list[str]
        Explicit feature list, already validated by prepare_pit_features.
    numeric_features : list[str]
        Subset of feature_columns to standardise with StandardScaler.
    categorical_features : list[str]
        Subset of feature_columns to encode with OneHotEncoder.
    estimator_name : str, default='logistic_regression'
        One of 'logistic_regression', 'random_forest', or
        'hist_gradient_boosting'.
    group_kfold_splits : int, default=5
        Number of folds. Requires at least this many unique race_id values.
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        'pipeline'       : fitted Pipeline (refit on full data after CV)
        'cv_metrics'     : dict of mean precision, recall, f1, roc_auc, brier
        'cv_predictions' : array of out-of-fold predicted probabilities
        'estimator_name' : str
    """
    X, y = prepare_pit_features(df, feature_columns)
    groups = df["race_id"].values

    if estimator_name == "hist_gradient_boosting":
        # HistGBT is scale-invariant (no StandardScaler needed) and handles
        # categoricals natively via OrdinalEncoder + categorical_features.
        # Output column order: [cat cols...] + [num cols...]
        n_cat = len(categorical_features)
        n_num = len(numeric_features)

        if n_cat > 0:
            cat_mask = [True] * n_cat + [False] * n_num
            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
                    ("num", "passthrough", numeric_features),
                ],
                remainder="drop",
            )
        else:
            # per-class models: car_class is constant, no categorical encoding needed
            cat_mask = None
            preprocessor = ColumnTransformer(
                transformers=[("num", "passthrough", numeric_features)],
                remainder="drop",
            )

        estimator = HistGradientBoostingClassifier(
            categorical_features=cat_mask,
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
        )

    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ],
            remainder="drop",
        )

        if estimator_name == "logistic_regression":
            # class_weight='balanced' reweights each sample so the minority class
            # (pit laps, ~5-10% of laps) receives proportionally higher penalty on
            # misclassification. without this, the model learns "always predict no-pit"
            # and still achieves >90% accuracy while being completely useless.
            estimator = LogisticRegression(
                class_weight="balanced",
                random_state=random_state,
                max_iter=1000,
                solver="lbfgs",
            )

        elif estimator_name == "random_forest":
            base_estimator = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
            )
            # random forest probabilities are skewed towards 0 and 1 because trees
            # vote by majority. CalibratedClassifierCV with Platt scaling (sigmoid)
            # corrects this. known limitation: the final refit below uses cv=3 with
            # random folds, which does not respect race_id groups. acceptable for now
            # but should be replaced with a grouped calibration split in a future iteration.
            estimator = CalibratedClassifierCV(
                base_estimator,
                method="sigmoid",
                cv=3,
            )

        else:
            raise ValueError(
                f"Unknown estimator_name '{estimator_name}'. "
                "Choose 'logistic_regression', 'random_forest', or 'hist_gradient_boosting'."
            )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    gkf = GroupKFold(n_splits=group_kfold_splits)
    proba_collection = np.zeros(len(X))
    fold_metrics = {"precision": [], "recall": [], "f1": [], "roc_auc": [], "brier": []}

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        proba_val = pipeline.predict_proba(X_val)[:, 1]
        proba_collection[val_idx] = proba_val

        pred_val = proba_val >= 0.5
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, pred_val, average="binary", zero_division=0
        )
        fold_metrics["precision"].append(p)
        fold_metrics["recall"].append(r)
        fold_metrics["f1"].append(f1)
        fold_metrics["roc_auc"].append(roc_auc_score(y_val, proba_val))
        fold_metrics["brier"].append(brier_score_loss(y_val, proba_val))

    # refit on full data so the returned pipeline can be persisted and used
    # for inference. cv metrics above are the honest estimate of generalisation.
    pipeline.fit(X, y)

    return {
        "pipeline": pipeline,
        "cv_metrics": {k: float(np.mean(v)) for k, v in fold_metrics.items()},
        "cv_predictions": proba_collection,
        "estimator_name": estimator_name,
    }

def predict_pit_curve(
    pipeline,
    race_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """
    Return predicted pit probability for each lap in race_df.

    Intended for analysis and visualisation -- plot the returned Series
    against lap number to see where the model's confidence rises through
    a stint. For a single actionable lap number, use predict_pit_window.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
        Output of train_pit_model['pipeline']. Must have been trained with
        the same feature_columns passed here.
    race_df : pd.DataFrame
        Laps for a single car, in lap order. Must contain all columns in
        feature_columns.
    feature_columns : list[str]
        Feature columns the pipeline was trained on. Must not include any
        column in FORBIDDEN.

    Returns
    -------
    pd.Series
        Predicted pit probability per lap, indexed to match race_df.
        Values are in [0, 1].
    """
    forbidden_requested = set(feature_columns) & FORBIDDEN
    if forbidden_requested:
        raise ValueError(
            "The following columns are forbidden as features: "
            f"{sorted(forbidden_requested)}."
        )

    missing = set(feature_columns) - set(race_df.columns)
    if missing:
        raise ValueError(
            f"The following feature columns are missing from race_df: {sorted(missing)}."
        )

    X = race_df[feature_columns].copy()
    probabilities = pipeline.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=race_df.index, name="pit_probability")


def predict_pit_window(
    pipeline: Pipeline,
    race_df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float = 0.5,
) -> int | None:
    """
    Predict the next pit lap for a car given its current race state.

    Internally computes a pit probability for every lap in race_df, then
    returns the lap number of the first lap whose probability exceeds the
    threshold. This is the function a strategist would call mid-race to
    get a single actionable recommendation.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline from train_pit_model.
    race_df : pd.DataFrame
        Laps for a single car, in lap order. Must contain all columns in
        feature_columns plus a 'lap_number' column for the return value.
    feature_columns : list[str]
        Exact feature columns used during training.
    threshold : float, default=0.5
        Probability threshold above which a lap is predicted as a pit lap.
        Lower values increase recall (more pit laps flagged) at the cost of
        precision (more false alarms). A strategist might tune this to 0.3
        to get early warning, or 0.7 for high-confidence predictions only.

    Returns
    -------
    int or None
        Lap number of the predicted pit stop, or None if no lap exceeds
        the threshold. None is an honest answer -- the model is saying it
        sees no strong pit signal in the remaining laps, which is itself
        useful information to a strategist.
    """
    pit_curve = predict_pit_curve(pipeline, race_df, feature_columns)
    # find the first lap where probability exceeds the threshold
    candidates = pit_curve[pit_curve >= threshold]
    if candidates.empty:
        return None
    # the index of pit_curve matches race_df's index, so map back to lap_number
    first_pit_idx = candidates.index[0]
    return int(race_df.loc[first_pit_idx, 'lap_number'])


def train_pit_models_per_class(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    estimator_name: str = "hist_gradient_boosting",
    group_kfold_splits: int = 14,
    random_state: int = 42,
) -> dict:
    """Train one pit model per car class and return a dict of fitted pipelines.

    Each model sees only its own class's laps, so it learns class-specific
    stint lengths, fuel windows, and degradation patterns without cross-class
    noise. car_class is excluded from feature_columns since it is constant
    within each model.

    Parameters
    ----------
    df : pd.DataFrame
        Full lap dataset. Must contain car_class and race_id columns.
    feature_columns : list[str]
        Features for each class model. Should NOT include car_class.
    numeric_features : list[str]
        Subset of feature_columns to pass through numerically.
    estimator_name : str
        Estimator to use. Currently only 'hist_gradient_boosting' is supported.
    group_kfold_splits : int
        Max number of CV folds. Capped per class at the number of races
        that class appears in.
    random_state : int
        Random seed.

    Returns
    -------
    dict with keys:
        'models'      : dict mapping car_class str to fitted Pipeline
        'cv_metrics'  : dict mapping car_class str to CV metrics dict
        'classes'     : list of car_class strings that were trained
    """
    classes = sorted(df["car_class"].dropna().unique())
    models: dict = {}
    all_cv_metrics: dict = {}

    for car_class in classes:
        class_df = df[df["car_class"] == car_class].copy()
        n_races = class_df["race_id"].nunique()
        n_pit = int(class_df["is_in_lap"].sum())

        if n_races < 2:
            print(f"  {car_class}: skipped (only {n_races} race, need >= 2 for CV)")
            continue

        folds = min(group_kfold_splits, n_races)
        result = train_pit_model(
            class_df,
            feature_columns=feature_columns,
            numeric_features=numeric_features,
            categorical_features=[],
            estimator_name=estimator_name,
            group_kfold_splits=folds,
            random_state=random_state,
        )
        models[car_class] = result["pipeline"]
        all_cv_metrics[car_class] = result["cv_metrics"]
        cv = result["cv_metrics"]
        print(
            f"  {car_class}: {len(class_df):,} laps  {n_pit} pit stops  "
            f"F1={cv['f1']:.3f}  Recall={cv['recall']:.3f}  Precision={cv['precision']:.3f}"
        )

    return {
        "models": models,
        "cv_metrics": all_cv_metrics,
        "classes": list(classes),
    }


def predict_pit_curve_per_class(
    models: dict,
    race_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Predict pit probability using the appropriate class-specific model per lap.

    Parameters
    ----------
    models : dict
        Dict mapping car_class str to fitted Pipeline, as returned by
        train_pit_models_per_class['models'].
    race_df : pd.DataFrame
        Lap DataFrame containing car_class and all feature_columns.
    feature_columns : list[str]
        Feature columns the per-class models were trained on.

    Returns
    -------
    pd.Series
        Predicted pit probability per lap, aligned to race_df.index.
        Laps whose car_class has no trained model default to 0.5.
    """
    result = pd.Series(0.5, index=race_df.index, name="pit_probability")
    for car_class, model in models.items():
        mask = race_df["car_class"] == car_class
        if mask.any():
            result.loc[mask] = predict_pit_curve(model, race_df[mask], feature_columns)
    return result