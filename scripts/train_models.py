#!/usr/bin/env python3
"""
Training script for the wec-analytics ML pipeline.

Fetches all training races from Al Kamel Systems (uses local cache if already
downloaded), runs the full feature engineering pipeline, trains a pace regression
model and a pit window classifier, evaluates both with leave-one-race-out
cross-validation, and saves fitted models + metadata sidecars to models_trained/.

Usage (from project root with venv active):
    python scripts/train_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from wec_analytics.analysis.laps import detect_traffic_lap
from wec_analytics.analysis.stints import detect_outliers
from wec_analytics.ingestion.alkamelsystems import extract_session_id, fetch_session
from wec_analytics.ingestion.models import clean_session
from wec_analytics.ingestion.sessions import SESSION_MAP
from wec_analytics.ml.evaluation import print_pace_comparison, run_pace_evaluation
from wec_analytics.ml.degradation import enrich_with_deg_slope
from wec_analytics.ml.features import LAP_CLEAN_FLAGS, build_lap_features, enrich_with_norm_stint_age
from wec_analytics.ml.pace import train_pace_model
from wec_analytics.ml.persistence import make_versioned_path, save_model
from wec_analytics.ml.pit_window import train_pit_model


TRAINING_URLS = [
    # Spa-Francorchamps 2019 -- Round 7 of the 2018-2019 Super Season
    "http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Fuji Speedway 2018 -- Round 4
    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/04_FUJI%20SPEEDWAY/246_FIA%20WEC/201810141100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Shanghai International Circuit 2018 -- Round 5
    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/05_SHANGHAI%20INTERNATIONAL%20CIRCUIT/256_FIA%20WEC/201811181100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Silverstone 2018 -- Round 3
    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/03_SILVERSTONE/239_FIA%20WEC/201808191200_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Sebring 2019 -- Round 6 (8-hour race)
    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/06_SEBRING/260_FIA%20WEC/201903151600_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
    # Le Mans 2019 -- Round 8, 24 Hours of Le Mans
    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/08_LE%20MANS/276_FIA%20WEC/201906151500_Race/24_Hour%2024/23_Analysis_Race_Hour%2024.CSV",
    # 2019-2020 season
    # Silverstone 2019 -- Round 1 (4-hour race)
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/01_SILVERSTONE/285_FIA%20WEC/201909011200_Race/Hour%204/23_Analysis_Race_Hour%204.CSV",
    # Fuji Speedway 2019 -- Round 2
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/02_FUJI%20SPEEDWAY/294_FIA%20WEC/201910061100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Shanghai 2019 -- Round 3 (4-hour race)
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/03_SHANGHAI%20INTERNATIONAL%20CIRCUIT/01_FIA%20WEC/201911101200_Race/Hour%204/23_Analysis_Race_Hour%204.CSV",
    # Bahrain 2019 -- Round 4 (8-hour race)
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/04_BAHRAIN%20INTERNATIONAL%20CIRCUIT/307_FIA%20WEC/201912141500_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
    # Circuit of the Americas 2020 -- Round 5
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/05_CIRCUIT%20OF%20THE%20AMERICAS/311_FIA%20WEC/202002231200_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Spa-Francorchamps 2020 -- Round 6
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/06_SPA%20FRANCORCHAMPS/327_FIA%20WEC/202008151330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    # Le Mans 2020 -- Round 7, 24 Hours of Le Mans
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/07_LE%20MANS/335_FIA%20WEC/202009191430_Race/Hour%2024/23_Analysis_Race_Hour%2024.CSV",
    # Bahrain 2020 -- Round 8 season finale (8-hour race)
    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/08_BAHRAIN%20INTERNATIONAL%20CIRCUIT/348_FIA%20WEC/202011141400_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
]

# Pit classifier features -- all safe against the FORBIDDEN leakage list.
# rolling_pace uses closed='left' so it contains only prior laps' times.
# class_pace_delta is excluded because it embeds the current lap's lap_time,
# which correlates with is_in_lap (in-laps are systematically slower).
PIT_FEATURE_COLUMNS = ["stint_age", "norm_stint_age", "pace_trend", "rolling_pace", "lap_number", "car_class"]
PIT_NUMERIC_FEATURES = ["stint_age", "norm_stint_age", "pace_trend", "rolling_pace", "lap_number"]
PIT_CATEGORICAL_FEATURES = ["car_class"]

DIRTY_FLAGS = LAP_CLEAN_FLAGS


def _load_session(url: str) -> pd.DataFrame:
    """Run the full Phase 1 + feature engineering pipeline for one race URL."""
    session_id = extract_session_id(url)
    meta = SESSION_MAP[session_id]

    raw = fetch_session(url)
    cleaned = clean_session(raw)
    with_outliers = detect_outliers(cleaned)
    with_traffic = detect_traffic_lap(with_outliers)
    featured = enrich_with_norm_stint_age(build_lap_features(with_traffic))
    featured["race_id"] = meta.race_id

    n_cars = featured["car_number"].nunique()
    print(f"  {meta.race_id}: {len(featured):,} laps  ({n_cars} cars)")
    return featured


def main() -> None:
    print("=== wec-analytics: model training ===\n")

    print("Loading sessions...")
    sessions = [_load_session(url) for url in TRAINING_URLS]
    all_laps = pd.concat(sessions, ignore_index=True)

    n_races = all_laps["race_id"].nunique()
    print(f"\nCombined dataset: {len(all_laps):,} laps  |  {n_races} races\n")

    # ------------------------------------------------------------------
    # Pace regression model
    # ------------------------------------------------------------------
    print("--- Pace regression model ---")

    clean_laps = (
        all_laps[~all_laps[DIRTY_FLAGS].any(axis=1)]
        .dropna(subset=["rolling_pace"])
        .copy()
    )
    removed = len(all_laps) - len(clean_laps)
    print(f"Clean laps: {len(clean_laps):,}  (removed {removed:,} dirty/NaN rows)")

    clean_laps = enrich_with_deg_slope(clean_laps)
    print(f"Degradation slope enriched: {(clean_laps['deg_slope'] != 0.0).sum():,} laps with non-zero slope")

    pace_pipeline = train_pace_model(clean_laps)

    # run_pace_evaluation clones the pipeline internally so passing the
    # already-fitted object here is safe -- each fold gets a fresh copy
    eval_results = run_pace_evaluation(clean_laps, pace_pipeline, cv=n_races)
    print_pace_comparison(eval_results)

    pace_path = make_versioned_path("pace_linear")
    save_model(
        pace_pipeline,
        pace_path,
        metadata={
            "model_type": "pace_regression",
            "estimator": "HistGradientBoostingRegressor",
            "features": ["stint_age", "lap_number", "car_class", "deg_slope"],
            "target": "lap_time_minus_rolling_pace",
            "training_races": sorted(all_laps["race_id"].unique().tolist()),
            "n_clean_laps": int(len(clean_laps)),
            "cv_rmse_mean": eval_results["model"]["rmse_mean"],
            "cv_rmse_std": eval_results["model"]["rmse_std"],
            "baseline_rmse_mean": eval_results["baseline"]["rmse_mean"],
        },
    )

    # ------------------------------------------------------------------
    # Pit window classifier
    # ------------------------------------------------------------------
    print("\n--- Pit window classifier ---")

    pit_df = all_laps.dropna(subset=PIT_FEATURE_COLUMNS).copy()
    n_pit_laps = pit_df["is_in_lap"].sum()
    print(f"Training laps: {len(pit_df):,}  ({int(n_pit_laps)} pit laps, {n_pit_laps/len(pit_df)*100:.1f}%)")

    pit_result = train_pit_model(
        pit_df,
        feature_columns=PIT_FEATURE_COLUMNS,
        numeric_features=PIT_NUMERIC_FEATURES,
        categorical_features=PIT_CATEGORICAL_FEATURES,
        estimator_name="hist_gradient_boosting",
        group_kfold_splits=n_races,
    )

    cv = pit_result["cv_metrics"]
    print(f"\nCV results (GroupKFold, {n_races} folds -- leave-one-race-out):")
    print(f"  F1:        {cv['f1']:.3f}")
    print(f"  ROC AUC:   {cv['roc_auc']:.3f}")
    print(f"  Precision: {cv['precision']:.3f}")
    print(f"  Recall:    {cv['recall']:.3f}")
    print(f"  Brier:     {cv['brier']:.3f}")

    pit_path = make_versioned_path("pit_logistic")
    save_model(
        pit_result["pipeline"],
        pit_path,
        metadata={
            "model_type": "pit_window_classifier",
            "estimator": "HistGradientBoostingClassifier",
            "feature_columns": PIT_FEATURE_COLUMNS,
            "numeric_features": PIT_NUMERIC_FEATURES,
            "categorical_features": PIT_CATEGORICAL_FEATURES,
            "training_races": sorted(all_laps["race_id"].unique().tolist()),
            "n_training_laps": int(len(pit_df)),
            "cv_f1": cv["f1"],
            "cv_roc_auc": cv["roc_auc"],
            "cv_brier": cv["brier"],
            "cv_precision": cv["precision"],
            "cv_recall": cv["recall"],
        },
    )

    # ------------------------------------------------------------------
    print("\n=== Training complete ===")
    print(f"Pace model:  {pace_path}")
    print(f"Pit model:   {pit_path}")


if __name__ == "__main__":
    main()
