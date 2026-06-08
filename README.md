# wec-analytics

Machine learning pipeline for FIA WEC race strategy analysis. Takes raw Al Kamel Systems timing CSVs, builds a clean lap-level feature dataset, and trains models to predict pace, pit windows, and strategy patterns.

**[Live demo](https://wec-analytics-obljzqgpkghba3w4wb4rmr.streamlit.app)** | **[Portfolio](https://jooondam.github.io/wec-analytics)**

---

## What it covers

| Layer | Output |
|---|---|
| Ingestion | Fetches and caches Al Kamel CSVs, parses lap times and flag columns |
| Cleaning | Detects outlier laps, in/out laps, traffic laps, assigns stint IDs |
| Feature engineering | Rolling median pace, stint age, class pace delta, tyre deg slope |
| Pace regression | HistGradientBoostingRegressor predicting deviation from rolling pace |
| Pit classifier | HistGradientBoostingClassifier predicting pit-stop probability per lap |
| Anomaly detection | Isolation Forest and LOF for multi-feature lap anomaly scoring |
| Strategy clustering | KMeans / DBSCAN on per-car strategy features with PCA / UMAP projection |
| Association rules | Apriori on 15-minute race windows to surface strategy event patterns |
| Dimensionality reduction | PCA and UMAP for 2D projection of any feature matrix |

## Dataset

- 6 races from the 2018-2019 FIA WEC Super Season
- 46,088 laps across Spa, Fuji, Shanghai, Silverstone, Sebring, and Le Mans
- 231 car entries across Hypercar, LMP2, LMGTE Pro, LMGTE Am

Data from [Al Kamel Systems](https://www.alkamelsystems.com), the official FIA WEC timing provider.

## Setup

Python 3.11+.

```bash
git clone https://github.com/jooondam/wec-analytics.git
cd wec-analytics
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Seven tabs: Race Overview, Pace Residuals, Pit Probability, Tyre Degradation, Strategy Clusters, Anomaly Detection, Strategy Patterns. Race data is cached after the first fetch.

## Train the models

```bash
python scripts/train_models.py
```

Fetches all six races, runs the full pipeline, trains pace and pit models with leave-one-race-out CV, saves versioned `.joblib` artifacts to `models_trained/`.

## Tests

```bash
python -m pytest tests/ -q
```

107 tests covering ingestion, cleaning, feature engineering, and all ML modules.

## Project structure

```
wec_analytics/
  ingestion/       fetch, cache, parse Al Kamel CSVs
  analysis/        outlier detection, stint assignment, traffic flagging
  ml/
    features.py    build_lap_features, build_stint_features
    pace.py        train_pace_model, predict_pace_session
    pit_window.py  train_pit_model, predict_pit_curve
    degradation.py fit_all_stints, enrich_with_deg_slope
    clustering.py  cluster_strategies (KMeans / DBSCAN)
    anomaly.py     detect_lap_anomalies (IsolationForest / LOF)
    association.py mine_strategy_rules (Apriori)
    reduction.py   reduce_to_2d (PCA / UMAP)
    evaluation.py  GroupKFold CV, baseline comparison
scripts/
  train_models.py  full training run
tests/             pytest suite (107 tests)
docs/              portfolio page (GitHub Pages)
app.py             Streamlit dashboard
```

## Core API

```python
from wec_analytics.ingestion.alkamelsystems import fetch_session
from wec_analytics.ingestion.models import clean_session
from wec_analytics.analysis.stints import detect_outliers
from wec_analytics.analysis.laps import detect_traffic_lap
from wec_analytics.ml.features import build_lap_features
from wec_analytics.ml.degradation import enrich_with_deg_slope
from wec_analytics.ml.pace import train_pace_model, predict_pace_session
from wec_analytics.ml.pit_window import train_pit_model, predict_pit_curve
from wec_analytics.ml.anomaly import detect_lap_anomalies
from wec_analytics.ml.association import mine_strategy_rules
from wec_analytics.ml.reduction import reduce_to_2d

url = "http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV"

laps = build_lap_features(
    detect_traffic_lap(
        detect_outliers(
            clean_session(fetch_session(url))
        )
    )
)
laps = enrich_with_deg_slope(laps)

pace_model = train_pace_model(
    laps[~laps[["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]].any(axis=1)]
)
annotated = predict_pace_session(pace_model, laps)

rules  = mine_strategy_rules(laps, min_support=0.05, min_confidence=0.6)
coords, _ = reduce_to_2d(laps[["class_pace_delta", "stint_age"]], method="umap")
```

## Data notice

This project does not redistribute any timing data. Session CSV URLs must be supplied by the user and are intended for personal and research use only.
