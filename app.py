"""
wec-analytics Streamlit app.

Three views:
  Race Overview    -- lap time scatter by stint, class pace comparison
  Pace Residuals   -- predicted vs actual pace per car, residual trace
  Pit Probability  -- model pit probability curve with actual pit markers
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from wec_analytics.analysis.laps import detect_traffic_lap
from wec_analytics.analysis.stints import detect_outliers
from wec_analytics.ingestion.alkamelsystems import extract_session_id, fetch_session
from wec_analytics.ingestion.models import clean_session
from wec_analytics.ingestion.sessions import SESSION_MAP
from wec_analytics.ml.features import build_lap_features, enrich_with_norm_stint_age
from wec_analytics.ml.clustering import MIN_CARS_TO_CLUSTER, cluster_strategies
from wec_analytics.ml.anomaly import (
    DEFAULT_CONTAMINATION,
    VALID_METHODS,
    compare_with_iqr,
    detect_lap_anomalies,
)
from wec_analytics.ml.association import mine_strategy_rules
from wec_analytics.ml.reduction import explained_variance_ratio, reduce_to_2d
from wec_analytics.ml.degradation import MIN_STINT_LAPS, compare_degradation, enrich_with_deg_slope, fit_all_stints
from wec_analytics.ml.features import LAP_CLEAN_FLAGS
from wec_analytics.ml.pace import predict_pace_session
from wec_analytics.ml.persistence import load_model
from wec_analytics.ml.pit_window import predict_pit_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_URLS = {
    # 2018-2019 Super Season
    "SPA_2019_RACE":         "http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "FUJI_2018_RACE":        "https://fiawec.alkamelsystems.com/Results/08_2018-2019/04_FUJI%20SPEEDWAY/246_FIA%20WEC/201810141100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SHANGHAI_2018_RACE":    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/05_SHANGHAI%20INTERNATIONAL%20CIRCUIT/256_FIA%20WEC/201811181100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SILVERSTONE_2018_RACE": "https://fiawec.alkamelsystems.com/Results/08_2018-2019/03_SILVERSTONE/239_FIA%20WEC/201808191200_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SEBRING_2019_RACE":     "https://fiawec.alkamelsystems.com/Results/08_2018-2019/06_SEBRING/260_FIA%20WEC/201903151600_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
    "LE_MANS_2019_RACE":     "https://fiawec.alkamelsystems.com/Results/08_2018-2019/08_LE%20MANS/276_FIA%20WEC/201906151500_Race/24_Hour%2024/23_Analysis_Race_Hour%2024.CSV",
    # 2019-2020 season
    "SILVERSTONE_2019_RACE": "https://fiawec.alkamelsystems.com/Results/09_2019-2020/01_SILVERSTONE/285_FIA%20WEC/201909011200_Race/Hour%204/23_Analysis_Race_Hour%204.CSV",
    "FUJI_2019_RACE":        "https://fiawec.alkamelsystems.com/Results/09_2019-2020/02_FUJI%20SPEEDWAY/294_FIA%20WEC/201910061100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SHANGHAI_2019_RACE":    "https://fiawec.alkamelsystems.com/Results/09_2019-2020/03_SHANGHAI%20INTERNATIONAL%20CIRCUIT/01_FIA%20WEC/201911101200_Race/Hour%204/23_Analysis_Race_Hour%204.CSV",
    "BAHRAIN_2019_RACE":     "https://fiawec.alkamelsystems.com/Results/09_2019-2020/04_BAHRAIN%20INTERNATIONAL%20CIRCUIT/307_FIA%20WEC/201912141500_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
    "COTA_2020_RACE":        "https://fiawec.alkamelsystems.com/Results/09_2019-2020/05_CIRCUIT%20OF%20THE%20AMERICAS/311_FIA%20WEC/202002231200_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SPA_2020_RACE":         "https://fiawec.alkamelsystems.com/Results/09_2019-2020/06_SPA%20FRANCORCHAMPS/327_FIA%20WEC/202008151330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "LE_MANS_2020_RACE":     "https://fiawec.alkamelsystems.com/Results/09_2019-2020/07_LE%20MANS/335_FIA%20WEC/202009191430_Race/Hour%2024/23_Analysis_Race_Hour%2024.CSV",
    "BAHRAIN_2020_RACE":     "https://fiawec.alkamelsystems.com/Results/09_2019-2020/08_BAHRAIN%20INTERNATIONAL%20CIRCUIT/348_FIA%20WEC/202011141400_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
}

RACE_LABELS = {
    # 2018-2019 Super Season
    "SPA_2019_RACE":         "Spa 2019 (6h)",
    "FUJI_2018_RACE":        "Fuji 2018 (6h)",
    "SHANGHAI_2018_RACE":    "Shanghai 2018 (6h)",
    "SILVERSTONE_2018_RACE": "Silverstone 2018 (6h)",
    "SEBRING_2019_RACE":     "Sebring 2019 (8h)",
    "LE_MANS_2019_RACE":     "Le Mans 2019 (24h)",
    # 2019-2020 season
    "SILVERSTONE_2019_RACE": "Silverstone 2019 (4h)",
    "FUJI_2019_RACE":        "Fuji 2019 (6h)",
    "SHANGHAI_2019_RACE":    "Shanghai 2019 (4h)",
    "BAHRAIN_2019_RACE":     "Bahrain 2019 (8h)",
    "COTA_2020_RACE":        "COTA 2020 (6h)",
    "SPA_2020_RACE":         "Spa 2020 (6h)",
    "LE_MANS_2020_RACE":     "Le Mans 2020 (24h)",
    "BAHRAIN_2020_RACE":     "Bahrain 2020 (8h)",
}

PIT_FEATURE_COLUMNS = ["stint_age", "norm_stint_age", "pace_trend", "rolling_pace", "lap_number", "car_class"]

MODELS_DIR = Path("models_trained")


# ---------------------------------------------------------------------------
# Model and data loaders
# ---------------------------------------------------------------------------

def _find_latest(pattern: str) -> Path | None:
    """Return the most recently created file matching a glob pattern."""
    matches = sorted(MODELS_DIR.glob(pattern))
    return matches[-1] if matches else None


@st.cache_resource
def get_pace_model():
    path = _find_latest("*_pace_linear.joblib")
    if path is None:
        return None, None
    model, meta = load_model(path)
    return model, meta


@st.cache_resource
def get_pit_model():
    path = _find_latest("*_pit_logistic.joblib")
    if path is None:
        return None, None
    model, meta = load_model(path)
    return model, meta


@st.cache_data(show_spinner="Computing degradation curves...")
def get_degradation_summary(_laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    return compare_degradation(_laps)


@st.cache_data(show_spinner="Analysing race strategies...")
def get_strategy_clusters(_laps: pd.DataFrame, method: str, n_clusters, projection: str, race_id: str) -> tuple:
    return cluster_strategies(_laps, n_clusters=n_clusters, method=method, projection=projection)


@st.cache_data(show_spinner="Projecting features to 2D...")
def get_2d_projection(_df: pd.DataFrame, feature_cols: tuple, method: str, data_key: str) -> pd.DataFrame:
    feat = _df[list(feature_cols)].fillna(0.0)
    coords, transformer = reduce_to_2d(feat, method=method)
    ev = explained_variance_ratio(transformer)
    coords["ev_x"] = float(ev[0]) if ev is not None else None
    coords["ev_y"] = float(ev[1]) if ev is not None else None
    return coords


@st.cache_data(show_spinner="Running anomaly detection...")
def get_anomaly_scores(_laps: pd.DataFrame, method: str, contamination: float, race_id: str) -> pd.DataFrame:
    return detect_lap_anomalies(_laps, method=method, contamination=contamination)


@st.cache_data(show_spinner="Comparing anomaly methods...")
def get_iqr_comparison(_laps: pd.DataFrame, method: str, contamination: float, race_id: str) -> pd.DataFrame:
    return compare_with_iqr(_laps, method=method, contamination=contamination)


@st.cache_data(show_spinner="Mining strategy patterns...")
def get_strategy_rules(_laps: pd.DataFrame, min_support: float, min_confidence: float, race_id: str) -> pd.DataFrame:
    return mine_strategy_rules(_laps, min_support=min_support, min_confidence=min_confidence)


@st.cache_data(show_spinner="Loading session data...")
def get_session(race_id: str) -> pd.DataFrame:
    url = SESSION_URLS[race_id]
    session_id = extract_session_id(url)
    meta = SESSION_MAP[session_id]

    raw = fetch_session(url)
    cleaned = clean_session(raw)
    with_outliers = detect_outliers(cleaned)
    with_traffic = detect_traffic_lap(with_outliers)
    featured = enrich_with_norm_stint_age(build_lap_features(with_traffic))
    featured["race_id"] = meta.race_id
    return enrich_with_deg_slope(featured)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WEC Analytics",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("WEC Analytics")
    st.caption("FIA WEC 2018-2019 Super Season")
    st.divider()

    selected_race_id = st.selectbox(
        "Race",
        options=list(RACE_LABELS.keys()),
        format_func=lambda k: RACE_LABELS[k],
    )

    laps = get_session(selected_race_id)

    classes = sorted(laps["car_class"].dropna().unique())
    selected_classes = st.multiselect(
        "Car class filter",
        options=classes,
        default=classes,
    )

    cars_in_class = sorted(
        laps[laps["car_class"].isin(selected_classes)]["car_number"].unique()
    )
    selected_car = st.selectbox("Car (for detail views)", options=cars_in_class)

    st.divider()
    pace_model, pace_meta = get_pace_model()
    pit_model, pit_meta = get_pit_model()

    if pace_model is None:
        st.warning("No pace model found. Run `python scripts/train_models.py` first.")
    else:
        st.success("Pace model loaded")

    if pit_model is None:
        st.warning("No pit model found. Run `python scripts/train_models.py` first.")
    else:
        st.success("Pit model loaded")

# ---------------------------------------------------------------------------
# Filtered views
# ---------------------------------------------------------------------------

laps_filtered = laps[laps["car_class"].isin(selected_classes)].copy()
car_laps = laps[laps["car_number"] == selected_car].copy()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_pace, tab_pit, tab_deg, tab_cluster, tab_anomaly, tab_rules = st.tabs(
    ["Race Overview", "Pace Residuals", "Pit Probability", "Tyre Degradation", "Strategy Clusters", "Anomaly Detection", "Strategy Patterns"]
)

# ── Tab 1: Race Overview ────────────────────────────────────────────────────

with tab_overview:
    st.subheader(f"{RACE_LABELS[selected_race_id]}: Race Overview")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total laps", f"{len(laps_filtered):,}")
    col_b.metric("Cars", laps_filtered["car_number"].nunique())
    col_c.metric(
        "Fastest lap",
        f"{laps_filtered['lap_time'].min():.3f}s",
    )
    col_d.metric(
        "Pit stops",
        int(laps_filtered["is_in_lap"].sum()),
    )

    st.divider()

    # Lap time scatter -- one point per clean lap, coloured by class
    clean = laps_filtered[
        ~laps_filtered[LAP_CLEAN_FLAGS].any(axis=1)
    ]

    fig_scatter = px.scatter(
        clean,
        x="lap_number",
        y="lap_time",
        color="car_class",
        hover_data=["car_number", "stint_age"],
        labels={"lap_number": "Lap", "lap_time": "Lap time (s)", "car_class": "Class"},
        title="Clean lap times by lap number",
        height=420,
        opacity=0.55,
    )
    fig_scatter.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Stint degradation for the selected car
    st.subheader(f"Car #{selected_car}: stint degradation")

    car_clean = car_laps[
        ~car_laps[LAP_CLEAN_FLAGS].any(axis=1)
    ].copy()

    if car_clean.empty:
        st.info("No clean laps for this car.")
    else:
        fig_deg = px.scatter(
            car_clean,
            x="stint_age",
            y="lap_time",
            color="stint_id",
            labels={
                "stint_age": "Lap in stint",
                "lap_time": "Lap time (s)",
                "stint_id": "Stint",
            },
            title=f"Car #{selected_car}: lap time vs stint age",
            height=380,
        )
        fig_deg.update_traces(marker=dict(size=6))
        st.plotly_chart(fig_deg, use_container_width=True)

# ── Tab 2: Pace Residuals ───────────────────────────────────────────────────

with tab_pace:
    st.subheader(f"Car #{selected_car}: Pace Residuals")

    if pace_model is None:
        st.error("Train a pace model first: `python scripts/train_models.py`")
    else:
        annotated = predict_pace_session(pace_model, car_laps)

        valid = annotated.dropna(subset=["predicted_pace", "pace_residual"])

        if valid.empty:
            st.info("Not enough laps to generate predictions (rolling_pace may be NaN).")
        else:
            col1, col2 = st.columns(2)
            col1.metric(
                "Mean residual",
                f"{valid['pace_residual'].mean():+.3f}s",
                help="Positive = slower than model expected",
            )
            col2.metric(
                "Residual std",
                f"{valid['pace_residual'].std():.3f}s",
            )

            # Actual vs predicted
            fig_pace = go.Figure()
            fig_pace.add_trace(go.Scatter(
                x=valid["lap_number"],
                y=valid["lap_time"],
                mode="markers",
                name="Actual",
                marker=dict(size=5, opacity=0.7),
            ))
            fig_pace.add_trace(go.Scatter(
                x=valid["lap_number"],
                y=valid["predicted_pace"],
                mode="lines",
                name="Predicted",
                line=dict(width=2, dash="dot"),
            ))
            fig_pace.update_layout(
                title=f"Car #{selected_car}: actual vs predicted lap time",
                xaxis_title="Lap",
                yaxis_title="Lap time (s)",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_pace, use_container_width=True)

            # Residual trace
            residual_colours = valid["pace_residual"].apply(
                lambda r: "red" if r > 0 else "green"
            )
            fig_res = go.Figure()
            fig_res.add_hline(y=0, line_dash="dash", line_color="grey")
            fig_res.add_trace(go.Bar(
                x=valid["lap_number"],
                y=valid["pace_residual"],
                marker_color=residual_colours,
                name="Residual",
            ))
            fig_res.update_layout(
                title="Pace residual (actual - predicted)  |  red = slower than model",
                xaxis_title="Lap",
                yaxis_title="Residual (s)",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_res, use_container_width=True)

# ── Tab 3: Pit Probability ──────────────────────────────────────────────────

with tab_pit:
    st.subheader(f"Car #{selected_car}: Pit Probability")

    if pit_model is None:
        st.error("Train a pit model first: `python scripts/train_models.py`")
    else:
        pit_laps = car_laps.dropna(subset=PIT_FEATURE_COLUMNS).copy()

        if pit_laps.empty:
            st.info("No laps with complete features for this car.")
        else:
            threshold = st.slider(
                "Pit probability threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Horizontal line showing where the model would call a pit stop.",
            )

            try:
                proba = predict_pit_curve(pit_model, pit_laps, PIT_FEATURE_COLUMNS)
            except Exception as exc:
                st.error(
                    f"Pit model is incompatible with the installed sklearn version: {exc}. "
                    "Re-run `python scripts/train_models.py` to retrain."
                )
                proba = None

            if proba is not None:
                pit_laps = pit_laps.copy()
                pit_laps["pit_probability"] = proba.values

                actual_pit_laps = pit_laps[pit_laps["is_in_lap"]]["lap_number"].tolist()

                fig_pit = go.Figure()

                # Probability curve
                fig_pit.add_trace(go.Scatter(
                    x=pit_laps["lap_number"],
                    y=pit_laps["pit_probability"],
                    mode="lines",
                    name="Pit probability",
                    line=dict(width=2),
                ))

                # Threshold line
                fig_pit.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Threshold {threshold:.2f}",
                    annotation_position="top right",
                )

                # Actual pit laps as vertical lines
                for lap in actual_pit_laps:
                    fig_pit.add_vline(
                        x=lap,
                        line_dash="dot",
                        line_color="red",
                        opacity=0.6,
                    )

                # Predicted pit laps (probability > threshold) as green markers
                predicted_pit_laps = pit_laps[pit_laps["pit_probability"] >= threshold]["lap_number"].tolist()
                if predicted_pit_laps:
                    fig_pit.add_trace(go.Scatter(
                        x=predicted_pit_laps,
                        y=[threshold] * len(predicted_pit_laps),
                        mode="markers",
                        marker=dict(color="green", size=8, symbol="triangle-up"),
                        name="Predicted pit",
                        showlegend=True,
                    ))

                # Legend entry for actual pits
                if actual_pit_laps:
                    fig_pit.add_trace(go.Scatter(
                        x=[actual_pit_laps[0]],
                        y=[0],
                        mode="markers",
                        marker=dict(color="red", size=8, symbol="line-ns", line_width=2),
                        name="Actual pit",
                        showlegend=True,
                    ))

                fig_pit.update_layout(
                    title=f"Car #{selected_car}: pit probability by lap",
                    xaxis_title="Lap",
                    yaxis_title="P(pit this lap)",
                    yaxis=dict(range=[0, 1]),
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_pit, use_container_width=True)

                # Summary stats for threshold
                actual_set = set(actual_pit_laps)
                predicted_set = set(predicted_pit_laps)
                true_positives = len(actual_set & predicted_set)
                recall = true_positives / len(actual_set) if actual_set else 0.0
                precision = true_positives / len(predicted_set) if predicted_set else 0.0

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted pit laps", len(predicted_set))
                m2.metric("Actual stops caught", f"{true_positives} / {len(actual_set)}", f"Recall {recall:.0%}")
                m3.metric("Precision", f"{precision:.0%}", help="Of flagged laps, how many were real pit laps")

                st.caption(
                    f"Red dotted lines = actual pit stops. "
                    f"Green triangles = laps where model exceeds threshold. "
                    f"Adjust the slider to trade off recall vs false alarms."
                )

# -- Tab 4: Tyre Degradation -------------------------------------------------

with tab_deg:
    st.subheader(f"Car #{selected_car}: Tyre Degradation")

    deg_summary = get_degradation_summary(laps, selected_race_id)

    if deg_summary.empty:
        st.info("No stints with enough clean laps to fit a degradation curve.")
    else:
        # Per-car selected car curves
        car_stints = fit_all_stints(car_laps)

        if car_stints.empty:
            st.info(f"Car #{selected_car} has no stints with {MIN_STINT_LAPS}+ clean laps.")
        else:
            clean_car = car_laps[
                ~car_laps[LAP_CLEAN_FLAGS].any(axis=1)
            ].copy()

            fig_curves = go.Figure()

            for _, row in car_stints.iterrows():
                stint_laps = clean_car[clean_car["stint_id"] == row["stint_id"]]
                x_pts = stint_laps["stint_age"].to_numpy(dtype=float)
                y_pts = stint_laps["lap_time"].to_numpy(dtype=float)

                x_fit = np.linspace(x_pts.min(), x_pts.max(), 60)
                if row["best_model"] == "linear":
                    y_fit = row["linear_slope"] * x_fit + row["linear_intercept"]
                    slope_label = f"{row['linear_slope']:+.3f} s/lap"
                else:
                    y_fit = row["quadratic_a"] * x_fit ** 2 + row["quadratic_b"] * x_fit + row["quadratic_c"]
                    slope_label = f"quad b={row['quadratic_b']:+.3f}"

                colour = px.colors.qualitative.Plotly[int(row["stint_id"]) % 10]

                fig_curves.add_trace(go.Scatter(
                    x=x_pts, y=y_pts, mode="markers",
                    marker=dict(size=6, color=colour, opacity=0.6),
                    name=f"Stint {int(row['stint_id'])}",
                    legendgroup=f"stint_{int(row['stint_id'])}",
                    showlegend=True,
                ))
                fig_curves.add_trace(go.Scatter(
                    x=x_fit, y=y_fit, mode="lines",
                    line=dict(width=2, color=colour),
                    name=f"Stint {int(row['stint_id'])}: {slope_label}",
                    legendgroup=f"stint_{int(row['stint_id'])}",
                    showlegend=True,
                ))

            fig_curves.update_layout(
                title=f"Car #{selected_car}: lap time vs stint position with fitted curves",
                xaxis_title="Lap in stint",
                yaxis_title="Lap time (s)",
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_curves, use_container_width=True)

        # Session degradation ranking
        st.subheader(f"{RACE_LABELS[selected_race_id]}: Degradation ranking")

        CLASS_COLOURS = {"Hypercar": "#1f77b4", "LMP2": "#ff7f0e", "LMGT3": "#2ca02c"}
        bar_colours = [
            "#e74c3c" if row["car_number"] == selected_car
            else CLASS_COLOURS.get(row["car_class"], "#aec7e8")
            for _, row in deg_summary.iterrows()
        ]

        fig_rank = go.Figure(go.Bar(
            x=deg_summary["deg_slope_mean"],
            y=deg_summary["car_number"].astype(str),
            orientation="h",
            marker_color=bar_colours,
            text=deg_summary["deg_slope_mean"].apply(lambda v: f"{v:+.3f} s/lap"),
            textposition="outside",
        ))
        fig_rank.update_layout(
            title="Mean degradation slope per car (lower = less tyre wear)",
            xaxis_title="Degradation slope (s/lap)",
            yaxis_title="Car",
            height=max(300, 30 * len(deg_summary)),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        st.caption(
            f"Red bar = selected car #{selected_car}. "
            f"Slope from {MIN_STINT_LAPS}+ clean laps per stint. "
            f"Model selected per stint by BIC (linear or quadratic)."
        )

# -- Tab 5: Strategy Clusters ------------------------------------------------

with tab_cluster:
    st.subheader(f"{RACE_LABELS[selected_race_id]}: Strategy Clusters")

    col_m, col_k, col_proj = st.columns([1, 2, 1])
    with col_m:
        cluster_method = st.radio("Clustering", ["kmeans", "dbscan"], horizontal=True)
    with col_k:
        if cluster_method == "kmeans":
            auto_k = st.checkbox("Auto-select k (silhouette)", value=True)
            n_clusters_input = "auto" if auto_k else st.slider("k", 2, 8, 3)
        else:
            n_clusters_input = "auto"
            st.caption("DBSCAN auto-tunes epsilon from k-NN distances. Cluster -1 = outlier cars.")
    with col_proj:
        projection_method = st.radio("Projection", ["PCA", "UMAP"], horizontal=True,
                                     help="PCA: linear, fast. UMAP: preserves local structure better.")
    cluster_projection = projection_method.lower()

    n_cars = laps["car_number"].nunique()
    if n_cars < MIN_CARS_TO_CLUSTER:
        st.info(f"Need at least {MIN_CARS_TO_CLUSTER} cars to cluster. This session has {n_cars}.")
    else:
        try:
            labels_df, meta = get_strategy_clusters(laps, cluster_method, n_clusters_input, cluster_projection, selected_race_id)
        except Exception as exc:
            st.error(f"Clustering failed: {exc}")
            labels_df = None
            meta = None

        if labels_df is not None:
            pca_plot = meta["pca_coords"].merge(
                labels_df[["car_number", "car_class", "cluster_label",
                            "n_stints", "stint_length_mean", "deg_slope_mean"]],
                on="car_number",
            )
            pca_plot["is_selected"] = pca_plot["car_number"] == selected_car
            pca_plot["marker_size"] = pca_plot["is_selected"].map({True: 16, False: 8})
            pca_plot["cluster_str"] = pca_plot["cluster_label"].astype(str)

            ev = meta["pca_explained_variance"]
            if ev is not None:
                x_label = f"PC1 ({ev[0]:.0%} var)"
                y_label = f"PC2 ({ev[1]:.0%} var)"
            else:
                x_label = "UMAP-1"
                y_label = "UMAP-2"

            fig_pca = px.scatter(
                pca_plot,
                x="PC1", y="PC2",
                color="cluster_str",
                size="marker_size",
                hover_data={
                    "car_number": True,
                    "car_class": True,
                    "n_stints": True,
                    "stint_length_mean": ":.1f",
                    "deg_slope_mean": ":.3f",
                    "cluster_str": False,
                    "marker_size": False,
                    "is_selected": False,
                },
                labels={"cluster_str": "Cluster", "PC1": x_label, "PC2": y_label},
                title=f"{cluster_projection.upper()} projection of strategy features ({meta['n_clusters']} clusters, clustering={cluster_method})",
                height=460,
            )
            fig_pca.update_traces(marker=dict(line=dict(width=1, color="white")))
            st.plotly_chart(fig_pca, use_container_width=True)

            # silhouette vs k chart for KMeans auto mode
            if cluster_method == "kmeans" and meta["k_scores"]:
                k_df = pd.DataFrame([
                    {"k": k, "Silhouette": s} for k, s in meta["k_scores"].items()
                ])
                chosen_k = meta["n_clusters"]
                fig_sil = px.line(
                    k_df, x="k", y="Silhouette",
                    markers=True,
                    title="Silhouette score by k (higher = better-separated clusters)",
                    height=260,
                )
                fig_sil.add_vline(
                    x=chosen_k, line_dash="dash", line_color="orange",
                    annotation_text=f"chosen k={chosen_k}",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_sil, use_container_width=True)

            # Cluster summary table
            summary_cols = ["cluster_label"] + [c for c in labels_df.columns if c in [
                "n_stints", "stint_length_mean", "stint_length_std",
                "class_delta_mean", "consistency_mean", "deg_slope_mean",
            ]]
            cluster_summary = (
                labels_df[summary_cols]
                .groupby("cluster_label")
                .mean()
                .round(3)
                .reset_index()
                .rename(columns={"cluster_label": "Cluster"})
            )
            st.caption("Mean feature values per cluster (interpret to name each archetype):")
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

            if meta["silhouette"] is not None:
                ev_note = (
                    f" PCA explains {ev.sum():.0%} of variance." if ev is not None
                    else " UMAP does not expose explained variance."
                )
                st.caption(
                    f"Silhouette score: {meta['silhouette']:.3f} "
                    f"(0 = overlapping, 1 = perfectly separated)." + ev_note
                )

# -- Tab 6: Anomaly Detection ----------------------------------------------------

with tab_anomaly:
    st.subheader("Lap Anomaly Detection")

    col_m, col_c = st.columns([1, 1])
    with col_m:
        anomaly_method_label = st.radio(
            "Method",
            options=["Isolation Forest", "Local Outlier Factor (LOF)"],
            horizontal=True,
            key="anomaly_method",
        )
    with col_c:
        anomaly_contamination = st.slider(
            "Contamination (expected anomaly rate)",
            min_value=0.01,
            max_value=0.20,
            value=float(DEFAULT_CONTAMINATION),
            step=0.01,
            format="%.2f",
            key="anomaly_contamination",
        )

    anomaly_method = "isolation_forest" if "Isolation" in anomaly_method_label else "lof"

    annotated_laps = get_anomaly_scores(laps, anomaly_method, anomaly_contamination, selected_race_id)

    st.divider()

    # Scatter: all laps coloured by anomaly score
    st.subheader("Anomaly score by lap")
    fig_anomaly = px.scatter(
        annotated_laps,
        x="lap_number",
        y="lap_time",
        color="anomaly_score",
        color_continuous_scale="Reds",
        hover_data={
            "car_number": True,
            "car_class": True,
            "stint_age": True,
            "is_lap_anomaly": True,
            "anomaly_score": ":.3f",
        },
        labels={
            "lap_number": "Lap",
            "lap_time": "Lap time (s)",
            "anomaly_score": "Anomaly score",
        },
        title=f"All laps coloured by anomaly score ({anomaly_method_label}, contamination={anomaly_contamination:.0%})",
        height=420,
        opacity=0.65,
    )
    fig_anomaly.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_anomaly, use_container_width=True)

    # Top anomalous laps table
    top_n = 20
    top_anomalous = (
        annotated_laps[annotated_laps["is_lap_anomaly"]]
        .sort_values("anomaly_score", ascending=False)
        .head(top_n)[
            ["car_number", "car_class", "lap_number", "stint_age",
             "lap_time", "class_pace_delta", "is_traffic_lap",
             "is_in_lap", "is_out_lap", "anomaly_score"]
        ]
        .copy()
    )
    top_anomalous["anomaly_score"] = top_anomalous["anomaly_score"].round(4)
    top_anomalous["class_pace_delta"] = top_anomalous["class_pace_delta"].round(3)
    top_anomalous["lap_time"] = top_anomalous["lap_time"].round(3)

    st.caption(f"Top {min(top_n, len(top_anomalous))} highest-scoring flagged laps:")
    st.dataframe(top_anomalous, use_container_width=True, hide_index=True)

    st.divider()

    # IQR comparison
    st.subheader("Agreement with Phase 1 IQR detector")
    st.caption(
        "IQR operates on lap time only. Multi-feature ML catches anomalies "
        "normal in time but unusual in context (ml_only). IQR catches clear "
        "safety-car laps that blend into the bulk distribution (iqr_only)."
    )

    comparison = get_iqr_comparison(laps, anomaly_method, anomaly_contamination, selected_race_id)

    label_map = {
        "both_flagged": "Both (IQR + ML)",
        "iqr_only": "IQR only",
        "ml_only": "ML only",
        "neither_flagged": "Neither",
    }
    comparison["label"] = comparison["agreement_type"].map(label_map)

    fig_compare = px.bar(
        comparison,
        x="label",
        y="n_laps",
        color="label",
        color_discrete_map={
            "Both (IQR + ML)": "#e74c3c",
            "IQR only": "#e67e22",
            "ML only": "#9b59b6",
            "Neither": "#2ecc71",
        },
        text="pct_laps",
        labels={"label": "Agreement type", "n_laps": "Number of laps"},
        title="IQR vs ML anomaly flag agreement",
        height=340,
    )
    fig_compare.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_compare.update_layout(showlegend=False)
    st.plotly_chart(fig_compare, use_container_width=True)

    ml_only_count = int(comparison.loc[comparison["agreement_type"] == "ml_only", "n_laps"].iloc[0])
    iqr_only_count = int(comparison.loc[comparison["agreement_type"] == "iqr_only", "n_laps"].iloc[0])
    both_count = int(comparison.loc[comparison["agreement_type"] == "both_flagged", "n_laps"].iloc[0])

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Agreement (both)", both_count)
    col_s2.metric("IQR misses (ML only)", ml_only_count,
                  help="Laps the multi-feature model flags that IQR considers clean")
    col_s3.metric("ML misses (IQR only)", iqr_only_count,
                  help="Laps IQR flags that the ML model scores as normal")

    st.divider()

    # 2D feature projection coloured by anomaly score
    st.subheader("2D Feature Projection")
    st.caption(
        "Projects the anomaly feature space to 2D. Points coloured by anomaly score: "
        "dark red = most anomalous. PCA is linear and fast; UMAP preserves local neighbourhood structure."
    )

    proj_col, _ = st.columns([1, 3])
    with proj_col:
        anomaly_proj_method = st.radio(
            "Projection method",
            ["PCA", "UMAP"],
            horizontal=True,
            key="anomaly_proj",
            help="UMAP may be slow on large sessions.",
        )

    _ANOMALY_FEAT_COLS = tuple(
        c for c in ["class_pace_delta", "stint_age", "is_traffic_lap", "is_in_lap", "is_out_lap"]
        if c in annotated_laps.columns
    )

    if len(_ANOMALY_FEAT_COLS) >= 2:
        _proj_data_key = f"{selected_race_id}_{anomaly_method}_{anomaly_contamination}"
        proj_coords = get_2d_projection(
            annotated_laps, _ANOMALY_FEAT_COLS, anomaly_proj_method.lower(), _proj_data_key
        )
        proj_plot = annotated_laps[["car_number", "car_class", "lap_number",
                                    "lap_time", "anomaly_score", "is_lap_anomaly"]].copy()
        proj_plot["proj_x"] = proj_coords["x"].values
        proj_plot["proj_y"] = proj_coords["y"].values

        ev_x = proj_coords["ev_x"].iloc[0]
        ev_y = proj_coords["ev_y"].iloc[0]
        x_axis_label = f"PC1 ({ev_x:.0%} var)" if ev_x is not None else "UMAP-1"
        y_axis_label = f"PC2 ({ev_y:.0%} var)" if ev_y is not None else "UMAP-2"

        fig_proj = px.scatter(
            proj_plot,
            x="proj_x", y="proj_y",
            color="anomaly_score",
            color_continuous_scale="Reds",
            symbol="is_lap_anomaly",
            symbol_map={True: "x", False: "circle"},
            hover_data={
                "car_number": True,
                "car_class": True,
                "lap_number": True,
                "lap_time": ":.3f",
                "anomaly_score": ":.3f",
                "proj_x": False,
                "proj_y": False,
            },
            labels={
                "proj_x": x_axis_label,
                "proj_y": y_axis_label,
                "anomaly_score": "Anomaly score",
                "is_lap_anomaly": "Flagged",
            },
            title=f"{anomaly_proj_method} of anomaly features — flagged laps marked ×",
            height=440,
            opacity=0.65,
        )
        fig_proj.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_proj, use_container_width=True)

# -- Tab 7: Strategy Patterns (Association Rules) ----------------------------

with tab_rules:
    st.subheader(f"{RACE_LABELS[selected_race_id]}: Strategy Patterns")
    st.caption(
        "Association rules mined from 15-minute race windows. "
        "Each window is a transaction listing events: pit stop, SC period, traffic, driver change. "
        "Rules like {sc_period} -> {next_pit} reveal strategy responses with confidence and lift."
    )

    col_sup, col_conf = st.columns(2)
    with col_sup:
        min_support = st.slider(
            "Min support",
            min_value=0.01,
            max_value=0.30,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Fraction of windows the itemset must appear in.",
        )
    with col_conf:
        min_confidence = st.slider(
            "Min confidence",
            min_value=0.30,
            max_value=0.95,
            value=0.60,
            step=0.05,
            format="%.2f",
            help="P(consequent | antecedent). Higher = more reliable rule.",
        )

    rules = get_strategy_rules(laps, min_support, min_confidence, selected_race_id)

    if rules.empty:
        st.info("No rules found at these thresholds -- try lowering min support.")
    else:
        def _shorten(items_str: str) -> str:
            """Strip class_ prefix and wrap long lists across two lines."""
            return items_str.replace("class_", "")

        rules = rules.copy()
        rules["ant_short"] = rules["antecedents_str"].apply(_shorten)
        rules["con_short"] = rules["consequents_str"].apply(_shorten)

        best_rule = rules.iloc[0]
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Total rules", len(rules))
        col_r2.metric("Max lift", f"{rules['lift'].max():.2f}")
        col_r3.metric(
            "Top rule (by lift)",
            f"{best_rule['ant_short']} -> {best_rule['con_short']}",
            help=f"confidence={best_rule['confidence']:.2f}, lift={best_rule['lift']:.2f}",
        )

        st.divider()

        # Top 10 rules bar chart
        top10 = rules.head(10).copy()
        top10["rule_label"] = top10["ant_short"] + "  ->  " + top10["con_short"]
        fig_rules = px.bar(
            top10.iloc[::-1],  # reverse so highest lift is at top
            x="lift",
            y="rule_label",
            orientation="h",
            color="confidence",
            color_continuous_scale="Blues",
            color_continuous_midpoint=0.75,
            range_color=[0.0, 1.0],
            labels={"lift": "Lift", "rule_label": "Rule", "confidence": "Confidence"},
            title="Top 10 rules by lift",
            height=max(340, 46 * len(top10)),
        )
        fig_rules.update_layout(
            yaxis=dict(autorange=True, tickfont=dict(size=12)),
            margin=dict(l=320),
            coloraxis_colorbar=dict(title="Conf", tickvals=[0, 0.5, 1.0]),
        )
        st.plotly_chart(fig_rules, use_container_width=True)

        # Full rules table
        display_cols = ["ant_short", "con_short", "support", "confidence", "lift"]
        display = rules.head(30)[display_cols].copy()
        display["support"] = display["support"].round(3)
        display["confidence"] = display["confidence"].round(3)
        display["lift"] = display["lift"].round(3)
        display = display.rename(columns={
            "ant_short": "Antecedents",
            "con_short": "Consequents",
        })
        st.caption(f"Top {min(30, len(rules))} rules sorted by lift:")
        st.dataframe(display, use_container_width=True, hide_index=True)
