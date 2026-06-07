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
from wec_analytics.ml.features import build_lap_features
from wec_analytics.ml.clustering import MIN_CARS_TO_CLUSTER, cluster_strategies
from wec_analytics.ml.degradation import MIN_STINT_LAPS, _fit_all_stints, compare_degradation
from wec_analytics.ml.pace import predict_pace_session
from wec_analytics.ml.persistence import load_model
from wec_analytics.ml.pit_window import predict_pit_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_URLS = {
    "SPA_2019_RACE":        "http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "FUJI_2018_RACE":       "https://fiawec.alkamelsystems.com/Results/08_2018-2019/04_FUJI%20SPEEDWAY/246_FIA%20WEC/201810141100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SHANGHAI_2018_RACE":   "https://fiawec.alkamelsystems.com/Results/08_2018-2019/05_SHANGHAI%20INTERNATIONAL%20CIRCUIT/256_FIA%20WEC/201811181100_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SILVERSTONE_2018_RACE":"https://fiawec.alkamelsystems.com/Results/08_2018-2019/03_SILVERSTONE/239_FIA%20WEC/201808191200_Race/Hour%206/23_Analysis_Race_Hour%206.CSV",
    "SEBRING_2019_RACE":    "https://fiawec.alkamelsystems.com/Results/08_2018-2019/06_SEBRING/260_FIA%20WEC/201903151600_Race/Hour%208/23_Analysis_Race_Hour%208.CSV",
}

RACE_LABELS = {
    "SPA_2019_RACE":        "Spa-Francorchamps 2019",
    "FUJI_2018_RACE":       "Fuji Speedway 2018",
    "SHANGHAI_2018_RACE":   "Shanghai 2018",
    "SILVERSTONE_2018_RACE":"Silverstone 2018",
    "SEBRING_2019_RACE":    "Sebring 2019 (8h)",
}

PIT_FEATURE_COLUMNS = ["stint_age", "rolling_pace", "lap_number", "car_class"]

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
def get_degradation_summary(_laps: pd.DataFrame) -> pd.DataFrame:
    return compare_degradation(_laps)


@st.cache_data(show_spinner="Analysing race strategies...")
def get_strategy_clusters(_laps: pd.DataFrame, method: str, n_clusters) -> tuple:
    return cluster_strategies(_laps, n_clusters=n_clusters, method=method)


@st.cache_data(show_spinner="Loading session data...")
def get_session(race_id: str) -> pd.DataFrame:
    url = SESSION_URLS[race_id]
    session_id = extract_session_id(url)
    meta = SESSION_MAP[session_id]

    raw = fetch_session(url)
    cleaned = clean_session(raw)
    with_outliers = detect_outliers(cleaned)
    with_traffic = detect_traffic_lap(with_outliers)
    featured = build_lap_features(with_traffic)
    featured["race_id"] = meta.race_id
    return featured


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

tab_overview, tab_pace, tab_pit, tab_deg, tab_cluster = st.tabs(
    ["Race Overview", "Pace Residuals", "Pit Probability", "Tyre Degradation", "Strategy Clusters"]
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
        ~laps_filtered[["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]].any(axis=1)
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
        ~car_laps[["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]].any(axis=1)
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

                st.caption(
                    f"Red dotted lines = actual pit stops (laps {actual_pit_laps}). "
                    f"Orange dashed line = probability threshold. "
                    f"A good model shows rising probability in the laps before each red line."
                )

# -- Tab 4: Tyre Degradation -------------------------------------------------

with tab_deg:
    st.subheader(f"Car #{selected_car}: Tyre Degradation")

    deg_summary = get_degradation_summary(laps)

    if deg_summary.empty:
        st.info("No stints with enough clean laps to fit a degradation curve.")
    else:
        # Per-car selected car curves
        car_stints = _fit_all_stints(car_laps)

        if car_stints.empty:
            st.info(f"Car #{selected_car} has no stints with {MIN_STINT_LAPS}+ clean laps.")
        else:
            clean_car = car_laps[
                ~car_laps[["is_outlier", "is_in_lap", "is_out_lap", "is_traffic_lap"]].any(axis=1)
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

    col_m, col_k = st.columns([1, 2])
    with col_m:
        cluster_method = st.radio("Method", ["kmeans", "dbscan"], horizontal=True)
    with col_k:
        if cluster_method == "kmeans":
            auto_k = st.checkbox("Auto-select k (silhouette)", value=True)
            n_clusters_input = "auto" if auto_k else st.slider("k", 2, 8, 3)
        else:
            n_clusters_input = "auto"
            st.caption("DBSCAN auto-tunes epsilon from k-NN distances. Cluster -1 = outlier cars.")

    n_cars = laps["car_number"].nunique()
    if n_cars < MIN_CARS_TO_CLUSTER:
        st.info(f"Need at least {MIN_CARS_TO_CLUSTER} cars to cluster. This session has {n_cars}.")
    else:
        try:
            labels_df, meta = get_strategy_clusters(laps, cluster_method, n_clusters_input)
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
                labels={"cluster_str": "Cluster", "PC1": f"PC1 ({ev[0]:.0%})", "PC2": f"PC2 ({ev[1]:.0%})"},
                title=f"PCA projection of strategy features ({meta['n_clusters']} clusters, method={cluster_method})",
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
                st.caption(
                    f"Silhouette score: {meta['silhouette']:.3f} "
                    f"(0 = overlapping, 1 = perfectly separated). "
                    f"PCA explains {ev.sum():.0%} of variance."
                )
