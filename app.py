"""
wec-analytics Streamlit app.

Three views:
  Race Overview    -- lap time scatter by stint, class pace comparison
  Pace Residuals   -- predicted vs actual pace per car, residual trace
  Pit Probability  -- model pit probability curve with actual pit markers
"""

from pathlib import Path

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
    page_icon=":racing_car:",
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

tab_overview, tab_pace, tab_pit = st.tabs(
    ["Race Overview", "Pace Residuals", "Pit Probability"]
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
        annotated = predict_pace_session(
            _find_latest("*_pace_linear.joblib"),
            car_laps,
        )

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

            proba = predict_pit_curve(pit_model, pit_laps, PIT_FEATURE_COLUMNS)
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
