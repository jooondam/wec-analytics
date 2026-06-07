"""
Tyre degradation curve fitting for wec_analytics ML layer.

Fits linear and quadratic polynomials to lap time vs stint position for each
clean stint, selects the better model by BIC, and aggregates per-car summaries
for session-level comparison.

The degradation slope (seconds per lap) is the headline number race engineers
quote when discussing tyre behaviour. A positive slope means the car is getting
slower as the stint progresses; a negative slope (rare) suggests improving
conditions or a very short warm-up window captured in the data.
"""

import numpy as np
import pandas as pd

from wec_analytics.ml.features import LAP_CLEAN_FLAGS

MIN_STINT_LAPS = 5
CLEAN_LAP_FLAGS = LAP_CLEAN_FLAGS


def fit_degradation_curve(stint: pd.DataFrame) -> dict:
    """Fit linear and quadratic degradation curves to a single clean stint.

    Parameters
    ----------
    stint : pd.DataFrame
        Rows for one (car_number, stint_id) group, already filtered to clean
        laps. Must contain columns ``stint_age`` and ``lap_time``.

    Returns
    -------
    dict
        Empty dict when the stint has fewer than MIN_STINT_LAPS rows.
        Otherwise a dict with keys:

        - ``n_laps`` -- number of laps fitted
        - ``linear_slope``, ``linear_intercept`` -- coefficients
        - ``linear_r2``, ``linear_aic``, ``linear_bic`` -- fit quality
        - ``quadratic_a``, ``quadratic_b``, ``quadratic_c`` -- coefficients
        - ``quadratic_r2``, ``quadratic_aic``, ``quadratic_bic``
        - ``best_model`` -- ``"linear"`` or ``"quadratic"`` (lower BIC wins)
        - ``deg_slope`` -- headline degradation rate in seconds per lap
          (``linear_slope`` when best is linear; ``quadratic_b`` when quadratic)

    Raises
    ------
    KeyError
        If ``stint_age`` or ``lap_time`` are missing from the input.
    """
    required = ["stint_age", "lap_time"]
    missing = [c for c in required if c not in stint.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    if len(stint) < MIN_STINT_LAPS:
        return {}

    x = stint["stint_age"].to_numpy(dtype=float)
    y = stint["lap_time"].to_numpy(dtype=float)
    n = len(x)

    def _metrics(coeffs: np.ndarray, degree: int) -> tuple[float, float, float]:
        y_hat = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        k = degree + 1
        # log-likelihood approximation: assumes Gaussian errors
        log_rss_n = np.log(ss_res / n) if ss_res > 0 else -np.inf
        aic = float(n * log_rss_n + 2 * k)
        bic = float(n * log_rss_n + k * np.log(n))
        return r2, aic, bic

    lin = np.polyfit(x, y, 1)
    lin_r2, lin_aic, lin_bic = _metrics(lin, 1)

    quad = np.polyfit(x, y, 2)
    quad_r2, quad_aic, quad_bic = _metrics(quad, 2)

    best = "linear" if lin_bic <= quad_bic else "quadratic"
    deg_slope = float(lin[0]) if best == "linear" else float(quad[1])

    return {
        "n_laps": n,
        "linear_slope": float(lin[0]),
        "linear_intercept": float(lin[1]),
        "linear_r2": lin_r2,
        "linear_aic": lin_aic,
        "linear_bic": lin_bic,
        "quadratic_a": float(quad[0]),
        "quadratic_b": float(quad[1]),
        "quadratic_c": float(quad[2]),
        "quadratic_r2": quad_r2,
        "quadratic_aic": quad_aic,
        "quadratic_bic": quad_bic,
        "best_model": best,
        "deg_slope": deg_slope,
    }


def fit_all_stints(session: pd.DataFrame) -> pd.DataFrame:
    """Apply fit_degradation_curve to every qualifying stint in a session.

    Parameters
    ----------
    session : pd.DataFrame
        Full session lap DataFrame from build_lap_features. All laps are
        included; this function applies the clean-lap filter internally.

    Returns
    -------
    pd.DataFrame
        One row per qualifying stint (>= MIN_STINT_LAPS clean laps) with all
        columns from fit_degradation_curve plus ``car_number``, ``car_class``,
        and ``stint_id``. Empty DataFrame if no qualifying stints exist.
    """
    present_flags = [f for f in CLEAN_LAP_FLAGS if f in session.columns]
    if present_flags:
        clean = session[~session[present_flags].any(axis=1)].copy()
    else:
        clean = session.copy()

    rows = []
    for (car_number, stint_id), group in clean.groupby(["car_number", "stint_id"]):
        result = fit_degradation_curve(group)
        if not result:
            continue
        car_class = group["car_class"].iloc[0] if "car_class" in group.columns else None
        rows.append({"car_number": car_number, "car_class": car_class, "stint_id": stint_id, **result})

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def enrich_with_deg_slope(laps: pd.DataFrame) -> pd.DataFrame:
    """Add a per-stint ``deg_slope`` column to a lap-level DataFrame.

    Computes the degradation slope for each (car_number, stint_id) via
    fit_all_stints and merges it back onto every lap in that stint.
    Stints with fewer than MIN_STINT_LAPS clean laps receive ``deg_slope = 0.0``.

    Parameters
    ----------
    laps : pd.DataFrame
        Lap-level DataFrame from build_lap_features. Must contain
        ``car_number`` and ``stint_id`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``laps`` with a ``deg_slope`` column appended.
        All existing columns are preserved.
    """
    stints = fit_all_stints(laps)
    out = laps.copy()

    if stints.empty:
        out["deg_slope"] = 0.0
        return out

    slope_map = stints[["car_number", "stint_id", "deg_slope"]]
    out = out.merge(slope_map, on=["car_number", "stint_id"], how="left")
    out["deg_slope"] = out["deg_slope"].fillna(0.0)
    return out


def compare_degradation(session: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-stint degradation fits into a per-car summary.

    Parameters
    ----------
    session : pd.DataFrame
        Full session lap DataFrame from build_lap_features.

    Returns
    -------
    pd.DataFrame
        One row per car with columns:
        ``car_number``, ``car_class``, ``n_stints``,
        ``deg_slope_mean``, ``deg_slope_median``, ``deg_slope_std``,
        ``best_r2_mean``, ``deg_rank``.
        Sorted ascending by ``deg_rank`` (rank 1 = least degradation).
        Empty DataFrame if no qualifying stints exist.
    """
    stints = fit_all_stints(session)
    if stints.empty:
        return pd.DataFrame()

    stints["best_r2"] = stints.apply(
        lambda r: r["linear_r2"] if r["best_model"] == "linear" else r["quadratic_r2"],
        axis=1,
    )

    summary = (
        stints.groupby(["car_number", "car_class"])
        .agg(
            n_stints=("deg_slope", "count"),
            deg_slope_mean=("deg_slope", "mean"),
            deg_slope_median=("deg_slope", "median"),
            deg_slope_std=("deg_slope", "std"),
            best_r2_mean=("best_r2", "mean"),
        )
        .reset_index()
    )

    summary["deg_rank"] = summary["deg_slope_mean"].rank(method="min", ascending=True).astype(int)
    return summary.sort_values("deg_rank").reset_index(drop=True)
