"""
Multi-feature lap anomaly detection for wec_analytics ML layer.

Generalises Phase 1's univariate IQR detector (detect_outliers in
analysis/stints.py) to multi-feature anomaly detection. A lap may be
normal in lap time alone but anomalous in the combination of lap time,
stint position, traffic exposure, and class-relative pace.

Two complementary methods are provided:

  Isolation Forest -- tree-based; anomalies isolate quickly. Fast,
    robust to high dimensionality, good default choice.

  Local Outlier Factor -- density-based; flags points that are
    significantly less dense than their neighbours. Catches anomalies
    embedded inside the bulk distribution that IF can miss.

Both methods produce continuous scores (higher = more anomalous), which
lets you tune sensitivity rather than committing to a fixed label.
Scores are computed per car class to avoid treating cross-class pace
differences as anomalies.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from wec_analytics.ml.features import LAP_CLEAN_FLAGS

ANOMALY_FEATURES = [
    "class_pace_delta",  # deviation from class-best lap this lap number
    "stint_age",         # position within stint (lap 1 vs mid-stint vs late)
    "is_traffic_lap",    # 1 if impeded by a slower car
    "is_in_lap",         # 1 if pit-entry lap (slow by design)
    "is_out_lap",        # 1 if pit-exit lap (tyre warm-up artefact)
]

DEFAULT_CONTAMINATION = 0.05
VALID_METHODS = ("isolation_forest", "lof")
MIN_LAPS_TO_SCORE = 10  # minimum laps in a class to fit a meaningful model


def _prepare_features(group: pd.DataFrame) -> np.ndarray:
    """Extract and clean the feature matrix for one car-class group."""
    X = group[ANOMALY_FEATURES].copy()
    for col in ("is_traffic_lap", "is_in_lap", "is_out_lap"):
        if col in X.columns:
            X[col] = X[col].astype(float)
    # class_pace_delta is NaN when every lap in a lap_number is an outlier
    X["class_pace_delta"] = X["class_pace_delta"].fillna(0.0)
    return X.to_numpy(dtype=float)


def detect_lap_anomalies(
    session: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = DEFAULT_CONTAMINATION,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score each lap for multi-feature anomalousness and apply a binary flag.

    Fits Isolation Forest or Local Outlier Factor within each car class
    separately, so cross-class pace differences are never mistaken for
    anomalies. Returns the session DataFrame with two new columns.

    Parameters
    ----------
    session : pd.DataFrame
        Lap-level DataFrame from build_lap_features. Must contain all
        ANOMALY_FEATURES columns plus ``car_class``.
    method : 'isolation_forest' or 'lof'
        Anomaly detection algorithm.
    contamination : float
        Expected fraction of anomalous laps in the data, in (0, 0.5).
        Used to calibrate model decision boundaries. When ``threshold``
        is None, also determines what fraction of laps are flagged.
    threshold : float or None
        If given, flag laps where ``anomaly_score >= threshold``.
        If None, flag the top ``contamination`` fraction by score.

    Returns
    -------
    pd.DataFrame
        Copy of ``session`` with two columns appended:

        - ``anomaly_score`` (float): higher = more anomalous. Comparable
          within a session but not across sessions or methods.
        - ``is_lap_anomaly`` (bool): True when the lap is flagged.

        Cars whose class has fewer than MIN_LAPS_TO_SCORE laps receive
        ``anomaly_score = 0.0`` and ``is_lap_anomaly = False``.

    Raises
    ------
    ValueError
        If ``method`` is not one of VALID_METHODS.
    ValueError
        If ``contamination`` is not in (0, 0.5).
    KeyError
        If required feature columns are missing from ``session``.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if not 0 < contamination < 0.5:
        raise ValueError(f"contamination must be in (0, 0.5), got {contamination}")

    required = ANOMALY_FEATURES + ["car_class"]
    missing = [c for c in required if c not in session.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            "Ensure input is from build_lap_features."
        )

    out = session.copy()
    out["anomaly_score"] = 0.0
    out["is_lap_anomaly"] = False

    for car_class, group in session.groupby("car_class"):
        idx = group.index
        if len(group) < MIN_LAPS_TO_SCORE:
            continue

        X_raw = _prepare_features(group)
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )
            clf.fit(X)
            # score_samples: lower is more anomalous; negate so higher = worse
            scores = -clf.score_samples(X)

        else:  # lof
            n_neighbors = min(20, len(group) - 1)
            clf = LocalOutlierFactor(
                contamination=contamination,
                novelty=False,
                n_neighbors=n_neighbors,
            )
            clf.fit_predict(X)
            # negative_outlier_factor_: lower is more anomalous; negate
            scores = -clf.negative_outlier_factor_

        out.loc[idx, "anomaly_score"] = scores.astype(float)

    # apply flag
    if threshold is not None:
        out["is_lap_anomaly"] = out["anomaly_score"] >= threshold
    else:
        cutoff = out["anomaly_score"].quantile(1.0 - contamination)
        out["is_lap_anomaly"] = out["anomaly_score"] >= cutoff

    return out


def compare_with_iqr(
    session: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = DEFAULT_CONTAMINATION,
) -> pd.DataFrame:
    """Compare ML anomaly flags against the Phase 1 IQR outlier flag.

    The disagreement between methods is informative: multi-feature ML
    catches anomalies that are normal in lap time alone, while IQR may
    catch clear safety-car laps that blend into the density model.

    Parameters
    ----------
    session : pd.DataFrame
        Lap-level DataFrame from build_lap_features. Must contain
        ``is_outlier`` (from detect_outliers) in addition to the
        columns required by detect_lap_anomalies.
    method : 'isolation_forest' or 'lof'
        Anomaly detection algorithm passed to detect_lap_anomalies.
    contamination : float
        Contamination parameter passed to detect_lap_anomalies.

    Returns
    -------
    pd.DataFrame
        One row per agreement type with columns:

        - ``agreement_type``: one of ``"both_flagged"``, ``"iqr_only"``,
          ``"ml_only"``, ``"neither_flagged"``
        - ``n_laps`` (int): number of laps in this bucket
        - ``pct_laps`` (float): percentage of total laps

    Raises
    ------
    KeyError
        If ``is_outlier`` is missing (run detect_outliers first).
    """
    if "is_outlier" not in session.columns:
        raise KeyError(
            "Missing 'is_outlier' column. Run detect_outliers() before compare_with_iqr()."
        )

    annotated = detect_lap_anomalies(session, method=method, contamination=contamination)

    iqr = annotated["is_outlier"]
    ml = annotated["is_lap_anomaly"]
    total = len(annotated)

    buckets = {
        "both_flagged": int((iqr & ml).sum()),
        "iqr_only": int((iqr & ~ml).sum()),
        "ml_only": int((~iqr & ml).sum()),
        "neither_flagged": int((~iqr & ~ml).sum()),
    }

    return pd.DataFrame([
        {
            "agreement_type": k,
            "n_laps": v,
            "pct_laps": round(v / total * 100, 2),
        }
        for k, v in buckets.items()
    ])
