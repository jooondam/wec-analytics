"""
Association rule mining for WEC strategy events.

Discretises each race into fixed-width time windows and builds one
boolean transaction per (car, window) listing which events occurred.
Apriori is then run across all pooled sessions to find frequent itemsets
and high-confidence rules such as {sc_period, q1} -> {next_pit}.
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

DEFAULT_WINDOW_MINUTES = 15
DEFAULT_MIN_SUPPORT = 0.05
DEFAULT_MIN_CONFIDENCE = 0.5
LOOK_AHEAD_WINDOWS = 2
SC_FIELD_FRACTION = 0.5
TRAFFIC_LAP_FRACTION = 0.30

_ITEM_COLUMNS = [
    "pit_stop", "driver_change", "traffic_heavy", "outlier_lap",
    "sc_period", "q1", "q2", "q3", "q4", "next_pit",
]


def _parse_elapsed_minutes(elapsed: pd.Series) -> pd.Series:
    """Parse Al Kamel 'M:SS.mmm' cumulative elapsed strings to float minutes."""
    def _convert(val):
        try:
            parts = str(val).split(":")
            if len(parts) != 2:
                return float("nan")
            return int(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, TypeError):
            return float("nan")

    return elapsed.map(_convert)


def _assign_windows(elapsed_minutes: pd.Series, window_minutes: int) -> pd.Series:
    return (elapsed_minutes // window_minutes).astype("Int64")


def _detect_sc_windows(session: pd.DataFrame, window_col: str) -> set:
    """Return window indices where > SC_FIELD_FRACTION of cars have outlier (non-pit) laps."""
    n_total_cars = session["car_number"].nunique()
    if n_total_cars == 0:
        return set()

    # Only non-pit outlier laps count as SC evidence
    sc_laps = session[session["is_outlier"] & ~session["is_in_lap"]].copy()

    sc_windows = set()
    for window_id, group in sc_laps.groupby(window_col):
        n_outlier_cars = group["car_number"].nunique()
        if n_outlier_cars / n_total_cars >= SC_FIELD_FRACTION:
            sc_windows.add(window_id)

    return sc_windows


def _build_transactions(session: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
    """Build one boolean row per (car, window) from a cleaned session DataFrame."""
    required = ["elapsed", "car_number", "driver_name", "car_class",
                "is_in_lap", "is_out_lap", "is_outlier", "is_traffic_lap"]
    missing = [c for c in required if c not in session.columns]
    if missing:
        raise KeyError(
            f"Session is missing columns required for transaction building: {missing}. "
            "Ensure the data has been processed by the full Phase 1 pipeline."
        )

    df = session.copy()
    df = df.sort_values(["car_number", "lap_number"] if "lap_number" in df.columns else ["car_number"])

    elapsed_minutes = _parse_elapsed_minutes(df["elapsed"])
    if elapsed_minutes.isna().all():
        raise KeyError(
            "Could not parse any 'elapsed' values. Expected 'M:SS.mmm' format "
            "(e.g. '142:06.349'). Check the elapsed column contents."
        )

    df["_elapsed_min"] = elapsed_minutes
    df["_window"] = _assign_windows(elapsed_minutes, window_minutes)

    # Drop rows with unparseable elapsed
    df = df.dropna(subset=["_elapsed_min"])

    sc_windows = _detect_sc_windows(df, "_window")

    max_window = int(df["_window"].max()) if not df["_window"].isna().all() else 0
    total_windows = max_window + 1

    def _quarter(w):
        frac = w / max(total_windows - 1, 1)
        if frac < 0.25:
            return "q1"
        elif frac < 0.50:
            return "q2"
        elif frac < 0.75:
            return "q3"
        return "q4"

    # Collect car classes for one-hot columns
    classes = sorted(df["car_class"].dropna().unique())

    rows = []
    for (car_number, window_id), group in df.groupby(["car_number", "_window"], observed=True):
        w = int(window_id)
        quarter = _quarter(w)

        car_class_val = group["car_class"].iloc[0] if not group["car_class"].isna().all() else None

        row = {
            "car_number": car_number,
            "window_id": w,
            "pit_stop": bool(group["is_in_lap"].any()),
            "driver_change": (
                group["driver_name"].iloc[0] != group["driver_name"].iloc[-1]
                if len(group) > 1 else False
            ),
            "traffic_heavy": (
                group["is_traffic_lap"].mean() >= TRAFFIC_LAP_FRACTION
            ),
            "outlier_lap": bool(
                (group["is_outlier"] & ~group["is_in_lap"]).any()
            ),
            "sc_period": w in sc_windows,
            "q1": quarter == "q1",
            "q2": quarter == "q2",
            "q3": quarter == "q3",
            "q4": quarter == "q4",
            "next_pit": False,  # filled in below
        }

        for cls in classes:
            row[f"class_{cls}"] = (car_class_val == cls)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    tx = pd.DataFrame(rows)

    # Fill next_pit look-ahead
    car_groups = tx.groupby("car_number")
    tx = tx.copy()
    tx["next_pit"] = False

    for car_num, car_tx in car_groups:
        car_idx = car_tx.index
        for i, idx in enumerate(car_idx):
            w = tx.loc[idx, "window_id"]
            future_windows = car_tx[
                (car_tx["window_id"] > w) & (car_tx["window_id"] <= w + LOOK_AHEAD_WINDOWS)
            ]
            if not future_windows.empty and future_windows["pit_stop"].any():
                tx.loc[idx, "next_pit"] = True

    return tx


def build_strategy_transactions(
    session: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Build a boolean transaction table from a cleaned session DataFrame.

    Parameters
    ----------
    session : pd.DataFrame
        Cleaned and feature-engineered lap DataFrame. Must contain:
        elapsed, car_number, driver_name, car_class, is_in_lap, is_out_lap,
        is_outlier, is_traffic_lap.
    window_minutes : int
        Width of each time window in minutes. Default 15.

    Returns
    -------
    pd.DataFrame
        One row per (car, window). Identifier columns: car_number, window_id.
        Boolean item columns: pit_stop, driver_change, traffic_heavy,
        outlier_lap, sc_period, q1, q2, q3, q4, next_pit, class_{X}.
    """
    return _build_transactions(session, window_minutes)


def mine_strategy_rules(
    sessions,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Mine association rules from strategy event transactions.

    Parameters
    ----------
    sessions : pd.DataFrame or list[pd.DataFrame]
        One or more cleaned session DataFrames. Transactions are pooled.
    min_support : float
        Minimum fraction of transactions an itemset must appear in (0, 1).
    min_confidence : float
        Minimum confidence threshold for association rules (0, 1).
    window_minutes : int
        Time window width used when building transactions.

    Returns
    -------
    pd.DataFrame
        Rules sorted by lift descending with columns:
        antecedents, consequents, antecedents_str, consequents_str,
        support, confidence, lift, leverage, conviction.
        Empty DataFrame (with correct columns) if no rules are found.

    Raises
    ------
    ValueError
        If min_support or min_confidence are not in (0, 1).
    """
    if not (0 < min_support < 1):
        raise ValueError(f"min_support must be in (0, 1), got {min_support}")
    if not (0 < min_confidence < 1):
        raise ValueError(f"min_confidence must be in (0, 1), got {min_confidence}")

    if isinstance(sessions, pd.DataFrame):
        sessions = [sessions]

    all_transactions = []
    for session in sessions:
        tx = build_strategy_transactions(session, window_minutes=window_minutes)
        if not tx.empty:
            all_transactions.append(tx)

    if not all_transactions:
        return _empty_rules_df()

    pooled = pd.concat(all_transactions, ignore_index=True)

    # Select only boolean item columns (drop identifiers)
    id_cols = {"car_number", "window_id"}
    item_cols = [c for c in pooled.columns if c not in id_cols]

    items = pooled[item_cols].astype(bool)

    frequent_itemsets = apriori(items, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        return _empty_rules_df()

    rules = association_rules(
        frequent_itemsets, metric="confidence", min_threshold=min_confidence
    )

    if rules.empty:
        return _empty_rules_df()

    rules["antecedents_str"] = rules["antecedents"].apply(
        lambda fs: "{" + ", ".join(sorted(fs)) + "}"
    )
    rules["consequents_str"] = rules["consequents"].apply(
        lambda fs: "{" + ", ".join(sorted(fs)) + "}"
    )

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules


def _empty_rules_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "antecedents", "consequents", "antecedents_str", "consequents_str",
        "support", "confidence", "lift", "leverage", "conviction",
    ])
