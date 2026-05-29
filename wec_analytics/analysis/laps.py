import pandas as pd
import numpy as np

# filtering different classes with case invariance
def get_class_laps(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    """
    Filter a session DataFrame to laps belonging to a single car class.

    The comparison is case-insensitive, so ``"hypercar"`` and ``"Hypercar"``
    both match.

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame containing a ``class`` column.
    target_class : str
        Car class to filter for, e.g. ``"Hypercar"`` or ``"LMGT3"``.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` where ``class`` matches ``target_class``.
    """
    return df[df["class"].str.casefold() == target_class.casefold()]

def detect_traffic_lap(df: pd.DataFrame, slow_threshold: float = 0.03) -> pd.DataFrame:
    """
    Flag laps where a car was likely impeded by traffic.

    A lap is flagged if its time exceeds the car's average clean lap time
    by more than ``slow_threshold``. The average is computed from non‑outlier
    laps only. This version operates on the **full session DataFrame** and adds
    an ``is_traffic_lap`` column to every row — it never drops data.

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame with ``car_number``, ``lap_time``, and ``is_outlier``
        columns. Must have had ``detect_outliers`` applied first.
    slow_threshold : float, optional
        Fractional excess above the car's mean clean lap time used as the
        impeded cutoff. Defaults to ``0.03`` (3%).

    Returns
    -------
    pd.DataFrame
        A **copy** of the input DataFrame with the new boolean column
        ``is_traffic_lap`` appended. No rows are removed.

    Raises
    ------
    KeyError
        If the ``is_outlier`` column is absent — run ``detect_outliers`` first.
    """
    # i chose 3% which is the minimum threshold to where a lap is considered impeded
    # this guard clause ensures outlier flags exist so we can compute clean averages
    if "is_outlier" not in df.columns:
        raise KeyError(
            "Missing 'is_outlier' column. Run detect_outliers() before detect_traffic_lap()."
        )

    # work on a copy so the original DataFrame is never modified
    laps = df.copy()

    # compute per‑car mean of clean laps (non‑outlier only)
    clean_laps = laps[~laps["is_outlier"]]
    car_mean = clean_laps.groupby("car_number")["lap_time"].mean()

    # map the per‑car mean onto every row (including outliers)
    laps["_mean_clean"] = laps["car_number"].map(car_mean)

    # flag laps slower than the cutoff: (1 + threshold) * car_mean
    laps["is_traffic_lap"] = laps["lap_time"] > laps["_mean_clean"] * (1 + slow_threshold)

    # drop the temporary helper column to keep the output clean
    laps.drop(columns=["_mean_clean"], inplace=True)

    return laps

def compare_class_pace(df: pd.DataFrame, car_class: str, car_class_comp: str) -> pd.DataFrame:
    """
    Compare median lap time and pace consistency between two car classes.

    Outlier laps are excluded before aggregation. Consistency is expressed as
    the interquartile range of lap times — a lower IQR means more consistent
    pace.

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame with ``class``, ``lap_time``, and ``is_outlier``
        columns. Must have had ``detect_outliers`` applied first.
    car_class : str
        First car class to compare, e.g. ``"Hypercar"``.
    car_class_comp : str
        Second car class to compare, e.g. ``"LMGT3"``.

    Returns
    -------
    pd.DataFrame
        Two-row DataFrame indexed by class name with columns
        ``lap_time_median`` and ``lap_time_iqr``, both in seconds.

    Raises
    ------
    KeyError
        If the ``is_outlier`` column is absent — run ``detect_outliers`` first.
    """
    # guard clause to Verify the required column exists
    if "is_outlier" not in df.columns:
        raise KeyError(
            "The input DataFrame is missing the required 'is_outlier' column. "
            "Please run your outlier detection step before calling this function."
        )
    
    results = {}

    for class_name in [car_class, car_class_comp]:
        # filter to specific class using helper function
        class_df = get_class_laps(df, class_name)

        # filter out rows where outlier is true
        clean_df = class_df[~class_df["is_outlier"]]

        # calculate median and iqr of lap time
        median_val = clean_df["lap_time"].median()
        q75 = clean_df["lap_time"].quantile(0.75)
        q25 = clean_df["lap_time"].quantile(0.25)
        iqr_val = q75 - q25

        results[class_name] = {"lap_time_median": median_val, "lap_time_iqr": iqr_val}

    return pd.DataFrame(results).T  