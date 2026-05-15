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

def detect_traffic_lap(df: pd.DataFrame, car_number: str, slow_threshold: float = 0.03) -> pd.DataFrame:
    """
    Identify laps where a car was likely impeded by traffic.

    A lap is flagged as a traffic lap if its time exceeds the car's average
    clean lap time by more than ``slow_threshold``. The average is computed
    from non-outlier laps only.

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame with ``car_number``, ``lap_time``, and ``is_outlier``
        columns. Must have had ``detect_outliers`` applied first.
    car_number : str
        The car number to analyse, e.g. ``"8"``.
    slow_threshold : float, optional
        Fractional excess above the car's mean clean lap time used as the
        impeded cutoff. Defaults to ``0.03`` (3%).

    Returns
    -------
    pd.DataFrame
        Subset of the car's laps where ``lap_time`` exceeds the cutoff.

    Raises
    ------
    KeyError
        If the ``is_outlier`` column is absent — run ``detect_outliers`` first.
    """
    # i chose 3% which is the minimum threshold to where a lap is considered impeded
    # filter the DataFrame for the specific car
    if "is_outlier" not in df.columns:
        raise KeyError(
            "The input DataFrame is missing the required 'is_outlier' column. "
            "Please run detect_outliers() before calling this function."
        )
    car_df = df[df["car_number"] == car_number]
    # get average lap for specific car
    avg_lap = car_df[~car_df["is_outlier"]]["lap_time"].mean()
    # calculate the cutoff using the formula
    cutoff = avg_lap * (1 + slow_threshold)
    # return only laps that exceed cutoff
    return car_df[car_df['lap_time'] > cutoff]

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