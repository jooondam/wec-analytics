import pandas as pd
import numpy as np

# filtering different classes with case invariance
def get_class_laps(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    return df[df["class"].str.casefold() == target_class.casefold()]

def detect_traffic_lap(df: pd.DataFrame, car_number: str, slow_threshold: float = 0.03) -> pd.DataFrame:
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