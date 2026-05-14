import pandas as pd
import numpy as np

# filtering different classes with case invariance
def get_class_lap_time(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    return df[df["class"].str.casefold() == target_class.casefold()]

def detect_traffic_lap(df: pd.DataFrame, car_number: str, slow_threshold: float = 0.03) -> pd.DataFrame:
    # i chose 3% which is the minimum threshold to where a lap is considered impeded
    # filter the DataFrame for the specific car
    car_df = df[df["car_number"] == car_number]
    # get average lap for specific car
    avg_lap = car_df[~car_df["is_outlier"]]["lap_time"].mean()
    # calculate the cutoff using the formula
    cutoff = avg_lap * (1 + slow_threshold)
    # return only laps that exceed cutoff
    return car_df[car_df['lap_time'] > cutoff]