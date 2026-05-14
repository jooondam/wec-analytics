import pandas as pd
import numpy as np

def get_class_lap_time(df: pd.DataFrame, target_class: str) -> pd.DataFrame:
    return df[df["class"].str.casefold() == target_class.casefold()]