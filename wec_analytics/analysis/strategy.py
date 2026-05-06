import pandas as pd
import numpy as np

def calculate_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df['crossing_finish_line_in_pit'] == 'B') & 
            # checking that pit time in not empty and null
            ((df['pit_time'].notna()) & (df['pit_time'] != 0)),
            ['lap_number', 'car_number', 'team', 'class', 'pit_time', 'stint_number']]
    
