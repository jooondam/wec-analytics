import pandas as pd
import numpy as np

def calculate_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df['crossing_finish_line_in_pit'] == 'B') & 
            # checking that pit time in not empty and null
            ((df['pit_time'].notna()) & (df['pit_time'] != 0)),
            ['lap_number', 'car_number', 'team', 'class', 'pit_time', 'stint_number']]
    
def detect_undercut(df, car_number, rival_car):
    current_pit_stop = calculate_pit_stops(df)
    return current_pit_stop[current_pit_stop['car_number'].isin([car_number, rival_car])]


def _get_pit_lap_delta(df: pd.DataFrame, car_number: int, rival_car: int ) -> dict:
    # guard clause to make sure both cars have pitstop
    if car_number not in df['car_number'].values or \
    rival_car not in df['car_number'].values:
        return None
    
    car_lap = df[df['car_number'] == car_number]['lap_number'].iloc[0]
    rival_lap = df[df['car_number'] == rival_car]['lap_number'].iloc[0]

    # negative value means that your car pitted first, positve value means that your car pitted later
    lap_delta = car_lap - rival_lap

    return {
        "car_number": car_number,
        "rival_car": rival_car,
        "lap_delta": lap_delta,
        "car_pit_lap": car_lap,
        "rival_pit_lap": rival_lap
    }



def _compare_pace_in_window():

def _compare_position_change():