import pandas as pd
import numpy as np

def calculate_pit_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rows corresponding to actual pit stops from a session DataFrame.

    Works with both raw session data (uses ``crossing_finish_line_in_pit``)
    and the processed feature DataFrame (falls back to ``is_in_lap``).

    Parameters
    ----------
    df : pd.DataFrame
        Session DataFrame containing pit timing columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with at minimum ``lap_number``, ``car_number``,
        and ``pit_time``. Additional columns included when present.
    """
    if 'is_in_lap' in df.columns:
        # Processed data: is_in_lap is the authoritative flag (already cleaned)
        pit_mask = df['is_in_lap'].astype(bool)
    else:
        # Raw data: require crossing flag AND a recorded pit_time
        pit_mask = (df['crossing_finish_line_in_pit'] == 'B') & \
                   (df['pit_time'].notna()) & (df['pit_time'] != 0)

    mask = pit_mask

    base_cols = ['lap_number', 'car_number', 'pit_time']
    optional = ['team', 'class', 'car_class', 'stint_number', 'stint_id']
    cols = base_cols + [c for c in optional if c in df.columns]
    return df.loc[mask, cols]
    


def _get_pit_lap_delta(df: pd.DataFrame, car_number: int, rival_car: int) -> dict:
    if car_number not in df['car_number'].values or \
            rival_car not in df['car_number'].values:
        return None

    car_laps = sorted(df[df['car_number'] == car_number]['lap_number'].tolist())
    rival_laps = sorted(df[df['car_number'] == rival_car]['lap_number'].tolist())
    rival_set = set(rival_laps)
    car_set = set(car_laps)

    # Skip simultaneous stops (both cars pit on the same lap, e.g. safety car)
    # and use the first stop where each car pitted on a unique lap.
    car_laps_unique = [l for l in car_laps if l not in rival_set]
    rival_laps_unique = [l for l in rival_laps if l not in car_set]

    if not car_laps_unique or not rival_laps_unique:
        return None

    car_lap = car_laps_unique[0]
    rival_lap = rival_laps_unique[0]
    # negative = car pitted first, positive = car pitted later
    lap_delta = car_lap - rival_lap

    return {
        "car_number": car_number,
        "rival_car": rival_car,
        "lap_delta": lap_delta,
        "car_pit_lap": car_lap,
        "rival_pit_lap": rival_lap
    }



def _compare_pace_in_window(df: pd.DataFrame, delta_data: dict) -> dict:
    car_number = delta_data['car_number']
    rival_car = delta_data['rival_car']
    car_pit_lap = delta_data['car_pit_lap']
    rival_pit_lap = delta_data['rival_pit_lap']

    # defining the range, both the out lap and subsequent laps on fresh tires
    start_lap = car_pit_lap + 1
    end_lap = rival_pit_lap

    # guard clause
    if start_lap > end_lap:
        return None

    # filter for specific lap range for both cars
    window_df = df[df['lap_number'].isin(range(start_lap, end_lap + 1))]

    # calculate average pace for both of cars in this window
    car_pace = window_df[window_df['car_number'] == car_number]['lap_time'].mean()
    rival_pace = window_df[window_df['car_number'] == rival_car]['lap_time'].mean()

    return {
        "car_number": car_number,
        "rival_car": rival_car,
        "car_window_pace": car_pace,
        "rival_window_pace": rival_pace,
        "pace_delta": car_pace - rival_pace
    }


def _compare_position_change(df: pd.DataFrame, delta_data: dict) -> dict:
    car_number = delta_data['car_number']
    rival_car = delta_data['rival_car']
    car_pit_lap = delta_data['car_pit_lap']
    rival_pit_lap = delta_data['rival_pit_lap']

    # filter pre pit laps for the primary car
    car_pre_pit = df[(df['car_number'] == car_number) & (df['lap_number'] < car_pit_lap)]
    # filter pre pit laps for the rival car
    rival_pre_pit = df[(df['car_number'] == rival_car) & (df['lap_number'] < rival_pit_lap)]

    # calculate cumulative baseline lap times
    car_pre_base_time = car_pre_pit['lap_time'].sum()
    rival_pre_base_time = rival_pre_pit['lap_time'].sum()

    # filter post pit laps for the primary car
    car_post_pit = df[(df['car_number'] == car_number) & (df['lap_number'] <= car_pit_lap + 1)]
    # filter post pit laps for the rival car
    rival_post_pit = df[(df['car_number'] == rival_car) & (df['lap_number'] <= rival_pit_lap + 1)]

    # calculate cumulative baseline lap times
    car_post_base_time = car_post_pit['lap_time'].sum()
    rival_post_base_time = rival_post_pit['lap_time'].sum()

    # check if primary car was ahead before pit stop
    was_ahead_before = car_pre_base_time < rival_pre_base_time
    #check if primary car was ahead after pit stop
    is_ahead_after = car_post_base_time < rival_post_base_time
    # position changed if the leadership flipped between the two snapshot
    position_changed = was_ahead_before != is_ahead_after

    return {
        "car_pre_base_time": car_pre_base_time,
        "rival_pre_base_time": rival_pre_base_time,
        "car_post_base_time": car_post_base_time,
        "rival_post_base_time": rival_post_base_time,
        "position_changed": position_changed
    }

def detect_undercut(df: pd.DataFrame, car_number: int, rival_car: int) -> dict:
    """
    Determine whether a car attempted and succeeded in undercutting a rival.

    An undercut attempt is defined as the primary car pitting before the rival.
    Success requires both a pace advantage during the window between the two pit
    stops and a position change after the rival exits the pits.

    Parameters
    ----------
    df : pd.DataFrame
        Full session DataFrame (not the pit stops subset) containing lap times,
        car numbers, and pit stop data.
    car_number : int
        Car number of the car attempting the undercut.
    rival_car : int
        Car number of the rival being undercut.

    Returns
    -------
    dict
        Always contains ``car_number``, ``rival_car``, and
        ``undercut_attempted``. When ``undercut_attempted`` is ``False`` a
        ``reason`` string is included. When ``True`` and the window is valid,
        also contains ``undercut_successful`` (bool), ``pace_details``, and
        ``position_details``.
    """
    pit_stops_df = calculate_pit_stops(df)
    
    # delegate the guard clause and lap extraction to your existing helper
    delta_data = _get_pit_lap_delta(pit_stops_df, car_number, rival_car)

    # handle the scenario where a car did not pit (helper returned None)
    if delta_data is None:
        return {
            "car_number": car_number,
            "rival_car": rival_car,
            "undercut_attempted": False,
            "reason": "One or both cars did not make a valid pit stop."
        }

    # check if the primary car actually pitted first (is it an undercut attempt?)
    if delta_data["lap_delta"] >= 0:
        return {
            "car_number": car_number,
            "rival_car": rival_car,
            "undercut_attempted": False,
            "reason": f"Car {car_number} did not pit before rival {rival_car}."
        }

    # execute the downstream helper functions using delta_data
    pace_results = _compare_pace_in_window(df, delta_data)
    position_results = _compare_position_change(df, delta_data)

    # guard if the pacing window is invalid (e.g. pitted on sequential laps)
    if pace_results is None:
        return {
            "car_number": car_number,
            "rival_car": rival_car,
            "undercut_attempted": True,
            "reason": "Insufficient lap window between pit stops to evaluate pace."
        }

    return {
        "car_number": car_number,
        "rival_car": rival_car,
        "undercut_attempted": True,
        "car_pit_lap": delta_data["car_pit_lap"],
        "rival_pit_lap": delta_data["rival_pit_lap"],
        "undercut_successful": (pace_results["pace_delta"] < 0) and position_results["position_changed"],
        "pace_details": pace_results,
        "position_details": position_results
    }