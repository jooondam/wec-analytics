import numpy as np

def calculate_driver_stint(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['stint_change_detected'] = df['car_driver'] != df['car_driver'].shift(periods=1) 
    df_new['stint_number'] = df_new['stint_change_detected'].cumsum()
    return df_new
    
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    clean_laps = df_new[(df_new['crossing_finish_line_in_pit'] != 'B') & (df_new['pit_time'].isna() | (df_new['pit_time'] == 0))].copy()
    Q1 = clean_laps['lap_time'].quantile(0.25)
    Q3 = clean_laps['lap_time'].quantile(0.75)
    # get interquarile range
    IQR = Q3 - Q1
    # define both fences
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    # outlier
    df_new['is_outlier'] = (df_new['lap_time'] <= lower_fence) | (df_new['lap_time'] >= upper_fence)

    return df_new

def calculate_driver_stint_stats(df: pd.DataFrame) -> pd.DataFrame:
    # filter out outlier laps (safety car, pit laps) before aggregating
    inlier_laps = df[~df['is_outlier']].copy()

    # create lap position within each stint (resets to 1 at each new stint)
    # this makes degradation slopes comparable across different stints
    inlier_laps['lap_in_stint'] = inlier_laps.groupby('stint_number').cumcount() + 1

    # aggregate per-stint statistics using only clean laps
    stint_stats = inlier_laps.groupby('stint_number')['lap_time'].agg(
        avg_lap='mean',
        fastest_lap='min'
    )

    # calculate degradation slope for each stint using linear regression
    # uses a separate helper to handle the two-column nature of polyfit
    stint_slopes = inlier_laps.groupby('stint_number').apply(_calculate_degradation_slope)
    stint_stats['degradation_slope'] = stint_slopes

    # reset index so stint_number becomes a regular column in the output
    stint_stats = stint_stats.reset_index()
    return stint_stats


def _calculate_degradation_slope(stint_df: pd.DataFrame) -> float | None:
    # guard clause, need at least 3 laps for a meaningful trend line
    if len(stint_df) < 3:
        return None
    slope, intercept = np.polyfit(stint_df['lap_in_stint'], stint_df['lap_time'], 1)
    return float(slope)