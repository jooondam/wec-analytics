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