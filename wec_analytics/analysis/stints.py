def calculate_driver_stint(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['stint_change_detected'] = df['car_driver'] != df['car_driver'].shift(periods=1) 
    df_new['stint_number'] = df_new['stint_change_detected'].cumsum()
    return df_new
    
