def clean_session(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["lap_time"])
    df["lap_time"] = df["lap_time"].apply(parse_lap_time)
    return df

def parse_lap_time(laptime: str) -> float:
    laptime_seconds = laptime.split(":")

    numeric_laptime = [float(t) for t in laptime_seconds]
    
    total_numeric_laptime = numeric_laptime[0] * 60 + numeric_laptime[1]
    return total_numeric_laptime


