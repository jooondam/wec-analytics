import pandas as pd


def clean_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw Al Kamel Systems session DataFrame for downstream analysis.

    Strips and lowercases all column names, drops rows with no lap time, and
    converts the ``lap_time`` column from ``"M:SS.mmm"`` string format to
    fractional seconds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by ``fetch_session``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``lap_time`` expressed in seconds as a float.
    """
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["lap_time"])
    df["lap_time"] = df["lap_time"].apply(parse_lap_time)
    return df

def parse_lap_time(laptime: str) -> float:
    """
    Convert a lap time string in ``"M:SS.mmm"`` format to fractional seconds.

    Parameters
    ----------
    laptime : str
        Lap time string as provided by Al Kamel Systems, e.g. ``"3:42.571"``.

    Returns
    -------
    float
        Total lap time in seconds, e.g. ``222.571``.
    """
    laptime_seconds = laptime.split(":")

    numeric_laptime = [float(t) for t in laptime_seconds]

    total_numeric_laptime = numeric_laptime[0] * 60 + numeric_laptime[1]
    return total_numeric_laptime


