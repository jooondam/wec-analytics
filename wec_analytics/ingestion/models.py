import pandas as pd


def clean_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw Al Kamel Systems session DataFrame for downstream analysis.

    Strips and lowercases all column names, drops rows with no lap time,
    converts the ``lap_time`` column from ``"M:SS.mmm"`` string format to
    fractional seconds, and adds ``is_in_lap`` and ``is_out_lap`` boolean
    columns derived from pit crossing and pit time data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by ``fetch_session``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``lap_time`` in seconds and pit lap flags.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["lap_time"])
    df["lap_time"] = df["lap_time"].apply(parse_lap_time)

    # flag laps where the car crossed the finish line entering the pit
    df["is_in_lap"] = df["crossing_finish_line_in_pit"] == "B"

    # flag the lap immediately after an in-lap for each car
    df["is_out_lap"] = (
        df.groupby("number")["is_in_lap"]
        .shift(1)
        .fillna(False)
    )

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


