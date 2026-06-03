import re
from pathlib import Path

import pandas as pd
import requests
from urllib.parse import unquote

def extract_session_id(url: str) -> str:
    """Extract the session_id directory segment from an Al Kamel Systems URL.

    Parameters
    ----------
    url : str
        Al Kamel Systems CSV URL. The session_id is the path segment matching
        YYYYMMDDHHMM_SessionType, e.g. '201905041330_Race'.

    Returns
    -------
    str
        Session ID string, e.g. '201905041330_Race'.

    Raises
    ------
    ValueError
        If no matching segment is found in the URL.
    """
    match = re.search(r'/(\d{12}_\w+)/', unquote(url))
    if not match:
        raise ValueError(
            f"Could not extract session_id from URL: {url}\n"
            "Expected a path segment matching YYYYMMDDHHMM_SessionType."
        )
    return match.group(1)


def read_csv_with_fallback(cache_path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            cache_path,
            sep=";",
            encoding="utf-8-sig",
            skipinitialspace=True,
            usecols=lambda c: not c.startswith("Unnamed"),
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            cache_path,
            sep=";",
            encoding="latin-1",
            skipinitialspace=True,
            usecols=lambda c: not c.startswith("Unnamed"),
        )


def fetch_session(url: str, cache_dir: str | Path = "cache") -> pd.DataFrame:
    """
    Download a session CSV from Al Kamel Systems, caching it locally on first fetch.

    On subsequent calls with the same URL the cached file is returned immediately
    without making a network request. Files are cached under a session_id
    subdirectory (e.g. cache/201905041330_Race/23_Analysis_Race_Hour 6.CSV)
    so that sessions with identical filenames don't collide.

    Parameters
    ----------
    url : str
        Direct URL to the Al Kamel Systems session CSV file.
    cache_dir : str or Path, optional
        Directory where downloaded files are stored. Created automatically if it
        does not exist. Defaults to ``"cache"``.

    Returns
    -------
    pd.DataFrame
        Raw session data with one row per lap as provided by Al Kamel Systems.

    Raises
    ------
    ConnectionError
        If the HTTP request fails (non-200 status) or a network error occurs.
    """
    session_id = extract_session_id(url)
    unquoted_filename = unquote(url.split("/")[-1])
    cache_path = Path(cache_dir) / session_id / unquoted_filename

    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return read_csv_with_fallback(cache_path)
    else:
        print(f"Fetching from URL: {url}")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(response.content)
                print(f"Saved to cache: {cache_path}")
                return read_csv_with_fallback(cache_path)
            else:
                raise ConnectionError(f"Failed to connect: {url} - status code {response.status_code}")
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to download: {url} - {e}")
