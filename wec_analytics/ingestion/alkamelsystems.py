from pathlib import Path
import pandas as pd
import requests
from urllib.parse import unquote

def read_csv_with_fallback(cache_path: str | Path ) -> pd.DataFrame:
    try:
        return pd.read_csv(cache_path, encoding= 'utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(cache_path, encoding= 'latin-1')

def fetch_session(url: str, cache_dir: str | Path = "cache") -> pd.DataFrame:

    filename = url.split("/")[-1]
    unquoted_filename = unquote(filename)
    cache_path = Path(cache_dir) / unquoted_filename

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
                raise ConnectionError(f"Failed to download: {url} - status code {e}")
        
        