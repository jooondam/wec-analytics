# wec-analytics

WEC timing data from Al Kamel Systems is rich but raw, lap times arrive as strings, pit stops are buried in flag columns, and there's no concept of a stint or an outlier. This library takes that raw CSV and gives you a clean analysis pipeline: fetch, clean, detect stints, flag outliers, then run whatever analysis you need.

Built for the FIA World Endurance Championship, targeting Hypercar and LMGT3 sessions.

## Requirements

Python 3.10 or later.

```bash
git clone https://github.com/jooondam/wec-analytics.git
cd wec-analytics
pip install -r requirements.txt
```

## Quick start

```python
from wec_analytics import (
    fetch_session,
    clean_session,
    calculate_driver_stint,
    detect_outliers,
    detect_traffic_lap,
    compare_class_pace,
    detect_undercut,
)

# URL from 2019 Spa 6 Hours; check fiawec.alkamelsystems.com for current archive structure
raw = fetch_session("http://fiawec.alkamelsystems.com/Results/08_2018-2019/07_SPA%20FRANCORCHAMPS/267_FIA%20WEC/201905041330_Race/Hour%206/23_Analysis_Race_Hour%206.CSV")
df = clean_session(raw)

# assign stint numbers and flag outlier laps — both required before analysis
df = calculate_driver_stint(df)
df = detect_outliers(df)

# find laps where car 8 was likely in traffic
traffic = detect_traffic_lap(df, car_number="8")

# compare median pace and consistency between two classes
pace = compare_class_pace(df, "Hypercar", "LMGT3")

# check if car 8 undercut car 50
result = detect_undercut(df, car_number=8, rival_car=50)
```

`fetch_session` caches the CSV locally on first download so subsequent calls are instant.

## Concepts

**Stints** are continuous runs by the same driver without a pit stop. `calculate_driver_stint` detects driver changes by comparing consecutive `car_driver` values and assigns a cumulative stint number to each lap.

**Outlier detection** uses the Tukey IQR fence method. The fences are fitted on clean racing laps only (no pit-lane crossings, no recorded pit time), then applied to the full dataset so that safety car laps, pit laps, and formation laps all get flagged. `detect_outliers` must be run before any pace analysis — `detect_traffic_lap` and `compare_class_pace` will raise a `KeyError` if the `is_outlier` column is absent.

**Traffic laps** are identified by comparing a car's lap time against its own average clean pace. A lap more than 3% slower than that average is treated as impeded — a threshold derived from WEC baseline data.

**Undercut detection** checks whether a car pitted before a rival, ran faster during the window between the two pit stops, and came out ahead. All three conditions are evaluated separately so you can inspect the pace and position evidence independently.

## Public API

```python
# ingestion
fetch_session(url, cache_dir="cache")   # download and cache a session CSV
clean_session(df)                        # normalise columns, parse lap times

# stints and outliers
calculate_driver_stint(df)               # add stint_number column
detect_outliers(df)                      # add is_outlier column
calculate_driver_stint_stats(df)         # per-stint avg, fastest lap, degradation slope

# lap analysis
get_class_laps(df, target_class)         # filter to one car class
detect_traffic_lap(df, car_number)       # laps likely impeded by traffic
compare_class_pace(df, class_a, class_b) # median pace and IQR by class

# strategy
calculate_pit_stops(df)                  # extract pit stop rows
detect_undercut(df, car_number, rival_car)
```

## Data

Timing data is sourced from [Al Kamel Systems](https://www.alkamelsystems.com), the official timing provider for the FIA World Endurance Championship. This library does not redistribute any timing data. Session CSV URLs must be supplied by the user and are intended for personal and research use only — redistribution of Al Kamel Systems data is not permitted.
