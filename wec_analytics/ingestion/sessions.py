# ingestion/sessions.py
"""Session metadata mapping for the WEC analytics pipeline."""

from collections import namedtuple

# lightweight container for race-level metadata — namedtuple keeps it
# immutable and gives free __repr__ without the overhead of a dataclass
SessionMeta = namedtuple("SessionMeta", ["race_id", "track", "year", "duration_hours"])

# maps the session_id extracted from an Al Kamel URL to its race context.
# keys follow the format YYYYMMDDHHMM_SessionType, e.g. "201905041330_Race" —
# this is the directory segment that extract_session_id pulls from the URL path.
# add one entry here for every session you want to use in training or evaluation;
# attach_race_id will raise a KeyError with an actionable message if a session
# is fetched but has no entry in this dict.
SESSION_MAP = {
    "201905041330_Race": SessionMeta(
        race_id="SPA_2019_RACE",
        track="Spa-Francorchamps",
        year=2019,
        duration_hours=6,
    ),
    "201810141100_Race": SessionMeta(
        race_id="FUJI_2018_RACE",
        track="Fuji Speedway",
        year=2018,
        duration_hours=6,
    ),
    "201811181100_Race": SessionMeta(
        race_id="SHANGHAI_2018_RACE",
        track="Shanghai International Circuit",
        year=2018,
        duration_hours=6,
    ),
    "201808191200_Race": SessionMeta(
        race_id="SILVERSTONE_2018_RACE",
        track="Silverstone",
        year=2018,
        duration_hours=6,
    ),
    "201903151600_Race": SessionMeta(
        race_id="SEBRING_2019_RACE",
        track="Sebring",
        year=2019,
        duration_hours=8,
    ),
    "201906151500_Race": SessionMeta(
        race_id="LE_MANS_2019_RACE",
        track="Circuit de la Sarthe",
        year=2019,
        duration_hours=24,
    ),
    # 2019-2020 season
    "201909011200_Race": SessionMeta(
        race_id="SILVERSTONE_2019_RACE",
        track="Silverstone",
        year=2019,
        duration_hours=4,
    ),
    "201910061100_Race": SessionMeta(
        race_id="FUJI_2019_RACE",
        track="Fuji Speedway",
        year=2019,
        duration_hours=6,
    ),
    "201911101200_Race": SessionMeta(
        race_id="SHANGHAI_2019_RACE",
        track="Shanghai International Circuit",
        year=2019,
        duration_hours=4,
    ),
    "201912141500_Race": SessionMeta(
        race_id="BAHRAIN_2019_RACE",
        track="Bahrain International Circuit",
        year=2019,
        duration_hours=8,
    ),
    "202002231200_Race": SessionMeta(
        race_id="COTA_2020_RACE",
        track="Circuit of the Americas",
        year=2020,
        duration_hours=6,
    ),
    "202008151330_Race": SessionMeta(
        race_id="SPA_2020_RACE",
        track="Spa-Francorchamps",
        year=2020,
        duration_hours=6,
    ),
    "202009191430_Race": SessionMeta(
        race_id="LE_MANS_2020_RACE",
        track="Circuit de la Sarthe",
        year=2020,
        duration_hours=24,
    ),
    "202011141400_Race": SessionMeta(
        race_id="BAHRAIN_2020_RACE",
        track="Bahrain International Circuit",
        year=2020,
        duration_hours=8,
    ),
}