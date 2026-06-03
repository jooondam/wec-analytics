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
}