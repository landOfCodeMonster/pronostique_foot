from backend.config import load_settings
from backend.football_api import FootballAPI

RAW = {
    "matches": [
        {
            "id": 101, "utcDate": "2026-06-30T18:00:00Z", "status": "FINISHED",
            "stage": "GROUP_STAGE",
            "homeTeam": {"name": "France"}, "awayTeam": {"name": "Brazil"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "id": 102, "utcDate": "2026-07-02T18:00:00Z", "status": "TIMED",
            "stage": "GROUP_STAGE",
            "homeTeam": {"name": "Spain"}, "awayTeam": {"name": "Japan"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
        {
            "id": 103, "utcDate": "2026-07-05T18:00:00Z", "status": "IN_PLAY",
            "stage": "GROUP_STAGE",
            "homeTeam": {"name": "Italy"}, "awayTeam": {"name": "Ghana"},
            "score": {"fullTime": {"home": 1, "away": 0}},
        },
    ]
}


def _api():
    settings = load_settings({"FOOTBALL_DATA_API_KEY": "x"})
    return FootballAPI(settings, fetcher=lambda path, params: RAW)


def test_get_matches_normalizes():
    matches = _api().get_matches()
    m = next(x for x in matches if x["id"] == 101)
    assert m["home_team"] == "France"
    assert m["home_goals"] == 2 and m["away_goals"] == 1
    assert m["status"] == "FINISHED"


def test_get_finished_matches_only_finished():
    api = _api()
    assert [m["id"] for m in api.get_finished_matches()] == [101]


def test_get_upcoming_includes_live_first():
    # Open matches include live (IN_PLAY/PAUSED) and scheduled; live sorts first
    # even when its kickoff is later than a scheduled match.
    api = _api()
    assert [m["id"] for m in api.get_upcoming_matches()] == [103, 102]


def test_is_live_flag():
    from backend.football_api import is_live
    assert is_live({"status": "IN_PLAY"}) is True
    assert is_live({"status": "PAUSED"}) is True
    assert is_live({"status": "TIMED"}) is False
