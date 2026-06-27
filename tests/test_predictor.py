from backend import storage
from backend.config import load_settings
from backend.football_api import FootballAPI
from backend.predictor import predict_upcoming

RAW = {
    "matches": [
        {"id": 1, "utcDate": "2026-06-20T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
         "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": 3, "away": 0}}},
        {"id": 2, "utcDate": "2026-06-21T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
         "homeTeam": {"name": "B"}, "awayTeam": {"name": "C"}, "score": {"fullTime": {"home": 0, "away": 2}}},
        {"id": 3, "utcDate": "2026-06-30T18:00:00Z", "status": "TIMED", "stage": "GROUP_STAGE",
         "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": None, "away": None}}},
    ]
}


def _setup(tmp_path):
    db = tmp_path / "t.db"
    storage.init_db(db)
    conn = storage.connect(db)
    settings = load_settings({"FOOTBALL_DATA_API_KEY": "x"})
    api = FootballAPI(settings, fetcher=lambda p, q: RAW)
    return conn, api


def test_predict_upcoming_returns_and_persists(tmp_path):
    conn, api = _setup(tmp_path)
    views = predict_upcoming(api, conn)
    assert len(views) == 1
    v = views[0]
    assert v["match_id"] == 3
    assert v["prob_home"] + v["prob_draw"] + v["prob_away"] == __import__("pytest").approx(1.0, abs=1e-6)
    # persisted
    stored = storage.get_prediction(conn, 3)
    assert stored is not None
    assert stored["model_version_id"] >= 1


def test_predict_upcoming_creates_active_version(tmp_path):
    conn, api = _setup(tmp_path)
    predict_upcoming(api, conn)
    assert storage.get_active_model_version(conn) is not None
