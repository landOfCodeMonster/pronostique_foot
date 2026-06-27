import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.config import load_settings
from backend.football_api import FootballAPI
from backend.main import build_app

RAW = {"matches": [
    {"id": 1, "utcDate": "2026-06-20T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": 2, "away": 0}}},
    {"id": 2, "utcDate": "2026-06-21T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "B"}, "awayTeam": {"name": "C"}, "score": {"fullTime": {"home": 0, "away": 1}}},
    {"id": 4, "utcDate": "2026-06-22T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "C"}, "awayTeam": {"name": "A"}, "score": {"fullTime": {"home": 1, "away": 1}}},
    {"id": 5, "utcDate": "2026-06-23T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "A"}, "awayTeam": {"name": "C"}, "score": {"fullTime": {"home": 3, "away": 1}}},
    {"id": 6, "utcDate": "2026-06-24T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "B"}, "awayTeam": {"name": "C"}, "score": {"fullTime": {"home": 2, "away": 2}}},
    {"id": 3, "utcDate": "2026-06-30T18:00:00Z", "status": "TIMED", "stage": "GROUP_STAGE",
     "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": None, "away": None}}},
]}


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "t.db"
    storage.init_db(db)
    conn = storage.connect(db)
    api = FootballAPI(load_settings({"FOOTBALL_DATA_API_KEY": "x"}), fetcher=lambda p, q: RAW)
    return TestClient(build_app(api, conn)), conn


def test_upcoming_endpoint_returns_predictions(client):
    c, _ = client
    r = c.get("/api/matches/upcoming")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["match_id"] == 3
    assert "prob_home" in data[0]


def test_reconcile_then_performance(client):
    c, _ = client
    c.get("/api/matches/upcoming")          # creates a prediction for match 3 (still upcoming)
    assert c.post("/api/reconcile").json()["reconciled"] == 0  # match 3 not finished
    perf = c.get("/api/performance").json()
    assert perf["count"] == 0


def test_tune_endpoint_creates_version(client):
    c, _ = client
    r = c.post("/api/tune")
    assert r.status_code == 200
    assert r.json()["model_version_id"] >= 1
