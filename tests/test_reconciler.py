from backend import storage
from backend.config import load_settings
from backend.football_api import FootballAPI
from backend.reconciler import reconcile

RAW = {
    "matches": [
        {"id": 1, "utcDate": "2026-06-20T18:00:00Z", "status": "FINISHED", "stage": "GROUP_STAGE",
         "homeTeam": {"name": "A"}, "awayTeam": {"name": "B"}, "score": {"fullTime": {"home": 2, "away": 2}}},
    ]
}


def _setup(tmp_path):
    db = tmp_path / "t.db"
    storage.init_db(db)
    conn = storage.connect(db)
    settings = load_settings({"FOOTBALL_DATA_API_KEY": "x"})
    return conn, FootballAPI(settings, fetcher=lambda p, q: RAW)


def _pred(match_id):
    return {
        "match_id": match_id, "competition": "WC", "home_team": "A", "away_team": "B",
        "match_utc_date": "2026-06-20T18:00:00Z", "model_version_id": 1,
        "pred_home": 1, "pred_away": 1, "prob_home": 0.4, "prob_draw": 0.3, "prob_away": 0.3,
        "prob_over25": 0.5, "prob_btts": 0.5, "lambda_home": 1.2, "lambda_away": 1.2,
        "reliability": "faible",
    }


def test_reconcile_attaches_real_score(tmp_path):
    conn, api = _setup(tmp_path)
    storage.save_prediction(conn, _pred(1))
    assert reconcile(api, conn) == 1
    joined = storage.predictions_with_results(conn)
    assert joined[0]["actual_home"] == 2 and joined[0]["actual_away"] == 2
    # idempotent: second run reconciles nothing new
    assert reconcile(api, conn) == 0


def test_reconcile_ignores_unpredicted_matches(tmp_path):
    conn, api = _setup(tmp_path)  # no predictions stored
    assert reconcile(api, conn) == 0


def test_reconcile_updates_corrected_score(tmp_path):
    # A previously stored result (1-2) is later corrected by the source to 2-2.
    conn, api = _setup(tmp_path)  # RAW match 1 is FINISHED 2-2
    storage.save_prediction(conn, _pred(1))
    storage.save_result(conn, 1, 1, 2, "FINISHED")  # stale/incorrect score
    assert reconcile(api, conn) == 1                 # detects and updates
    joined = storage.predictions_with_results(conn)
    assert joined[0]["actual_home"] == 2 and joined[0]["actual_away"] == 2
    assert reconcile(api, conn) == 0                 # now idempotent
