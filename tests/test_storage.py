import json
from backend import storage


def _conn(tmp_path):
    db = tmp_path / "t.db"
    storage.init_db(db)
    return storage.connect(db)


def _row(match_id=1, version_id=1):
    return {
        "match_id": match_id, "competition": "WC", "home_team": "A", "away_team": "B",
        "match_utc_date": "2026-06-30T18:00:00Z", "model_version_id": version_id,
        "pred_home": 2, "pred_away": 1, "prob_home": 0.5, "prob_draw": 0.3,
        "prob_away": 0.2, "prob_over25": 0.55, "prob_btts": 0.6,
        "lambda_home": 1.6, "lambda_away": 1.1, "reliability": "moyen",
    }


def test_model_version_active_is_unique(tmp_path):
    conn = _conn(tmp_path)
    v1 = storage.save_model_version(conn, json.dumps({"home_advantage": 1.0}), 0.2, 0.3, "v1", True)
    v2 = storage.save_model_version(conn, json.dumps({"home_advantage": 1.1}), 0.18, 0.29, "v2", True)
    active = storage.get_active_model_version(conn)
    assert active["id"] == v2
    assert len(storage.all_model_versions(conn)) == 2
    assert v1 != v2


def test_save_and_get_prediction(tmp_path):
    conn = _conn(tmp_path)
    storage.save_prediction(conn, _row())
    got = storage.get_prediction(conn, 1)
    assert got["home_team"] == "A"
    assert got["prob_home"] == 0.5


def test_prediction_upsert_replaces(tmp_path):
    conn = _conn(tmp_path)
    storage.save_prediction(conn, _row())
    row2 = _row(); row2["pred_home"] = 3
    storage.save_prediction(conn, row2)
    assert storage.get_prediction(conn, 1)["pred_home"] == 3


def test_results_and_join(tmp_path):
    conn = _conn(tmp_path)
    storage.save_prediction(conn, _row(match_id=1))
    storage.save_prediction(conn, _row(match_id=2))
    assert sorted(storage.predicted_match_ids_without_result(conn)) == [1, 2]
    storage.save_result(conn, 1, 2, 0, "FINISHED")
    assert storage.predicted_match_ids_without_result(conn) == [2]
    joined = storage.predictions_with_results(conn)
    assert len(joined) == 1
    assert joined[0]["actual_home"] == 2 and joined[0]["actual_away"] == 0
