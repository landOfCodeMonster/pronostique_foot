import math
from backend import storage
from backend.config import ModelParams, load_settings
from backend.football_api import FootballAPI
from backend.tuner import backtest, tune


def _finished(seq):
    out = []
    for i, (h, a, hg, ag) in enumerate(seq):
        out.append({
            "id": i + 1, "utc_date": f"2026-06-{10 + i:02d}T18:00:00Z", "status": "FINISHED",
            "stage": "GROUP_STAGE", "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag,
        })
    return out


SEQ = _finished([
    ("A", "B", 2, 0), ("C", "D", 1, 1), ("A", "C", 1, 0), ("B", "D", 0, 2),
    ("A", "D", 3, 1), ("C", "B", 2, 1), ("D", "A", 0, 1), ("B", "C", 1, 1),
])


def test_backtest_returns_finite_scores():
    rps, brier = backtest(SEQ, ModelParams(), min_history=3)
    assert math.isfinite(rps) and 0.0 <= rps <= 1.0
    assert math.isfinite(brier)


def test_backtest_insufficient_history_is_inf():
    rps, brier = backtest(SEQ[:2], ModelParams(), min_history=3)
    assert rps == float("inf")


def test_tune_creates_new_active_version(tmp_path):
    db = tmp_path / "t.db"
    storage.init_db(db)
    conn = storage.connect(db)
    raw = {"matches": [
        {"id": m["id"], "utcDate": m["utc_date"], "status": "FINISHED", "stage": m["stage"],
         "homeTeam": {"name": m["home_team"]}, "awayTeam": {"name": m["away_team"]},
         "score": {"fullTime": {"home": m["home_goals"], "away": m["away_goals"]}}}
        for m in SEQ
    ]}
    api = FootballAPI(load_settings({"FOOTBALL_DATA_API_KEY": "x"}), fetcher=lambda p, q: raw)
    version = tune(api, conn)
    assert version is not None
    assert version["backtest_rps"] is not None
    active = storage.get_active_model_version(conn)
    assert active["id"] == version["id"]
