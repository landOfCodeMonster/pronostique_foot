from __future__ import annotations

import json

from backend import storage
from backend.config import ModelParams
from backend.football_api import FootballAPI
from backend.metrics import brier, outcome_from_score, rps
from backend.poisson_model import compute_league_model, predict_match

PARAM_GRID: list[ModelParams] = [
    ModelParams(home_advantage=ha, shrinkage=sh)
    for ha in (1.0, 1.1, 1.2)
    for sh in (0.5, 1.0, 2.0)
]


def backtest(finished_matches: list[dict], params: ModelParams, min_history: int = 3) -> tuple[float, float]:
    matches = sorted(finished_matches, key=lambda m: m["utc_date"])
    rps_total = 0.0
    brier_total = 0.0
    scored = 0
    for k in range(min_history, len(matches)):
        history = matches[:k]
        target = matches[k]
        model = compute_league_model(history, params.shrinkage)
        p = predict_match(target["home_team"], target["away_team"], model, params)
        probs = (p.prob_home, p.prob_draw, p.prob_away)
        outcome = outcome_from_score(target["home_goals"], target["away_goals"])
        rps_total += rps(probs, outcome)
        brier_total += brier(probs, outcome)
        scored += 1
    if scored == 0:
        return float("inf"), float("inf")
    return rps_total / scored, brier_total / scored


def tune(api: FootballAPI, conn, grid: list[ModelParams] | None = None) -> dict | None:
    grid = grid or PARAM_GRID
    finished = api.get_finished_matches()

    best = None  # (rps, brier, params)
    for params in grid:
        r, b = backtest(finished, params)
        if r == float("inf"):
            continue
        if best is None or r < best[0]:
            best = (r, b, params)

    if best is None:
        return None

    r, b, params = best
    params_json = json.dumps({
        "home_advantage": params.home_advantage,
        "shrinkage": params.shrinkage,
        "max_goals": params.max_goals,
    })
    storage.save_model_version(conn, params_json, r, b, "tuned", True)
    return storage.get_active_model_version(conn)
