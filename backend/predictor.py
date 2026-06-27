from __future__ import annotations

import json

from backend import storage
from backend.config import DEFAULT_PARAMS, ModelParams
from backend.football_api import FootballAPI
from backend.poisson_model import MatchPrediction, compute_league_model, predict_match


def params_from_version(version: dict | None) -> ModelParams:
    if not version:
        return DEFAULT_PARAMS
    data = json.loads(version["params_json"])
    return ModelParams(
        home_advantage=data.get("home_advantage", DEFAULT_PARAMS.home_advantage),
        shrinkage=data.get("shrinkage", DEFAULT_PARAMS.shrinkage),
        max_goals=data.get("max_goals", DEFAULT_PARAMS.max_goals),
    )


def ensure_active_version(conn) -> dict:
    version = storage.get_active_model_version(conn)
    if version:
        return version
    params_json = json.dumps({
        "home_advantage": DEFAULT_PARAMS.home_advantage,
        "shrinkage": DEFAULT_PARAMS.shrinkage,
        "max_goals": DEFAULT_PARAMS.max_goals,
    })
    storage.save_model_version(conn, params_json, None, None, "default", True)
    return storage.get_active_model_version(conn)


def prediction_view(match: dict, p: MatchPrediction, version_id: int) -> dict:
    return {
        "match_id": match["id"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        "utc_date": match["utc_date"],
        "stage": match["stage"],
        "pred_home": p.most_likely_score[0],
        "pred_away": p.most_likely_score[1],
        "prob_home": p.prob_home,
        "prob_draw": p.prob_draw,
        "prob_away": p.prob_away,
        "prob_over25": p.prob_over25,
        "prob_btts": p.prob_btts,
        "reliability": p.reliability,
        "model_version_id": version_id,
    }


def predict_upcoming(api: FootballAPI, conn) -> list[dict]:
    version = ensure_active_version(conn)
    params = params_from_version(version)
    model = compute_league_model(api.get_finished_matches(), params.shrinkage)

    views = []
    for match in api.get_upcoming_matches():
        p = predict_match(match["home_team"], match["away_team"], model, params)
        view = prediction_view(match, p, version["id"])
        storage.save_prediction(conn, {
            "match_id": match["id"],
            "competition": api.settings.competition_code,
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "match_utc_date": match["utc_date"],
            "model_version_id": version["id"],
            "pred_home": p.most_likely_score[0],
            "pred_away": p.most_likely_score[1],
            "prob_home": p.prob_home, "prob_draw": p.prob_draw, "prob_away": p.prob_away,
            "prob_over25": p.prob_over25, "prob_btts": p.prob_btts,
            "lambda_home": p.lambda_home, "lambda_away": p.lambda_away,
            "reliability": p.reliability,
        })
        views.append(view)
    return views
