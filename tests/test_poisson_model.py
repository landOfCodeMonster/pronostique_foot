import numpy as np
import pytest
from backend.config import ModelParams
from backend.poisson_model import (
    compute_league_model,
    score_matrix,
    predict_match,
    reliability_label,
)

FINISHED = [
    {"home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 0},
    {"home_team": "B", "away_team": "C", "home_goals": 0, "away_goals": 2},
    {"home_team": "C", "away_team": "A", "home_goals": 1, "away_goals": 1},
    {"home_team": "A", "away_team": "C", "home_goals": 2, "away_goals": 1},
]


def test_score_matrix_sums_to_one():
    m = score_matrix(1.4, 1.1, 8)
    assert m.shape == (9, 9)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)


def test_strong_attack_has_strength_above_one():
    model = compute_league_model(FINISHED, shrinkage=0.0)
    # A scored 3+1+2 = 6 in 3 games = 2.0/game, well above league average
    assert model.strengths["A"].attack > 1.0
    assert model.league_avg_goals > 0


def test_predict_match_outputs_valid_probabilities():
    model = compute_league_model(FINISHED, shrinkage=1.0)
    p = predict_match("A", "B", model, ModelParams())
    assert p.prob_home + p.prob_draw + p.prob_away == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= p.prob_over25 <= 1.0
    assert 0.0 <= p.prob_btts <= 1.0
    assert len(p.most_likely_score) == 2


def test_stronger_team_more_likely_to_win():
    model = compute_league_model(FINISHED, shrinkage=0.0)
    p = predict_match("A", "B", model, ModelParams())
    assert p.prob_home > p.prob_away


def test_unknown_team_falls_back_to_average():
    model = compute_league_model(FINISHED, shrinkage=1.0)
    p = predict_match("A", "UNKNOWN", model, ModelParams())
    assert p.prob_home + p.prob_draw + p.prob_away == pytest.approx(1.0, abs=1e-6)


def test_reliability_label_thresholds():
    assert reliability_label(0, 1) == "faible"
    assert reliability_label(2, 2) == "moyen"
    assert reliability_label(4, 5) == "élevé"
