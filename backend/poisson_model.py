from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.config import ModelParams


@dataclass
class TeamStrength:
    attack: float
    defense: float
    matches_played: int


@dataclass
class LeagueModel:
    strengths: dict[str, TeamStrength]
    league_avg_goals: float


@dataclass
class MatchPrediction:
    lambda_home: float
    lambda_away: float
    most_likely_score: tuple[int, int]
    prob_home: float
    prob_draw: float
    prob_away: float
    prob_over25: float
    prob_btts: float
    reliability: str


def _shrink(raw: float, matches: int, shrinkage: float) -> float:
    """Pull a raw strength toward 1.0 when few matches are available."""
    weight = matches / (matches + shrinkage) if (matches + shrinkage) > 0 else 0.0
    return 1.0 + weight * (raw - 1.0)


def compute_league_model(matches: list[dict], shrinkage: float) -> LeagueModel:
    scored: dict[str, int] = {}
    conceded: dict[str, int] = {}
    played: dict[str, int] = {}
    total_goals = 0
    team_games = 0

    for m in matches:
        h, a = m["home_team"], m["away_team"]
        hg, ag = m["home_goals"], m["away_goals"]
        for t in (h, a):
            scored.setdefault(t, 0)
            conceded.setdefault(t, 0)
            played.setdefault(t, 0)
        scored[h] += hg
        scored[a] += ag
        conceded[h] += ag
        conceded[a] += hg
        played[h] += 1
        played[a] += 1
        total_goals += hg + ag
        team_games += 2

    league_avg = (total_goals / team_games) if team_games else 1.3

    strengths: dict[str, TeamStrength] = {}
    for t, n in played.items():
        raw_attack = (scored[t] / n) / league_avg if league_avg else 1.0
        raw_defense = (conceded[t] / n) / league_avg if league_avg else 1.0
        strengths[t] = TeamStrength(
            attack=_shrink(raw_attack, n, shrinkage),
            defense=_shrink(raw_defense, n, shrinkage),
            matches_played=n,
        )
    return LeagueModel(strengths=strengths, league_avg_goals=league_avg)


def _poisson_vector(lam: float, max_goals: int) -> np.ndarray:
    ks = np.arange(0, max_goals + 1)
    lgamma = np.array([math.lgamma(k + 1) for k in ks])
    log_pmf = -lam + ks * math.log(lam) - lgamma
    return np.exp(log_pmf)


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray:
    home = _poisson_vector(lambda_home, max_goals)
    away = _poisson_vector(lambda_away, max_goals)
    matrix = np.outer(home, away)
    total = matrix.sum()
    return matrix / total if total else matrix


def reliability_label(matches_home: int, matches_away: int) -> str:
    fewest = min(matches_home, matches_away)
    if fewest <= 1:
        return "faible"
    if fewest <= 3:
        return "moyen"
    return "élevé"


def predict_match(
    home_team: str, away_team: str, model: LeagueModel, params: ModelParams
) -> MatchPrediction:
    avg = model.league_avg_goals
    neutral = TeamStrength(attack=1.0, defense=1.0, matches_played=0)
    home = model.strengths.get(home_team, neutral)
    away = model.strengths.get(away_team, neutral)

    lambda_home = avg * home.attack * away.defense * params.home_advantage
    lambda_away = avg * away.attack * home.defense
    lambda_home = max(lambda_home, 0.05)
    lambda_away = max(lambda_away, 0.05)

    matrix = score_matrix(lambda_home, lambda_away, params.max_goals)

    prob_home = float(np.tril(matrix, -1).sum())
    prob_draw = float(np.trace(matrix))
    prob_away = float(np.triu(matrix, 1).sum())

    n = matrix.shape[0]
    idx = np.add.outer(np.arange(n), np.arange(n))
    prob_over25 = float(matrix[idx >= 3].sum())
    prob_btts = float(matrix[1:, 1:].sum())

    i, j = np.unravel_index(int(np.argmax(matrix)), matrix.shape)

    return MatchPrediction(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        most_likely_score=(int(i), int(j)),
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        prob_over25=prob_over25,
        prob_btts=prob_btts,
        reliability=reliability_label(home.matches_played, away.matches_played),
    )
