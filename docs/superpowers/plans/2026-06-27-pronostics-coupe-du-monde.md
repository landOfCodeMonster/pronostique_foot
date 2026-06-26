# Pronostics Coupe du Monde 2026 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web app that predicts World Cup 2026 match scores with a Poisson model, stores every prediction, reconciles it with the real result, and self-improves by re-tuning its hyperparameters on the stored history.

**Architecture:** A FastAPI backend exposes a JSON API and serves a static frontend. A pure-math Poisson core (`poisson_model`) turns team attack/defense strengths into a score matrix from which all markets (exact score, 1X2, Over/Under 2.5, BTTS) are derived. A `football_api` client (with disk cache) fetches real data from football-data.org. SQLite (`storage`) holds predictions, results, and model versions. `reconciler` attaches real scores to past predictions; `metrics` scores them (RPS/Brier/log-loss/calibration); `tuner` runs a walk-forward backtest to pick better hyperparameters and records a new model version.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, requests, NumPy, SQLite (stdlib `sqlite3`), pytest, httpx (FastAPI TestClient). Vanilla HTML/CSS/JS frontend.

## Global Constraints

- Python 3.11+ required (uses `tuple[int, int]` / `X | None` syntax natively).
- API key read only from env var `FOOTBALL_DATA_API_KEY`; never hard-coded, never sent to the frontend.
- Competition is parameterized via `COMPETITION_CODE` (default `WC`).
- football-data.org base URL: `https://api.football-data.org/v4`; auth header: `X-Auth-Token`; rate limit ~10 req/min → all network reads go through the disk cache.
- Pure modules (`poisson_model`, `metrics`) must have **no** network or DB imports and must be unit-testable in isolation.
- All tests run with `pytest` from the project root. Use `tmp_path` for any DB/cache files — never touch real `data/app.db` in tests.
- Probabilities for a match (`prob_home + prob_draw + prob_away`) must sum to ~1.0 (tolerance 1e-6) after matrix normalization.
- 1X2 outcome encoding is fixed everywhere: `0 = home win, 1 = draw, 2 = away win`.
- Reliability labels are exactly the strings `"faible"`, `"moyen"`, `"élevé"`.
- Every task ends with a commit. Use Conventional Commit prefixes (`feat:`, `test:`, `chore:`).

---

## File Structure

```
pronostique_foot/
├── backend/
│   ├── __init__.py
│   ├── config.py          # Settings + ModelParams, env loading
│   ├── poisson_model.py   # PURE: strengths, λ, score matrix, market derivation
│   ├── metrics.py         # PURE: rps, brier, log_loss, calibration, exact hit
│   ├── storage.py         # SQLite: predictions, results, model_versions
│   ├── cache.py           # disk cache get/set with TTL
│   ├── football_api.py    # football-data.org client (injectable fetcher)
│   ├── predictor.py       # orchestrate: strengths → predict → persist
│   ├── reconciler.py      # attach real scores to stored predictions
│   ├── tuner.py           # walk-forward backtest + hyperparameter search
│   └── main.py            # FastAPI app + routes + static serving
├── frontend/
│   ├── index.html
│   ├── performance.html
│   ├── style.css
│   └── app.js
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_poisson_model.py
│   ├── test_metrics.py
│   ├── test_storage.py
│   ├── test_cache.py
│   ├── test_football_api.py
│   ├── test_predictor.py
│   ├── test_reconciler.py
│   ├── test_tuner.py
│   └── test_api.py
├── data/                  # runtime SQLite (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

**Shared data shapes (used across tasks):**

A **normalized match dict**:
```python
{
    "id": int,
    "utc_date": str,        # ISO 8601, e.g. "2026-06-30T18:00:00Z"
    "status": str,          # "SCHEDULED" | "TIMED" | "IN_PLAY" | "FINISHED" | ...
    "stage": str,           # e.g. "GROUP_STAGE", "LAST_16"
    "home_team": str,
    "away_team": str,
    "home_goals": int | None,   # None until played
    "away_goals": int | None,
}
```

A **prediction row dict** (what `storage.save_prediction` accepts):
```python
{
    "match_id": int, "competition": str, "home_team": str, "away_team": str,
    "match_utc_date": str, "model_version_id": int,
    "pred_home": int, "pred_away": int,
    "prob_home": float, "prob_draw": float, "prob_away": float,
    "prob_over25": float, "prob_btts": float,
    "lambda_home": float, "lambda_away": float, "reliability": str,
}
```

---

### Task 1: Project scaffold, dependencies, and config

**Files:**
- Create: `requirements.txt`, `.env.example`, `backend/__init__.py`, `tests/__init__.py`, `backend/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `backend/config.py`:
    - `@dataclass(frozen=True) class ModelParams` with fields `home_advantage: float = 1.0`, `shrinkage: float = 1.0`, `max_goals: int = 8`.
    - `@dataclass(frozen=True) class Settings` with fields `api_key: str`, `competition_code: str`, `base_url: str`, `db_path: Path`, `cache_dir: Path`, `cache_ttl_seconds: int`.
    - `DEFAULT_PARAMS: ModelParams = ModelParams()`.
    - `load_settings(env: Mapping[str, str] | None = None) -> Settings` — reads from `env` (defaults to `os.environ`).

- [ ] **Step 1: Create dependency and env files**

`requirements.txt`:
```
fastapi==0.111.0
uvicorn[standard]==0.30.1
requests==2.32.3
numpy==1.26.4
pytest==8.2.2
httpx==0.27.0
```

`.env.example`:
```
FOOTBALL_DATA_API_KEY=your_token_here
COMPETITION_CODE=WC
```

`backend/__init__.py`: empty file.
`tests/__init__.py`: empty file.

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from backend.config import load_settings, DEFAULT_PARAMS, ModelParams


def test_load_settings_reads_api_key_and_defaults():
    env = {"FOOTBALL_DATA_API_KEY": "abc123"}
    s = load_settings(env)
    assert s.api_key == "abc123"
    assert s.competition_code == "WC"
    assert s.base_url == "https://api.football-data.org/v4"
    assert isinstance(s.db_path, Path)
    assert s.cache_ttl_seconds > 0


def test_load_settings_competition_override():
    env = {"FOOTBALL_DATA_API_KEY": "x", "COMPETITION_CODE": "FL1"}
    assert load_settings(env).competition_code == "FL1"


def test_default_params_are_neutral():
    assert DEFAULT_PARAMS == ModelParams(home_advantage=1.0, shrinkage=1.0, max_goals=8)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.config'`.

- [ ] **Step 4: Write minimal implementation**

`backend/config.py`:
```python
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelParams:
    home_advantage: float = 1.0
    shrinkage: float = 1.0
    max_goals: int = 8


DEFAULT_PARAMS = ModelParams()


@dataclass(frozen=True)
class Settings:
    api_key: str
    competition_code: str
    base_url: str
    db_path: Path
    cache_dir: Path
    cache_ttl_seconds: int


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = env if env is not None else os.environ
    return Settings(
        api_key=env.get("FOOTBALL_DATA_API_KEY", ""),
        competition_code=env.get("COMPETITION_CODE", "WC"),
        base_url="https://api.football-data.org/v4",
        db_path=PROJECT_ROOT / "data" / "app.db",
        cache_dir=PROJECT_ROOT / "data" / "cache",
        cache_ttl_seconds=3600,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example backend/__init__.py tests/__init__.py backend/config.py tests/test_config.py
git commit -m "feat: project scaffold and config module"
```

---

### Task 2: Poisson model core (pure math)

**Files:**
- Create: `backend/poisson_model.py`
- Test: `tests/test_poisson_model.py`

**Interfaces:**
- Consumes: `ModelParams` from `backend.config`.
- Produces:
  - `@dataclass class TeamStrength` fields `attack: float`, `defense: float`, `matches_played: int`.
  - `@dataclass class LeagueModel` fields `strengths: dict[str, TeamStrength]`, `league_avg_goals: float`.
  - `@dataclass class MatchPrediction` fields `lambda_home: float`, `lambda_away: float`, `most_likely_score: tuple[int, int]`, `prob_home: float`, `prob_draw: float`, `prob_away: float`, `prob_over25: float`, `prob_btts: float`, `reliability: str`.
  - `compute_league_model(matches: list[dict], shrinkage: float) -> LeagueModel` — `matches` are normalized finished matches (have integer `home_goals`/`away_goals`).
  - `score_matrix(lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray` — normalized so it sums to 1.0; `matrix[i][j] = P(home=i, away=j)`.
  - `predict_match(home_team: str, away_team: str, model: LeagueModel, params: ModelParams) -> MatchPrediction`.
  - `reliability_label(matches_home: int, matches_away: int) -> str` — returns `"faible"`/`"moyen"`/`"élevé"`.

- [ ] **Step 1: Write the failing test**

`tests/test_poisson_model.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_poisson_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.poisson_model'`.

- [ ] **Step 3: Write minimal implementation**

`backend/poisson_model.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_poisson_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/poisson_model.py tests/test_poisson_model.py
git commit -m "feat: Poisson model core with market derivation"
```

---

### Task 3: Scoring metrics (pure math)

**Files:**
- Create: `backend/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `outcome_from_score(home_goals: int, away_goals: int) -> int` → `0`/`1`/`2`.
  - `rps(probs: tuple[float, float, float], outcome: int) -> float` — Ranked Probability Score over ordered categories (home, draw, away).
  - `brier(probs: tuple[float, float, float], outcome: int) -> float`.
  - `log_loss(probs: tuple[float, float, float], outcome: int) -> float`.
  - `exact_hit(pred: tuple[int, int], actual: tuple[int, int]) -> bool`.
  - `calibration_bins(samples: list[tuple[float, int]], n_bins: int = 10) -> list[dict]` — each `sample` is `(predicted_prob_of_event, event_occurred 0/1)`; returns bins with keys `lower`, `upper`, `count`, `mean_pred`, `observed`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import math
import pytest
from backend.metrics import (
    outcome_from_score,
    rps,
    brier,
    log_loss,
    exact_hit,
    calibration_bins,
)


def test_outcome_from_score():
    assert outcome_from_score(2, 1) == 0
    assert outcome_from_score(1, 1) == 1
    assert outcome_from_score(0, 3) == 2


def test_rps_perfect_prediction_is_zero():
    assert rps((1.0, 0.0, 0.0), 0) == pytest.approx(0.0, abs=1e-12)


def test_rps_penalizes_distance_ordering():
    # predicting away when home happened (distance 2) is worse than predicting draw
    far = rps((0.0, 0.0, 1.0), 0)
    near = rps((0.0, 1.0, 0.0), 0)
    assert far > near


def test_brier_perfect_is_zero():
    assert brier((0.0, 1.0, 0.0), 1) == pytest.approx(0.0, abs=1e-12)


def test_brier_known_value():
    # probs (0.5,0.3,0.2), outcome home(0): (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25+0.09+0.04
    assert brier((0.5, 0.3, 0.2), 0) == pytest.approx(0.38, abs=1e-9)


def test_log_loss_perfect_is_near_zero():
    assert log_loss((1.0, 0.0, 0.0), 0) == pytest.approx(0.0, abs=1e-9)


def test_exact_hit():
    assert exact_hit((2, 1), (2, 1)) is True
    assert exact_hit((2, 1), (1, 1)) is False


def test_calibration_bins_basic():
    samples = [(0.05, 0), (0.15, 0), (0.95, 1), (0.85, 1)]
    bins = calibration_bins(samples, n_bins=10)
    populated = [b for b in bins if b["count"] > 0]
    assert sum(b["count"] for b in populated) == 4
    high = [b for b in populated if b["mean_pred"] > 0.5][0]
    assert high["observed"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.metrics'`.

- [ ] **Step 3: Write minimal implementation**

`backend/metrics.py`:
```python
import math


def outcome_from_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _one_hot(outcome: int) -> tuple[float, float, float]:
    return tuple(1.0 if i == outcome else 0.0 for i in range(3))  # type: ignore[return-value]


def rps(probs: tuple[float, float, float], outcome: int) -> float:
    e = _one_hot(outcome)
    cum_p = 0.0
    cum_e = 0.0
    total = 0.0
    for k in range(len(probs) - 1):
        cum_p += probs[k]
        cum_e += e[k]
        total += (cum_p - cum_e) ** 2
    return total / (len(probs) - 1)


def brier(probs: tuple[float, float, float], outcome: int) -> float:
    e = _one_hot(outcome)
    return sum((p - ei) ** 2 for p, ei in zip(probs, e))


def log_loss(probs: tuple[float, float, float], outcome: int) -> float:
    p = max(probs[outcome], 1e-15)
    return -math.log(p)


def exact_hit(pred: tuple[int, int], actual: tuple[int, int]) -> bool:
    return tuple(pred) == tuple(actual)


def calibration_bins(samples: list[tuple[float, int]], n_bins: int = 10) -> list[dict]:
    bins = []
    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins
        in_bin = [s for s in samples if (lower <= s[0] < upper) or (b == n_bins - 1 and s[0] == 1.0)]
        count = len(in_bin)
        mean_pred = sum(s[0] for s in in_bin) / count if count else 0.0
        observed = sum(s[1] for s in in_bin) / count if count else 0.0
        bins.append({
            "lower": lower, "upper": upper, "count": count,
            "mean_pred": mean_pred, "observed": observed,
        })
    return bins
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/metrics.py tests/test_metrics.py
git commit -m "feat: scoring metrics (RPS, Brier, log-loss, calibration)"
```

---

### Task 4: SQLite storage layer

**Files:**
- Create: `backend/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: nothing (takes a `sqlite3.Connection`).
- Produces:
  - `init_db(db_path: Path) -> None` — creates parent dir + tables if absent.
  - `connect(db_path: Path) -> sqlite3.Connection` — returns connection with `row_factory = sqlite3.Row`.
  - `save_model_version(conn, params_json: str, backtest_rps: float | None, backtest_brier: float | None, notes: str, is_active: bool) -> int` — if `is_active`, deactivates others; returns new id.
  - `get_active_model_version(conn) -> dict | None`.
  - `all_model_versions(conn) -> list[dict]`.
  - `save_prediction(conn, row: dict) -> int` — upserts on `match_id` (replace if a prediction for that match already exists); returns row id.
  - `get_prediction(conn, match_id: int) -> dict | None`.
  - `save_result(conn, match_id: int, actual_home: int, actual_away: int, status: str) -> None`.
  - `predicted_match_ids_without_result(conn) -> list[int]`.
  - `predictions_with_results(conn) -> list[dict]` — join exposing prediction fields plus `actual_home`, `actual_away`.

- [ ] **Step 1: Write the failing test**

`tests/test_storage.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.storage'`.

- [ ] **Step 3: Write minimal implementation**

`backend/storage.py`:
```python
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    params_json TEXT NOT NULL,
    backtest_rps REAL,
    backtest_brier REAL,
    notes TEXT,
    is_active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER UNIQUE NOT NULL,
    competition TEXT, home_team TEXT, away_team TEXT, match_utc_date TEXT,
    model_version_id INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    pred_home INTEGER, pred_away INTEGER,
    prob_home REAL, prob_draw REAL, prob_away REAL,
    prob_over25 REAL, prob_btts REAL,
    lambda_home REAL, lambda_away REAL, reliability TEXT
);
CREATE TABLE IF NOT EXISTS results (
    match_id INTEGER PRIMARY KEY,
    actual_home INTEGER, actual_away INTEGER,
    status TEXT, reconciled_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_PRED_FIELDS = [
    "match_id", "competition", "home_team", "away_team", "match_utc_date",
    "model_version_id", "pred_home", "pred_away", "prob_home", "prob_draw",
    "prob_away", "prob_over25", "prob_btts", "lambda_home", "lambda_away",
    "reliability",
]


def init_db(db_path: Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_model_version(conn, params_json, backtest_rps, backtest_brier, notes, is_active):
    if is_active:
        conn.execute("UPDATE model_versions SET is_active = 0")
    cur = conn.execute(
        "INSERT INTO model_versions (params_json, backtest_rps, backtest_brier, notes, is_active)"
        " VALUES (?, ?, ?, ?, ?)",
        (params_json, backtest_rps, backtest_brier, notes, 1 if is_active else 0),
    )
    conn.commit()
    return cur.lastrowid


def get_active_model_version(conn):
    r = conn.execute("SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1").fetchone()
    return dict(r) if r else None


def all_model_versions(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM model_versions ORDER BY id")]


def save_prediction(conn, row: dict) -> int:
    cols = ", ".join(_PRED_FIELDS)
    placeholders = ", ".join("?" for _ in _PRED_FIELDS)
    values = [row[f] for f in _PRED_FIELDS]
    cur = conn.execute(
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"
        f" ON CONFLICT(match_id) DO UPDATE SET "
        + ", ".join(f"{f}=excluded.{f}" for f in _PRED_FIELDS if f != "match_id"),
        values,
    )
    conn.commit()
    return cur.lastrowid


def get_prediction(conn, match_id: int):
    r = conn.execute("SELECT * FROM predictions WHERE match_id = ?", (match_id,)).fetchone()
    return dict(r) if r else None


def save_result(conn, match_id, actual_home, actual_away, status):
    conn.execute(
        "INSERT INTO results (match_id, actual_home, actual_away, status) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(match_id) DO UPDATE SET actual_home=excluded.actual_home,"
        " actual_away=excluded.actual_away, status=excluded.status",
        (match_id, actual_home, actual_away, status),
    )
    conn.commit()


def predicted_match_ids_without_result(conn):
    rows = conn.execute(
        "SELECT p.match_id FROM predictions p"
        " LEFT JOIN results r ON r.match_id = p.match_id WHERE r.match_id IS NULL"
    ).fetchall()
    return [r["match_id"] for r in rows]


def predictions_with_results(conn):
    rows = conn.execute(
        "SELECT p.*, r.actual_home, r.actual_away FROM predictions p"
        " JOIN results r ON r.match_id = p.match_id"
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_storage.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/storage.py tests/test_storage.py
git commit -m "feat: SQLite storage for predictions, results, model versions"
```

---

### Task 5: Disk cache + football-data.org client

**Files:**
- Create: `backend/cache.py`, `backend/football_api.py`
- Test: `tests/test_cache.py`, `tests/test_football_api.py`

**Interfaces:**
- Consumes: `Settings` from `backend.config`.
- Produces:
  - `backend/cache.py`: `cache_get(cache_dir: Path, key: str, ttl_seconds: int) -> dict | None`; `cache_set(cache_dir: Path, key: str, value: dict) -> None`.
  - `backend/football_api.py`:
    - `Fetcher = Callable[[str, dict], dict]` (path, params → JSON dict).
    - `class FootballAPI` with `__init__(self, settings: Settings, fetcher: Fetcher | None = None)`. When `fetcher` is `None`, a default HTTP+cache fetcher is built from `settings`.
    - `get_matches(self) -> list[dict]` → list of normalized match dicts (see File Structure).
    - `get_upcoming_matches(self) -> list[dict]` → normalized matches with status in `{"SCHEDULED", "TIMED"}`, sorted by `utc_date`.
    - `get_finished_matches(self) -> list[dict]` → normalized matches with status `"FINISHED"`, sorted by `utc_date`.

- [ ] **Step 1: Write the failing cache test**

`tests/test_cache.py`:
```python
from backend import cache


def test_cache_set_then_get(tmp_path):
    cache.cache_set(tmp_path, "k", {"v": 1})
    assert cache.cache_get(tmp_path, "k", ttl_seconds=60) == {"v": 1}


def test_cache_miss_when_expired(tmp_path):
    cache.cache_set(tmp_path, "k", {"v": 1})
    assert cache.cache_get(tmp_path, "k", ttl_seconds=0) is None


def test_cache_miss_unknown_key(tmp_path):
    assert cache.cache_get(tmp_path, "absent", ttl_seconds=60) is None
```

- [ ] **Step 2: Run cache test to verify it fails**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.cache'`.

- [ ] **Step 3: Implement cache**

`backend/cache.py`:
```python
import hashlib
import json
import time
from pathlib import Path


def _path(cache_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def cache_set(cache_dir: Path, key: str, value: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    payload = {"ts": time.time(), "value": value}
    _path(cache_dir, key).write_text(json.dumps(payload))


def cache_get(cache_dir: Path, key: str, ttl_seconds: int) -> dict | None:
    path = _path(cache_dir, key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if time.time() - payload["ts"] > ttl_seconds:
        return None
    return payload["value"]
```

- [ ] **Step 4: Run cache test to verify it passes**

Run: `python -m pytest tests/test_cache.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the failing API client test**

`tests/test_football_api.py`:
```python
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


def test_get_finished_and_upcoming_split():
    api = _api()
    assert [m["id"] for m in api.get_finished_matches()] == [101]
    assert [m["id"] for m in api.get_upcoming_matches()] == [102]
```

- [ ] **Step 6: Run API client test to verify it fails**

Run: `python -m pytest tests/test_football_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.football_api'`.

- [ ] **Step 7: Implement the API client**

`backend/football_api.py`:
```python
from typing import Callable

import requests

from backend.cache import cache_get, cache_set
from backend.config import Settings

Fetcher = Callable[[str, dict], dict]

_UPCOMING = {"SCHEDULED", "TIMED"}


def _default_fetcher(settings: Settings) -> Fetcher:
    def fetch(path: str, params: dict) -> dict:
        key = f"{path}?{sorted(params.items())}"
        cached = cache_get(settings.cache_dir, key, settings.cache_ttl_seconds)
        if cached is not None:
            return cached
        resp = requests.get(
            f"{settings.base_url}{path}",
            params=params,
            headers={"X-Auth-Token": settings.api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        cache_set(settings.cache_dir, key, data)
        return data

    return fetch


def _normalize(m: dict) -> dict:
    full = m.get("score", {}).get("fullTime", {})
    return {
        "id": m["id"],
        "utc_date": m["utcDate"],
        "status": m["status"],
        "stage": m.get("stage", ""),
        "home_team": m["homeTeam"]["name"],
        "away_team": m["awayTeam"]["name"],
        "home_goals": full.get("home"),
        "away_goals": full.get("away"),
    }


class FootballAPI:
    def __init__(self, settings: Settings, fetcher: Fetcher | None = None):
        self.settings = settings
        self._fetch = fetcher or _default_fetcher(settings)

    def get_matches(self) -> list[dict]:
        path = f"/competitions/{self.settings.competition_code}/matches"
        data = self._fetch(path, {})
        return [_normalize(m) for m in data.get("matches", [])]

    def get_finished_matches(self) -> list[dict]:
        finished = [m for m in self.get_matches() if m["status"] == "FINISHED"]
        return sorted(finished, key=lambda m: m["utc_date"])

    def get_upcoming_matches(self) -> list[dict]:
        upcoming = [m for m in self.get_matches() if m["status"] in _UPCOMING]
        return sorted(upcoming, key=lambda m: m["utc_date"])
```

- [ ] **Step 8: Run both test files to verify they pass**

Run: `python -m pytest tests/test_cache.py tests/test_football_api.py -v`
Expected: 5 passed.

- [ ] **Step 9: Commit**

```bash
git add backend/cache.py backend/football_api.py tests/test_cache.py tests/test_football_api.py
git commit -m "feat: disk cache and football-data.org client"
```

---

### Task 6: Predictor orchestration

**Files:**
- Create: `backend/predictor.py`
- Test: `tests/test_predictor.py`

**Interfaces:**
- Consumes: `FootballAPI`, `storage`, `ModelParams`, `compute_league_model`, `predict_match`.
- Produces:
  - `params_from_version(version: dict | None) -> ModelParams` — parse `params_json`; fall back to `DEFAULT_PARAMS` when `version` is `None`.
  - `ensure_active_version(conn) -> dict` — if no active model version exists, create one from `DEFAULT_PARAMS` and return it.
  - `prediction_view(match: dict, p: MatchPrediction, version_id: int) -> dict` — build a JSON-serializable dict combining match fields + prediction fields (keys: `match_id`, `home_team`, `away_team`, `utc_date`, `stage`, `pred_home`, `pred_away`, `prob_home`, `prob_draw`, `prob_away`, `prob_over25`, `prob_btts`, `reliability`, `model_version_id`).
  - `predict_upcoming(api: FootballAPI, conn) -> list[dict]` — compute league model from finished matches, predict each upcoming match, persist each via `storage.save_prediction`, and return a list of `prediction_view` dicts.

- [ ] **Step 1: Write the failing test**

`tests/test_predictor.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.predictor'`.

- [ ] **Step 3: Write minimal implementation**

`backend/predictor.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/predictor.py tests/test_predictor.py
git commit -m "feat: predictor orchestration with persistence"
```

---

### Task 7: Result reconciliation

**Files:**
- Create: `backend/reconciler.py`
- Test: `tests/test_reconciler.py`

**Interfaces:**
- Consumes: `FootballAPI`, `storage`.
- Produces:
  - `reconcile(api: FootballAPI, conn) -> int` — for every finished match returned by the API whose id has a stored prediction but no stored result, save the real score via `storage.save_result`; return the count of newly reconciled matches.

- [ ] **Step 1: Write the failing test**

`tests/test_reconciler.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reconciler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.reconciler'`.

- [ ] **Step 3: Write minimal implementation**

`backend/reconciler.py`:
```python
from backend import storage
from backend.football_api import FootballAPI


def reconcile(api: FootballAPI, conn) -> int:
    pending = set(storage.predicted_match_ids_without_result(conn))
    if not pending:
        return 0
    count = 0
    for match in api.get_finished_matches():
        if match["id"] in pending and match["home_goals"] is not None:
            storage.save_result(
                conn, match["id"], match["home_goals"], match["away_goals"], match["status"]
            )
            count += 1
    return count
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reconciler.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/reconciler.py tests/test_reconciler.py
git commit -m "feat: reconcile predictions with real scores"
```

---

### Task 8: Self-improvement tuner (walk-forward backtest)

**Files:**
- Create: `backend/tuner.py`
- Test: `tests/test_tuner.py`

**Interfaces:**
- Consumes: `FootballAPI`, `storage`, `compute_league_model`, `predict_match`, `metrics.rps`, `metrics.brier`, `metrics.outcome_from_score`, `ModelParams`.
- Produces:
  - `backtest(finished_matches: list[dict], params: ModelParams, min_history: int = 3) -> tuple[float, float]` — walk-forward over chronologically sorted finished matches; for each match at index `k >= min_history`, build the league model from matches `[0:k]` only (no leakage), predict match `k`, and accumulate RPS and Brier vs the real outcome. Returns `(mean_rps, mean_brier)`; returns `(inf, inf)` if no match is scorable.
  - `PARAM_GRID: list[ModelParams]` — candidate hyperparameter combinations (home_advantage × shrinkage).
  - `tune(api: FootballAPI, conn, grid: list[ModelParams] | None = None) -> dict | None` — backtest each candidate, pick the lowest mean RPS, save it as a new active model version with its backtest scores, and return that version dict. Returns `None` if there is not enough history to score any candidate.

- [ ] **Step 1: Write the failing test**

`tests/test_tuner.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tuner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tuner'`.

- [ ] **Step 3: Write minimal implementation**

`backend/tuner.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tuner.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/tuner.py tests/test_tuner.py
git commit -m "feat: walk-forward tuner for self-improvement"
```

---

### Task 9: FastAPI app and endpoints

**Files:**
- Create: `backend/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: all backend modules.
- Produces:
  - `build_app(api: FootballAPI, conn) -> FastAPI` — factory taking an injected API client and DB connection (so tests avoid network/disk).
  - `compute_performance(conn) -> dict` — aggregate metrics over `storage.predictions_with_results(conn)`: keys `count`, `rps`, `brier`, `log_loss`, `exact_rate`, `outcome_accuracy`, `calibration` (list from `metrics.calibration_bins` built on home-win probability vs home-win occurrence), `by_version` (list of `{model_version_id, count, rps}`).
  - Routes: `GET /api/matches/upcoming`, `GET /api/performance`, `POST /api/reconcile`, `POST /api/tune`, plus static mounts (see Task 10).
  - `app` — module-level instance built from real `Settings` for `uvicorn backend.main:app`.

- [ ] **Step 1: Write the failing test**

`tests/test_api.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.main'`.

- [ ] **Step 3: Write minimal implementation**

`backend/main.py`:
```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import storage
from backend.config import PROJECT_ROOT, load_settings
from backend.football_api import FootballAPI
from backend.metrics import (
    brier,
    calibration_bins,
    exact_hit,
    log_loss,
    outcome_from_score,
    rps,
)
from backend.predictor import predict_upcoming
from backend.reconciler import reconcile
from backend.tuner import tune

FRONTEND_DIR = PROJECT_ROOT / "frontend"


def compute_performance(conn) -> dict:
    rows = storage.predictions_with_results(conn)
    n = len(rows)
    if n == 0:
        return {"count": 0, "rps": None, "brier": None, "log_loss": None,
                "exact_rate": None, "outcome_accuracy": None, "calibration": [], "by_version": []}

    rps_sum = brier_sum = ll_sum = exact = correct = 0.0
    calib_samples = []
    by_version: dict[int, list[float]] = {}

    for row in rows:
        probs = (row["prob_home"], row["prob_draw"], row["prob_away"])
        outcome = outcome_from_score(row["actual_home"], row["actual_away"])
        r = rps(probs, outcome)
        rps_sum += r
        brier_sum += brier(probs, outcome)
        ll_sum += log_loss(probs, outcome)
        exact += 1.0 if exact_hit((row["pred_home"], row["pred_away"]),
                                  (row["actual_home"], row["actual_away"])) else 0.0
        predicted_outcome = max(range(3), key=lambda i: probs[i])
        correct += 1.0 if predicted_outcome == outcome else 0.0
        calib_samples.append((row["prob_home"], 1 if outcome == 0 else 0))
        by_version.setdefault(row["model_version_id"], []).append(r)

    return {
        "count": n,
        "rps": rps_sum / n,
        "brier": brier_sum / n,
        "log_loss": ll_sum / n,
        "exact_rate": exact / n,
        "outcome_accuracy": correct / n,
        "calibration": calibration_bins(calib_samples),
        "by_version": [
            {"model_version_id": v, "count": len(rs), "rps": sum(rs) / len(rs)}
            for v, rs in sorted(by_version.items())
        ],
    }


def build_app(api: FootballAPI, conn) -> FastAPI:
    app = FastAPI(title="Pronostics Coupe du Monde 2026")

    @app.get("/api/matches/upcoming")
    def upcoming():
        reconcile(api, conn)
        return predict_upcoming(api, conn)

    @app.get("/api/performance")
    def performance():
        return compute_performance(conn)

    @app.post("/api/reconcile")
    def do_reconcile():
        return {"reconciled": reconcile(api, conn)}

    @app.post("/api/tune")
    def do_tune():
        version = tune(api, conn)
        if version is None:
            return {"model_version_id": None, "message": "Pas assez de matchs joués pour ré-ajuster."}
        return {"model_version_id": version["id"], "backtest_rps": version["backtest_rps"]}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


def _build_default_app() -> FastAPI:
    settings = load_settings()
    storage.init_db(settings.db_path)
    conn = storage.connect(settings.db_path)
    return build_app(FootballAPI(settings), conn)


app = _build_default_app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: all tests pass (config, poisson, metrics, storage, cache, football_api, predictor, reconciler, tuner, api).

- [ ] **Step 6: Commit**

```bash
git add backend/main.py tests/test_api.py
git commit -m "feat: FastAPI app, endpoints, and performance aggregation"
```

---

### Task 10: Frontend — match predictions page

**Files:**
- Create: `frontend/index.html`, `frontend/style.css`, `frontend/app.js`

**Interfaces:**
- Consumes: `GET /api/matches/upcoming`, `POST /api/reconcile`.
- Produces: a static page rendering one prediction card per upcoming match.

> Note: this task has no automated unit test (static assets). Verification is manual via the running server in Task 11. Keep JS dependency-free.

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Pronostics Coupe du Monde 2026</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <header>
    <h1>⚽ Pronostics — Coupe du Monde 2026</h1>
    <nav><a href="/">Matchs</a> · <a href="/performance.html">Performance</a></nav>
    <p class="disclaimer">Estimations probabilistes issues d'un modèle statistique (Poisson).
      Ce ne sont pas des garanties. Jouez de manière responsable.</p>
  </header>
  <main id="matches">Chargement…</main>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `frontend/style.css`**

```css
:root {
  --bg: #0b1020; --card: #151c33; --accent: #36c2a8; --text: #eef1f8;
  --muted: #93a0c0; --home: #4f8cff; --draw: #b0b8d0; --away: #ff7a59;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
header { padding: 1.5rem 1rem; text-align: center; border-bottom: 1px solid #222b48; }
header h1 { margin: 0 0 .25rem; font-size: 1.5rem; }
nav a { color: var(--accent); text-decoration: none; }
.disclaimer { color: var(--muted); font-size: .8rem; max-width: 640px; margin: .5rem auto 0; }
main { max-width: 720px; margin: 0 auto; padding: 1rem; display: grid; gap: 1rem; }
.card { background: var(--card); border-radius: 14px; padding: 1rem 1.25rem; }
.card .teams { display: flex; justify-content: space-between; align-items: baseline; font-weight: 600; }
.card .date { color: var(--muted); font-size: .8rem; }
.score { text-align: center; font-size: 2rem; font-weight: 700; margin: .25rem 0; }
.bar { display: flex; height: 26px; border-radius: 8px; overflow: hidden; margin: .5rem 0; }
.bar span { display: flex; align-items: center; justify-content: center; font-size: .72rem; color: #0b1020; }
.bar .h { background: var(--home); } .bar .d { background: var(--draw); } .bar .a { background: var(--away); }
.badges { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .5rem; }
.badge { background: #20294a; border-radius: 999px; padding: .25rem .6rem; font-size: .75rem; color: var(--muted); }
.reliability { font-size: .72rem; }
.reliability.faible { color: #ff7a59; } .reliability.moyen { color: #ffce5c; } .reliability.élevé { color: var(--accent); }
.empty { text-align: center; color: var(--muted); padding: 2rem; }
```

- [ ] **Step 3: Create `frontend/app.js`**

```javascript
const pct = (x) => `${(x * 100).toFixed(0)}%`;

function fmtDate(iso) {
  try { return new Date(iso).toLocaleString("fr-FR", { dateStyle: "medium", timeStyle: "short" }); }
  catch { return iso; }
}

function card(m) {
  return `
  <article class="card">
    <div class="teams"><span>${m.home_team}</span><span class="date">${fmtDate(m.utc_date)}</span><span>${m.away_team}</span></div>
    <div class="score">${m.pred_home} – ${m.pred_away}</div>
    <div class="bar">
      <span class="h" style="width:${m.prob_home * 100}%">${pct(m.prob_home)}</span>
      <span class="d" style="width:${m.prob_draw * 100}%">${pct(m.prob_draw)}</span>
      <span class="a" style="width:${m.prob_away * 100}%">${pct(m.prob_away)}</span>
    </div>
    <div class="badges">
      <span class="badge">+2.5 buts : ${pct(m.prob_over25)}</span>
      <span class="badge">BTTS : ${pct(m.prob_btts)}</span>
      <span class="badge reliability ${m.reliability}">Fiabilité : ${m.reliability}</span>
    </div>
  </article>`;
}

async function load() {
  const el = document.getElementById("matches");
  try {
    const res = await fetch("/api/matches/upcoming");
    const data = await res.json();
    if (!data.length) { el.innerHTML = '<p class="empty">Aucun match à venir pour le moment.</p>'; return; }
    el.innerHTML = data.map(card).join("");
  } catch (e) {
    el.innerHTML = '<p class="empty">Erreur de chargement. Vérifiez la clé API et le serveur.</p>';
  }
}

load();
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html frontend/style.css frontend/app.js
git commit -m "feat: frontend predictions page"
```

---

### Task 11: Performance dashboard, README, and manual verification

**Files:**
- Create: `frontend/performance.html`, `README.md`
- Modify: none (reuses `frontend/style.css`).

**Interfaces:**
- Consumes: `GET /api/performance`, `POST /api/tune`.
- Produces: a dashboard page + project documentation; a verified running app.

- [ ] **Step 1: Create `frontend/performance.html`**

```html
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Performance — Pronostics CdM 2026</title>
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <header>
    <h1>📊 Performance du modèle</h1>
    <nav><a href="/">Matchs</a> · <a href="/performance.html">Performance</a></nav>
  </header>
  <main>
    <div class="card" id="summary">Chargement…</div>
    <div class="card" id="versions"></div>
    <button id="tune" class="badge" style="cursor:pointer">🔧 Lancer l'auto-amélioration</button>
  </main>
  <script>
    const fmt = (x) => (x === null || x === undefined) ? "—" : Number(x).toFixed(3);
    const pctOrDash = (x) => (x === null || x === undefined) ? "—" : (x * 100).toFixed(0) + "%";

    async function load() {
      const p = await (await fetch("/api/performance")).json();
      document.getElementById("summary").innerHTML = `
        <h2>Vue d'ensemble (${p.count} matchs notés)</h2>
        <p>RPS moyen : <b>${fmt(p.rps)}</b> (plus bas = meilleur)</p>
        <p>Score de Brier : <b>${fmt(p.brier)}</b> · Log-loss : <b>${fmt(p.log_loss)}</b></p>
        <p>Taux de score exact : <b>${pctOrDash(p.exact_rate)}</b> · Précision 1N2 : <b>${pctOrDash(p.outcome_accuracy)}</b></p>`;
      document.getElementById("versions").innerHTML =
        "<h2>Par version de modèle</h2>" +
        (p.by_version.length
          ? p.by_version.map(v => `<p>v${v.model_version_id} — ${v.count} matchs — RPS ${fmt(v.rps)}</p>`).join("")
          : "<p>Aucune donnée encore. Les métriques apparaîtront après les premiers matchs joués.</p>");
    }

    document.getElementById("tune").addEventListener("click", async () => {
      const r = await (await fetch("/api/tune", { method: "POST" })).json();
      alert(r.model_version_id ? `Nouvelle version v${r.model_version_id} (RPS backtest ${fmt(r.backtest_rps)})` : r.message);
      load();
    });

    load();
  </script>
</body>
</html>
```

- [ ] **Step 2: Create `README.md`**

````markdown
# Pronostics Coupe du Monde 2026

Application web qui prédit les scores des matchs à venir de la Coupe du Monde 2026
(modèle de Poisson), mémorise chaque pronostic, le compare au score réel, et
s'auto-améliore en ré-ajustant ses hyperparamètres sur l'historique.

## Prérequis
- Python 3.11+
- Une clé API gratuite **football-data.org** : https://www.football-data.org/client/register

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigner FOOTBALL_DATA_API_KEY
```

## Lancer
```bash
set -a && source .env && set +a
uvicorn backend.main:app --reload
```
Ouvrir http://127.0.0.1:8000 (matchs) et http://127.0.0.1:8000/performance.html (performance).

## Endpoints
- `GET /api/matches/upcoming` — matchs à venir + pronostics
- `GET /api/performance` — métriques (RPS, Brier, calibration…)
- `POST /api/reconcile` — associer les scores réels aux pronostics
- `POST /api/tune` — lancer l'auto-amélioration (nouvelle version de modèle)

## Modèle & honnêteté statistique
Modèle de Poisson (Dixon-Coles simplifié). Sur un seul tournoi (~64 matchs),
l'auto-amélioration est réelle mais modeste : l'architecture est conçue pour
progresser sur la durée. Les pronostics sont des estimations probabilistes, pas
des garanties. Jouez de manière responsable.

## Tests
```bash
python -m pytest -v
```

## Compétition
Paramétrable via `COMPETITION_CODE` dans `.env` (défaut `WC`). Si la Coupe du Monde
n'est pas accessible sur votre plan, basculez sur un code disponible (ex. `FL1`,
`PL`) et redémarrez.
````

- [ ] **Step 3: Run the full test suite (final gate)**

Run: `python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Manual verification with a real API key**

```bash
set -a && source .env && set +a
uvicorn backend.main:app --reload
```
Checklist:
- `http://127.0.0.1:8000/` shows match cards (or the "Aucun match à venir" message — both are valid; off-season or knockout gaps can legitimately return zero upcoming matches).
- `http://127.0.0.1:8000/performance.html` loads and the "Lancer l'auto-amélioration" button returns a version or the "pas assez de matchs" message.
- If a 403 appears in the server logs, the `WC` competition is not on your plan → set `COMPETITION_CODE` to an available code in `.env` and restart (documented in README).

- [ ] **Step 5: Commit**

```bash
git add frontend/performance.html README.md
git commit -m "feat: performance dashboard and project README"
```

---

## Self-Review

**1. Spec coverage:**
- §2 data source (football-data.org, WC, rate limit, key in env) → Tasks 1, 5 (cache + client + auth header).
- §3 Poisson method (strengths, λ, matrix, market derivation, shrinkage, reliability) → Task 2.
- §3.6 knockout / 90-min result → encoded in outcome (1X2 on full-time score) Task 3; disclaimer in frontend Task 10.
- §4 memory & reconciliation → Tasks 4 (storage), 7 (reconciler).
- §5 self-improvement (walk-forward, hyperparameters, model versions) → Task 8; versions table Task 4.
- §6 metrics (RPS, Brier, log-loss, exact, accuracy, calibration, by-version) → Task 3 + aggregation Task 9.
- §7 architecture / decoupling → file structure honored across tasks.
- §8 data model (predictions, results, model_versions) → Task 4 schema.
- §9 data flow (lazy reconcile + predict + persist) → Task 9 `upcoming` route.
- §10 endpoints → Task 9.
- §11 frontend (cards + dashboard) → Tasks 10, 11.
- §12 tests → each task is TDD; tuner walk-forward leakage test in Task 8.
- §14 risks (WC availability, sparse data, no upcoming matches, rate limit) → README + manual verification Task 11; empty-state in frontend Task 10; cache Task 5.

No spec requirement is left without a task.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" placeholders; every code step contains complete code. Frontend tasks (10) explicitly note they are manually verified, with the verification defined in Task 11.

**3. Type consistency:** Normalized match dict keys (`home_team`, `away_team`, `home_goals`, `away_goals`, `utc_date`, `status`, `stage`, `id`) are identical across `football_api`, `predictor`, `reconciler`, `tuner`, and tests. Prediction row keys match `_PRED_FIELDS` in `storage` and the dict built in `predictor.predict_upcoming`. `ModelParams(home_advantage, shrinkage, max_goals)` is used consistently in `config`, `poisson_model`, `predictor`, `tuner`. Outcome encoding `0/1/2` is consistent between `metrics.outcome_from_score`, `poisson_model` triangle sums, and `main.compute_performance`. Function names (`compute_league_model`, `predict_match`, `predict_upcoming`, `reconcile`, `tune`, `backtest`, `build_app`, `compute_performance`) match between definitions and call sites.
