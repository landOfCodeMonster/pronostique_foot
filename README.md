# Pronostics Coupe du Monde 2026

Application web qui prédit les scores des matchs à venir de la Coupe du Monde 2026
(modèle de Poisson), mémorise chaque pronostic, le compare au score réel, et
s'auto-améliore en ré-ajustant ses hyperparamètres sur l'historique.

## Ce que fait l'app

- **Score exact** le plus probable + probabilités **1N2** (victoire / nul / défaite)
- **+2.5 buts** et **les deux équipes marquent (BTTS)**
- **Indice de fiabilité** selon le nombre de matchs joués par chaque équipe
- **Mémoire** : chaque pronostic est stocké, puis réconcilié avec le score réel
- **Auto-amélioration** : ré-optimisation des hyperparamètres par backtest, avec
  versions de modèle comparables et suivi des métriques (RPS, Brier, calibration)

## Prérequis

- Python 3.9+ (testé sur 3.9 ; le code utilise `from __future__ import annotations`)
- Une clé API gratuite **football-data.org** : https://www.football-data.org/client/register

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis renseigner FOOTBALL_DATA_API_KEY
```

## Lancer

```bash
set -a && source .env && set +a
.venv/bin/python -m uvicorn backend.main:app --reload
```

- Matchs : http://127.0.0.1:8000
- Performance : http://127.0.0.1:8000/performance.html

## Endpoints

| Méthode | Route                     | Rôle                                            |
|---------|---------------------------|-------------------------------------------------|
| GET     | `/api/matches/upcoming`   | Matchs à venir + pronostics (et enregistrement) |
| GET     | `/api/performance`        | Métriques agrégées (RPS, Brier, calibration…)   |
| POST    | `/api/reconcile`          | Associer les scores réels aux pronostics        |
| POST    | `/api/tune`               | Lancer l'auto-amélioration (nouvelle version)   |

## Modèle & honnêteté statistique

Modèle de **Poisson** (Dixon-Coles simplifié) : forces d'attaque/défense calculées
sur les matchs joués → buts attendus → matrice des scores → tous les marchés.
Les pronostics de la phase à élimination directe portent sur le **temps réglementaire
(90 min)**.

Sur un seul tournoi (~64 matchs), l'auto-amélioration est **réelle mais modeste** :
l'architecture (mémoire + versions de modèle) est conçue pour progresser sur la durée.
Les pronostics sont des **estimations probabilistes, pas des garanties**. Jouez de
manière responsable.

## Tests

```bash
.venv/bin/python -m pytest -v
```

## Compétition

Paramétrable via `COMPETITION_CODE` dans `.env` (défaut `WC`). Si la Coupe du Monde
n'est pas accessible sur votre plan football-data.org, basculez sur un code disponible
(ex. `FL1` Ligue 1, `PL` Premier League) et redémarrez.

## Architecture

```
backend/
  config.py        Réglages + hyperparamètres du modèle
  poisson_model.py Cœur statistique (pur, testé sans réseau)
  metrics.py       RPS, Brier, log-loss, calibration (pur)
  storage.py       SQLite : predictions, results, model_versions
  cache.py         Cache disque (limite ~10 req/min)
  football_api.py  Client football-data.org
  predictor.py     Orchestration : forces → pronostic → persistance
  reconciler.py    Associe pronostics ↔ scores réels
  tuner.py         Backtest walk-forward + recherche d'hyperparamètres
  main.py          API FastAPI + service du frontend
frontend/          index.html, performance.html, style.css, app.js
```
