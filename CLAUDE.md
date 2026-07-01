# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A web app that predicts World Cup 2026 match scores from a Poisson model, stores every
prediction, reconciles it with the real result, and self-improves by re-tuning its
hyperparameters on the accumulated history. FastAPI backend + vanilla HTML/CSS/JS frontend
(served by the same app). French is the language for all user-facing copy, comments, and docs.

## Commands

```bash
# Setup (local dev uses SQLite; no Turso needed)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # set FOOTBALL_DATA_API_KEY

# Run locally (serves API + frontend on http://127.0.0.1:8000)
set -a && source .env && set +a
uvicorn backend.main:app --reload

# Tests (must stay green before pushing)
python -m pytest -q
python -m pytest tests/test_poisson_model.py -v          # one file
python -m pytest tests/test_api.py::test_history_endpoint_pairs_prediction_with_result -v  # one test
```

There is no linter/formatter configured. No CI runs tests automatically — run pytest locally.

## Architecture (the big picture)

Data flows in one direction, and the modules are layered by purity so the core is testable
without network or DB:

- **Pure core (no I/O, unit-tested directly):**
  - `backend/poisson_model.py` — team attack/defense strengths → expected goals → a full
    score-probability matrix, from which *every* market is derived (exact score, 1X2,
    Over/Under 2.5, BTTS). One matrix, all outputs.
  - `backend/metrics.py` — RPS (primary), Brier, log-loss, exact-hit, calibration bins.
    1X2 outcome encoding is fixed everywhere: `0 = home win, 1 = draw, 2 = away win`.
- **I/O layers:**
  - `backend/football_api.py` — football-data.org client. Takes an injectable `fetcher`
    (path, params → JSON); tests pass a fake fetcher, so **no network in tests**. The default
    fetcher wraps HTTP + disk cache. The whole app makes **one** upstream call
    (`/competitions/{code}/matches`), cached 60s → stays under the 10 req/min free-tier limit.
  - `backend/storage.py` — see "Dual-database" below.
  - `backend/cache.py` — disk cache (dir comes from `Settings.cache_dir`).
- **Orchestration:**
  - `backend/predictor.py` — build the league model from finished matches, predict each
    upcoming/live match, and **persist** the prediction.
  - `backend/reconciler.py` — attach real scores to stored predictions once matches finish;
    also **updates** a stored score if the source later corrects it (VAR/delayed update).
  - `backend/tuner.py` — the self-improvement loop: **walk-forward** backtest (only uses
    matches played before each target match — no data leakage) over a hyperparameter grid,
    minimizing mean RPS, saving the winner as a new active `model_version`.
  - `backend/main.py` — FastAPI app, endpoints, and `compute_performance` / `compute_history`
    aggregation. Serves the frontend via `StaticFiles`.

**The self-improvement loop is the point of the app:** predictions are stored with the
model version used → reconciled with real results → scored (RPS etc.) → hyperparameters
re-tuned → new model version. History/performance pages only ever show matches that were
predicted *and then* played + reconciled.

## Dual-database storage (most important non-obvious detail)

`backend/storage.py` is **driver-agnostic** and runs on two backends selected by env:

- **Local/tests:** plain SQLite (`data/app.db`, gitignored).
- **Production (Vercel):** Turso/libSQL, used when `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN`
  are set (`connect_turso`, lazy-imports `libsql_experimental`).

Rules that keep both working:
- All queries go through `_fetchall` / `_fetchone` / `_execute`, which build dict rows from
  `cursor.description` (works for both drivers) — **do not** reintroduce `sqlite3.Row`.
- **libSQL rejects list params** (`'list' object cannot be converted to 'PyTuple'`); the
  helpers coerce with `tuple(params)`. Never pass a list to `cursor.execute`.
- Schema is `CREATE TABLE IF NOT EXISTS` run statement-by-statement (`init_schema`) — libSQL
  has no `executescript`.
- The Turso path is **not** covered by the test suite (no local libSQL wheel on Python 3.9);
  it's only exercised in production. Validate DB changes against a live deploy.

Local and prod are **separate databases** — data seen locally is not in Turso and vice versa.

## Serverless / deployment constraints

- **Per-request connections:** `main.py` uses a `get_conn` factory (opened and closed per
  request), not a long-lived shared connection — required for serverless.
- **Read-only FS on Vercel** except `/tmp`: `config.load_settings` points `cache_dir` to
  `/tmp` when the `VERCEL` env var is present.
- **Vercel entrypoint** is declared in `pyproject.toml` (`[tool.vercel] entrypoint =
  "backend.main:app"`, with `[tool.uv] package = false`). There is no `vercel.json` — Vercel's
  native FastAPI framework routes everything to the app. Deploy dependencies live in
  `pyproject.toml` `[project.dependencies]`; `requirements.txt` is for local pip only.
- **Env vars** (set in Vercel, or `.env` locally): `FOOTBALL_DATA_API_KEY`,
  `COMPETITION_CODE` (default `WC`), and for prod `TURSO_DATABASE_URL` / `TURSO_AUTH_TOKEN`.

## Conventions

- Every `backend/*.py` starts with `from __future__ import annotations` — local Python is 3.9
  but the code uses 3.10+ union syntax (`X | None`). Keep this line when adding modules.
- Branch flow: work on `staging` → push (Vercel builds a Preview) → merge to `main` (Vercel
  Production at https://pronostique-foot.vercel.app). See `docs/DEPLOIEMENT.md`.
- Frontend has no build step; three pages (`index`, `history`, `performance`) share
  `style.css` and register a service worker (`sw.js`) for PWA install. `/api/*` is never cached.
