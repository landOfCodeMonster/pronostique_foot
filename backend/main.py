from __future__ import annotations

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


def compute_history(conn) -> list[dict]:
    """Finished, reconciled matches we predicted — prediction vs real score."""
    items = []
    for row in storage.predictions_with_results(conn):
        probs = (row["prob_home"], row["prob_draw"], row["prob_away"])
        outcome = outcome_from_score(row["actual_home"], row["actual_away"])
        predicted_outcome = max(range(3), key=lambda i: probs[i])
        items.append({
            "match_id": row["match_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "utc_date": row["match_utc_date"],
            "pred_home": row["pred_home"],
            "pred_away": row["pred_away"],
            "actual_home": row["actual_home"],
            "actual_away": row["actual_away"],
            "prob_home": row["prob_home"],
            "prob_draw": row["prob_draw"],
            "prob_away": row["prob_away"],
            "reliability": row["reliability"],
            "model_version_id": row["model_version_id"],
            "outcome_correct": predicted_outcome == outcome,
            "exact": exact_hit((row["pred_home"], row["pred_away"]),
                               (row["actual_home"], row["actual_away"])),
            "rps": rps(probs, outcome),
        })
    items.sort(key=lambda x: x["utc_date"], reverse=True)
    return items


def _close(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def build_app(api: FootballAPI, get_conn) -> FastAPI:
    """`get_conn` is a factory returning a fresh DB connection per request
    (required for serverless: no long-lived shared connection)."""
    app = FastAPI(title="Pronostics Coupe du Monde 2026")

    @app.get("/api/matches/upcoming")
    def upcoming():
        conn = get_conn()
        try:
            reconcile(api, conn)
            return predict_upcoming(api, conn)
        finally:
            _close(conn)

    @app.get("/api/matches/history")
    def history():
        conn = get_conn()
        try:
            reconcile(api, conn)
            return compute_history(conn)
        finally:
            _close(conn)

    @app.get("/api/performance")
    def performance():
        conn = get_conn()
        try:
            return compute_performance(conn)
        finally:
            _close(conn)

    @app.post("/api/reconcile")
    def do_reconcile():
        conn = get_conn()
        try:
            return {"reconciled": reconcile(api, conn)}
        finally:
            _close(conn)

    @app.post("/api/tune")
    def do_tune():
        conn = get_conn()
        try:
            version = tune(api, conn)
        finally:
            _close(conn)
        if version is None:
            return {"model_version_id": None, "message": "Pas assez de matchs joués pour ré-ajuster."}
        return {"model_version_id": version["id"], "backtest_rps": version["backtest_rps"]}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    return app


def _make_conn_factory(settings):
    if settings.turso_url:
        return lambda: storage.connect_turso(settings.turso_url, settings.turso_token)
    return lambda: storage.connect(settings.db_path)


def _build_default_app() -> FastAPI:
    settings = load_settings()
    get_conn = _make_conn_factory(settings)
    # Ensure the schema exists once at startup (idempotent: CREATE IF NOT EXISTS).
    conn = get_conn()
    storage.init_schema(conn)
    _close(conn)
    return build_app(FootballAPI(settings), get_conn)


app = _build_default_app()
