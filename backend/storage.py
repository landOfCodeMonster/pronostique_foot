from __future__ import annotations

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
    # check_same_thread=False: FastAPI runs sync routes in a worker thread, so the
    # connection is used from a different thread than the one that created it. Safe
    # here because this is a low-traffic, single-user app with serialized access.
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
