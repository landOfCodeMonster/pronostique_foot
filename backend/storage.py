from __future__ import annotations

import sqlite3
from pathlib import Path

# Schema as individual statements (libSQL has no executescript).
SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        params_json TEXT NOT NULL,
        backtest_rps REAL,
        backtest_brier REAL,
        notes TEXT,
        is_active INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER UNIQUE NOT NULL,
        competition TEXT, home_team TEXT, away_team TEXT, match_utc_date TEXT,
        model_version_id INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        pred_home INTEGER, pred_away INTEGER,
        prob_home REAL, prob_draw REAL, prob_away REAL,
        prob_over25 REAL, prob_btts REAL,
        lambda_home REAL, lambda_away REAL, reliability TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS results (
        match_id INTEGER PRIMARY KEY,
        actual_home INTEGER, actual_away INTEGER,
        status TEXT, reconciled_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""",
]

_PRED_FIELDS = [
    "match_id", "competition", "home_team", "away_team", "match_utc_date",
    "model_version_id", "pred_home", "pred_away", "prob_home", "prob_draw",
    "prob_away", "prob_over25", "prob_btts", "lambda_home", "lambda_away",
    "reliability",
]


# --- Driver-agnostic helpers (work for both sqlite3 and libSQL/Turso) ---
def _fetchall(conn, sql, params=()):
    cur = conn.cursor()
    # libSQL requires a tuple (it rejects lists); SQLite accepts both.
    cur.execute(sql, tuple(params))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetchone(conn, sql, params=()):
    rows = _fetchall(conn, sql, params)
    return rows[0] if rows else None


def _execute(conn, sql, params=()):
    cur = conn.cursor()
    # libSQL requires a tuple (it rejects lists); SQLite accepts both.
    cur.execute(sql, tuple(params))
    conn.commit()
    return cur


def _lastrowid(conn, cur):
    rid = getattr(cur, "lastrowid", None)
    if rid:
        return rid
    row = _fetchone(conn, "SELECT last_insert_rowid() AS id")
    return row["id"] if row else None


# --- Schema & connections ---
def init_schema(conn) -> None:
    cur = conn.cursor()
    for stmt in SCHEMA_STATEMENTS:
        cur.execute(stmt)
    conn.commit()


def connect(db_path) -> sqlite3.Connection:
    # Local / tests: plain SQLite. check_same_thread=False because FastAPI runs
    # sync routes in a worker thread. No row_factory: dicts are built via the
    # cursor.description helpers above, which also work for libSQL.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False)


def init_db(db_path) -> None:
    conn = connect(db_path)
    init_schema(conn)
    conn.close()


def connect_turso(url: str, token: str):
    # Remote-only connection to Turso (no local file → safe on serverless).
    import libsql_experimental as libsql  # lazy: only needed when Turso is configured
    return libsql.connect(database=url, auth_token=token)


# --- Model versions ---
def save_model_version(conn, params_json, backtest_rps, backtest_brier, notes, is_active):
    if is_active:
        _execute(conn, "UPDATE model_versions SET is_active = 0")
    cur = _execute(
        conn,
        "INSERT INTO model_versions (params_json, backtest_rps, backtest_brier, notes, is_active)"
        " VALUES (?, ?, ?, ?, ?)",
        (params_json, backtest_rps, backtest_brier, notes, 1 if is_active else 0),
    )
    return _lastrowid(conn, cur)


def get_active_model_version(conn):
    return _fetchone(
        conn, "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
    )


def all_model_versions(conn):
    return _fetchall(conn, "SELECT * FROM model_versions ORDER BY id")


# --- Predictions ---
def save_prediction(conn, row: dict) -> int:
    cols = ", ".join(_PRED_FIELDS)
    placeholders = ", ".join("?" for _ in _PRED_FIELDS)
    values = [row[f] for f in _PRED_FIELDS]
    cur = _execute(
        conn,
        f"INSERT INTO predictions ({cols}) VALUES ({placeholders})"
        f" ON CONFLICT(match_id) DO UPDATE SET "
        + ", ".join(f"{f}=excluded.{f}" for f in _PRED_FIELDS if f != "match_id"),
        values,
    )
    return _lastrowid(conn, cur)


def get_prediction(conn, match_id: int):
    return _fetchone(conn, "SELECT * FROM predictions WHERE match_id = ?", (match_id,))


# --- Results ---
def save_result(conn, match_id, actual_home, actual_away, status):
    _execute(
        conn,
        "INSERT INTO results (match_id, actual_home, actual_away, status) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(match_id) DO UPDATE SET actual_home=excluded.actual_home,"
        " actual_away=excluded.actual_away, status=excluded.status",
        (match_id, actual_home, actual_away, status),
    )


def get_result(conn, match_id: int):
    return _fetchone(conn, "SELECT * FROM results WHERE match_id = ?", (match_id,))


def predicted_match_ids_without_result(conn):
    rows = _fetchall(
        conn,
        "SELECT p.match_id AS match_id FROM predictions p"
        " LEFT JOIN results r ON r.match_id = p.match_id WHERE r.match_id IS NULL",
    )
    return [r["match_id"] for r in rows]


def predictions_with_results(conn):
    return _fetchall(
        conn,
        "SELECT p.*, r.actual_home AS actual_home, r.actual_away AS actual_away"
        " FROM predictions p JOIN results r ON r.match_id = p.match_id",
    )
