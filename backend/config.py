from __future__ import annotations

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
    # When set, storage uses Turso (libSQL) instead of local SQLite (for Vercel).
    turso_url: str
    turso_token: str


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = env if env is not None else os.environ
    # On Vercel the project filesystem is read-only; only /tmp is writable.
    on_vercel = bool(env.get("VERCEL"))
    cache_dir = Path("/tmp/pf-cache") if on_vercel else PROJECT_ROOT / "data" / "cache"
    return Settings(
        api_key=env.get("FOOTBALL_DATA_API_KEY", ""),
        competition_code=env.get("COMPETITION_CODE", "WC"),
        base_url="https://api.football-data.org/v4",
        db_path=PROJECT_ROOT / "data" / "app.db",
        cache_dir=cache_dir,
        # 60s keeps live scores fresh while capping upstream calls at ~1/min
        # (well under football-data.org's 10 requests/minute limit).
        cache_ttl_seconds=60,
        turso_url=env.get("TURSO_DATABASE_URL", ""),
        turso_token=env.get("TURSO_AUTH_TOKEN", ""),
    )
