from __future__ import annotations

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
