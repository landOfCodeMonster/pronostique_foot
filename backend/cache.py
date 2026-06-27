from __future__ import annotations

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
