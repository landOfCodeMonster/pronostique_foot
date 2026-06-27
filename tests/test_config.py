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
