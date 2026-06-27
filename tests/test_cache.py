from backend import cache


def test_cache_set_then_get(tmp_path):
    cache.cache_set(tmp_path, "k", {"v": 1})
    assert cache.cache_get(tmp_path, "k", ttl_seconds=60) == {"v": 1}


def test_cache_miss_when_expired(tmp_path):
    cache.cache_set(tmp_path, "k", {"v": 1})
    assert cache.cache_get(tmp_path, "k", ttl_seconds=0) is None


def test_cache_miss_unknown_key(tmp_path):
    assert cache.cache_get(tmp_path, "absent", ttl_seconds=60) is None
