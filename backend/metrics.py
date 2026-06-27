from __future__ import annotations

import math


def outcome_from_score(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _one_hot(outcome: int) -> tuple[float, float, float]:
    return tuple(1.0 if i == outcome else 0.0 for i in range(3))  # type: ignore[return-value]


def rps(probs: tuple[float, float, float], outcome: int) -> float:
    e = _one_hot(outcome)
    cum_p = 0.0
    cum_e = 0.0
    total = 0.0
    for k in range(len(probs) - 1):
        cum_p += probs[k]
        cum_e += e[k]
        total += (cum_p - cum_e) ** 2
    return total / (len(probs) - 1)


def brier(probs: tuple[float, float, float], outcome: int) -> float:
    e = _one_hot(outcome)
    return sum((p - ei) ** 2 for p, ei in zip(probs, e))


def log_loss(probs: tuple[float, float, float], outcome: int) -> float:
    p = max(probs[outcome], 1e-15)
    return -math.log(p)


def exact_hit(pred: tuple[int, int], actual: tuple[int, int]) -> bool:
    return tuple(pred) == tuple(actual)


def calibration_bins(samples: list[tuple[float, int]], n_bins: int = 10) -> list[dict]:
    bins = []
    for b in range(n_bins):
        lower = b / n_bins
        upper = (b + 1) / n_bins
        in_bin = [s for s in samples if (lower <= s[0] < upper) or (b == n_bins - 1 and s[0] == 1.0)]
        count = len(in_bin)
        mean_pred = sum(s[0] for s in in_bin) / count if count else 0.0
        observed = sum(s[1] for s in in_bin) / count if count else 0.0
        bins.append({
            "lower": lower, "upper": upper, "count": count,
            "mean_pred": mean_pred, "observed": observed,
        })
    return bins
