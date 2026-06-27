import math
import pytest
from backend.metrics import (
    outcome_from_score,
    rps,
    brier,
    log_loss,
    exact_hit,
    calibration_bins,
)


def test_outcome_from_score():
    assert outcome_from_score(2, 1) == 0
    assert outcome_from_score(1, 1) == 1
    assert outcome_from_score(0, 3) == 2


def test_rps_perfect_prediction_is_zero():
    assert rps((1.0, 0.0, 0.0), 0) == pytest.approx(0.0, abs=1e-12)


def test_rps_penalizes_distance_ordering():
    # predicting away when home happened (distance 2) is worse than predicting draw
    far = rps((0.0, 0.0, 1.0), 0)
    near = rps((0.0, 1.0, 0.0), 0)
    assert far > near


def test_brier_perfect_is_zero():
    assert brier((0.0, 1.0, 0.0), 1) == pytest.approx(0.0, abs=1e-12)


def test_brier_known_value():
    # probs (0.5,0.3,0.2), outcome home(0): (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25+0.09+0.04
    assert brier((0.5, 0.3, 0.2), 0) == pytest.approx(0.38, abs=1e-9)


def test_log_loss_perfect_is_near_zero():
    assert log_loss((1.0, 0.0, 0.0), 0) == pytest.approx(0.0, abs=1e-9)


def test_exact_hit():
    assert exact_hit((2, 1), (2, 1)) is True
    assert exact_hit((2, 1), (1, 1)) is False


def test_calibration_bins_basic():
    samples = [(0.05, 0), (0.15, 0), (0.95, 1), (0.85, 1)]
    bins = calibration_bins(samples, n_bins=10)
    populated = [b for b in bins if b["count"] > 0]
    assert sum(b["count"] for b in populated) == 4
    high = [b for b in populated if b["mean_pred"] > 0.5][0]
    assert high["observed"] == pytest.approx(1.0)
