"""Gate 3 / WDL math property tests."""

from __future__ import annotations

import math

import pytest

from engine.wdl import (
    centered_from_probs,
    centered_from_wdl,
    expected_from_probs,
    expected_from_wdl,
    wdl_to_probs,
)


def test_wdl_to_probs_normalizes():
    p_w, p_d, p_l = wdl_to_probs(76, 918, 6)
    assert abs(p_w + p_d + p_l - 1.0) < 1e-12
    assert p_w == pytest.approx(76 / 1000, abs=1e-9)


def test_property_expected_in_unit_interval():
    for wins, draws, losses in [
        (100, 0, 0),
        (0, 100, 0),
        (0, 0, 100),
        (50, 30, 20),
        (1, 1, 1),
        (800, 150, 50),
    ]:
        p = wdl_to_probs(wins, draws, losses)
        e = expected_from_probs(*p)
        s = centered_from_probs(*p)
        assert 0.0 <= e <= 1.0
        assert -1.0 <= s <= 1.0


def test_centered_extremes():
    assert centered_from_wdl(100, 0, 0) == pytest.approx(1.0)
    assert centered_from_wdl(0, 0, 100) == pytest.approx(-1.0)
    assert centered_from_wdl(0, 100, 0) == pytest.approx(0.0)
    assert centered_from_wdl(50, 0, 50) == pytest.approx(0.0)


def test_expected_half_for_draw():
    assert expected_from_wdl(1, 1, 1) == pytest.approx(0.5)


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        wdl_to_probs(-1, 2, 3)


def test_zero_sum_rejected():
    with pytest.raises(ValueError):
        wdl_to_probs(0, 0, 0)


def test_nan_rejected():
    with pytest.raises(ValueError):
        wdl_to_probs(math.nan, 1, 1)


def test_centered_from_wdl_consistency():
    s = centered_from_wdl(76, 918, 6)
    e = expected_from_wdl(76, 918, 6)
    assert s == pytest.approx(2 * e - 1)