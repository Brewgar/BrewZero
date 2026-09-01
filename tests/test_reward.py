"""Gate 3 / Reward mathematics tests (manually specified values)."""

from __future__ import annotations

import math

import pytest

from engine.reward import (
    compute_reward_components,
    dense_reward_components,
    stockfish_dense_reward,
    terminal_game_reward,
    training_reward,
)


def test_delta_reward_clipping():
    # delta_score=+5, coef .1 -> .5; clamped to rmax .2.
    dr, rr = dense_reward_components(5.0, 0.5, 0.1, 0.1, 1.0, rmax=0.2)
    assert dr == pytest.approx(0.2)
    assert rr == pytest.approx(-0.1 * math.tanh(0.5))


def test_regret_reward_sign():
    # Large regret -> regret_reward approaches -regret_coef.
    dr, rr = dense_reward_components(0.0, 10.0, 0.0, 0.1, 1.0, rmax=1.0)
    assert rr < 0
    assert rr == pytest.approx(-0.1 * math.tanh(10.0))


def test_regret_zero_is_zero():
    dr, rr = dense_reward_components(0.0, 0.0, 0.1, 0.1, 1.0, rmax=1.0)
    assert rr == 0.0


def test_tiny_negative_regret_kept():
    # Spec 20: tiny negative regret values are possible and not clamped to 0.
    dr, rr = dense_reward_components(0.0, -0.02, 0.1, 0.1, 1.0, rmax=1.0)
    assert rr > 0  # tanh of a tiny negative value minus-negated gives small positive
    assert rr == pytest.approx(0.1 * math.tanh(0.02))


def test_combined_sf_reward():
    assert stockfish_dense_reward(0.3, -0.2) == pytest.approx(0.1)


def test_terminal_reward_values():
    assert terminal_game_reward(1.0) == 1.0
    assert terminal_game_reward(0.0) == 0.0
    assert terminal_game_reward(-1.0) == -1.0


def test_terminal_reward_rejects_bad_values():
    with pytest.raises(ValueError):
        terminal_game_reward(0.5)
    with pytest.raises(ValueError):
        terminal_game_reward(2.0)


def test_training_reward_linear_combination():
    assert training_reward(0.4, 1.0, lambda_stockfish=0.25, lambda_game=1.0) == pytest.approx(1.1)


def test_full_components_nonterminal():
    c = compute_reward_components(
        delta_score=0.5,
        engine_regret=1.0,
        terminal_z_mover=None,
        delta_coef=0.1,
        regret_coef=0.1,
        regret_tau=1.0,
        rmax=0.5,
        lambda_stockfish=0.25,
        lambda_game=1.0,
    )
    assert c.terminal_game_reward == 0.0
    assert c.stockfish_dense_reward == pytest.approx(c.delta_reward + c.regret_reward)
    assert c.total_training_reward == pytest.approx(
        0.25 * c.stockfish_dense_reward + 0.0
    )
    assert c.delta_reward == pytest.approx(0.05)
    assert c.regret_reward == pytest.approx(-0.1 * math.tanh(1.0))


def test_full_components_terminal_win():
    c = compute_reward_components(
        delta_score=0.2,
        engine_regret=0.0,
        terminal_z_mover=1.0,
        delta_coef=0.1,
        regret_coef=0.1,
        regret_tau=1.0,
        rmax=0.5,
        lambda_stockfish=0.0,
        lambda_game=1.0,
    )
    assert c.terminal_game_reward == 1.0
    assert c.total_training_reward == pytest.approx(1.0)


def test_terminal_only_configuration():
    # Experiment A: r_train == r_game.
    c = compute_reward_components(
        delta_score=0.0,
        engine_regret=0.0,
        terminal_z_mover=None,
        delta_coef=0.0,
        regret_coef=0.0,
        regret_tau=1.0,
        rmax=1.0,
        lambda_stockfish=0.0,
        lambda_game=1.0,
    )
    assert c.total_training_reward == 0.0
    assert c.stockfish_dense_reward == 0.0