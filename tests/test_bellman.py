"""Gate 4 -- Bellman-equation and game-value target tests (spec Section 27).

The training-value Bellman operator under the alternating-perspective
convention is

    V(s_t) = r_t - gamma * V(s_{t+1})

with a MANDATORY negative sign on the next-state value (opponent
perspective).  These tests verify the sign, the fixed-point property, and
the game-value target semantics (zero-sum, bounds, truncation bootstrap).
"""

from __future__ import annotations

import pytest

from train.gae import game_value_targets, td_errors


# ------------------------------------------------- training-value Bellman
def test_bellman_fixed_point_gives_zero_td_truncated():
    # Construct values that exactly satisfy V_t = r_t - gamma*V_{t+1} with
    # the truncation bootstrap V_3 = 0.4 (stm perspective at boundary):
    #   V_2 = 0.3 - 0.9*0.4   = -0.06
    #   V_1 = -0.2 - 0.9*V_2  = -0.146
    #   V_0 =  0.1 - 0.9*V_1  =  0.2314
    gamma = 0.9
    boot = 0.4
    v2 = 0.3 - gamma * boot
    v1 = -0.2 - gamma * v2
    v0 = 0.1 - gamma * v1
    d = td_errors([0.1, -0.2, 0.3], [v0, v1, v2], gamma, False, boot)
    assert d == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_bellman_fixed_point_gives_zero_td_terminal():
    # V_2 = r_2, V_1 = r_1 - V_2, V_0 = r_0 - V_1 (gamma = 1, terminal).
    r = [0.3, -0.4, 0.6]
    v2 = r[2]
    v1 = r[1] - v2
    v0 = r[0] - v1
    d = td_errors(r, [v0, v1, v2], 1.0, True)
    assert d == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_opponent_perspective_sign_is_negative():
    # Verifies the exact equation delta_t = r_t - gamma*V_{t+1} - V_t on a
    # case where a positive sign would produce a clearly different number.
    # delta_0 = 0.5 - 0.5*0.8 - 0.1 = 0.0  (correct, negative sign)
    # wrong:   0.5 + 0.5*0.8 - 0.1 = 0.8
    d = td_errors([0.5], [0.1], 0.5, False, 0.8)
    assert d == pytest.approx([0.0])
    assert d[0] != pytest.approx(0.8)


def test_terminal_td_has_no_bootstrap_term():
    # delta_last = r_last - V_last exactly (spec Section 29).
    d = td_errors([0.7], [0.25], 0.99, True)
    assert d == pytest.approx([0.7 - 0.25])


def test_truncated_td_uses_negative_bootstrap():
    # delta_last = r_last - gamma*bootstrap - V_last  (sign from Sec. 27).
    d = td_errors([0.7], [0.25], 0.5, False, 0.3)
    assert d == pytest.approx([0.7 - 0.5 * 0.3 - 0.25])


# ------------------------------------------------- game-value semantics
def test_game_value_zero_sum_between_plies():
    # White wins: White plies get +1, Black plies get -1; z_B = -z_W.
    t = game_value_targets(4, True, 1.0, z_white=1.0)
    assert t == [1.0, -1.0, 1.0, -1.0]
    for a, b in zip(t, t[1:]):
        assert b == -a


def test_game_value_black_win_and_draw():
    assert game_value_targets(3, True, 1.0, z_white=-1.0) == [-1.0, 1.0, -1.0]
    assert game_value_targets(3, True, 1.0, z_white=0.0) == [0.0, 0.0, 0.0]


def test_game_value_targets_bounded_terminal():
    for z in (-1.0, 0.0, 1.0):
        for target in game_value_targets(10, True, 1.0, z_white=z):
            assert -1.0 <= target <= 1.0


def test_truncation_is_not_scored_as_a_draw():
    # Spec Section 50: bootstrap the boundary value; do NOT invent z = 0.
    # gamma=1, n=3, V_game_stm(s_3)=0.8 (opponent of last mover is doing
    # well): targets = (-1)^(3-t) * 0.8 = [-0.8, 0.8, -0.8].
    t = game_value_targets(3, False, 1.0, bootstrap_game_value=0.8)
    assert t == pytest.approx([-0.8, 0.8, -0.8])
    assert 0.0 not in t


def test_truncated_game_value_targets_alternate_and_bounded():
    # gamma=0.5, n=4, bootstrap=-0.5 (V_game_stm at boundary s_4):
    #   target_t = (-0.5)^(4-t) * (-0.5):
    #   t=0: (-0.5)^4 * -0.5 = -0.03125
    #   t=1: (-0.5)^3 * -0.5 =  0.0625
    #   t=2: (-0.5)^2 * -0.5 = -0.125
    #   t=3: (-0.5)^1 * -0.5 =  0.25   (last mover sees -V_stm(s_4))
    t = game_value_targets(4, False, 0.5, bootstrap_game_value=-0.5)
    assert t == pytest.approx([-0.03125, 0.0625, -0.125, 0.25])
    assert all(-1.0 <= x <= 1.0 for x in t)


def test_game_value_invalid_inputs_raise():
    with pytest.raises(ValueError):
        game_value_targets(3, True, 1.0, z_white=0.5)  # invalid z
    with pytest.raises(ValueError):
        game_value_targets(3, True, 1.0, z_white=1.0, bootstrap_game_value=0.1)
    with pytest.raises(ValueError):
        game_value_targets(3, False, 1.0)  # truncated without bootstrap
    with pytest.raises(ValueError):
        game_value_targets(3, False, 1.0, z_white=1.0)  # z on truncated
    with pytest.raises(ValueError):
        game_value_targets(3, False, 1.0, bootstrap_game_value=1.5)  # range
    with pytest.raises(ValueError):
        game_value_targets(3, True, 1.2, z_white=1.0)  # gamma out of range
    with pytest.raises(ValueError):
        game_value_targets(3, True, 1.0, z_white=float("nan"))
