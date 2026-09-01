"""Gate 4 -- GAE tests (spec Section 79).

Every expected number below was computed BY HAND from the definitions

    delta_t = r_t - gamma * V(s_{t+1}) - V(s_t)          (stm perspective)
    A_t     = sum_l (-gamma*lambda)^l * delta_{t+l}

so that a sign error in the implementation cannot pass unnoticed.
Trajectories of >= 3 consecutive plies are used so that the alternating
perspective signs are actually exercised.
"""

from __future__ import annotations

import math

import pytest

from train.gae import gae, return_targets, td_errors


def test_td_errors_manual_three_plies_truncated():
    # gamma=1, bootstrap V_stm(s_3)=0.25 (opponent perspective at boundary).
    # delta_0 = 0.2 - 0.3 - 0.1        = -0.2
    # delta_1 = -0.1 - (-0.2) - 0.3    = -0.2
    # delta_2 = 0.5 - 0.25 - (-0.2)    =  0.45
    d = td_errors([0.2, -0.1, 0.5], [0.1, 0.3, -0.2], 1.0, False, 0.25)
    assert d == pytest.approx([-0.2, -0.2, 0.45])


def test_td_errors_manual_three_plies_terminal():
    # Terminal: no bootstrap -> delta_2 = 0.5 - 0 - (-0.2) = 0.7.
    d = td_errors([0.2, -0.1, 0.5], [0.1, 0.3, -0.2], 1.0, True)
    assert d == pytest.approx([-0.2, -0.2, 0.7])


def test_td_error_sign_convention_detects_wrong_sign():
    # If the next-state value were added (+gamma*V_{t+1}) instead of
    # subtracted, delta_0 would be 0.3 + 0.2 - 0.1 = 0.4.  The mandatory
    # opponent-perspective negative sign (spec Sec. 27) gives 0.0.
    d = td_errors([0.3], [0.1], 1.0, False, 0.2)
    assert d == pytest.approx([0.0])
    assert d[0] != pytest.approx(0.4)


def test_gae_manual_lambda_half_truncated():
    # Backward recursion A_t = delta_t + (-gamma*lambda)*A_{t+1}:
    #   A_2 = 0.45
    #   A_1 = -0.2 - 0.5*0.45      = -0.425
    #   A_0 = -0.2 - 0.5*(-0.425)  =  0.0125
    a = gae([0.2, -0.1, 0.5], [0.1, 0.3, -0.2], 1.0, 0.5, False, 0.25)
    assert a == pytest.approx([0.0125, -0.425, 0.45])


def test_gae_manual_lambda_half_terminal():
    # delta = [-0.2, -0.2, 0.7]:
    #   A_2 = 0.7
    #   A_1 = -0.2 - 0.5*0.7   = -0.55
    #   A_0 = -0.2 - 0.5*(-0.55) = 0.075
    a = gae([0.2, -0.1, 0.5], [0.1, 0.3, -0.2], 1.0, 0.5, True)
    assert a == pytest.approx([0.075, -0.55, 0.7])


def test_gae_lambda1_equals_alternating_monte_carlo():
    # With lambda=1, gamma=1: A_t = sum_l (-1)^l r_{t+l} - V_t.
    r = [0.2, -0.1, 0.5]
    v = [0.1, 0.3, -0.2]
    a = gae(r, v, 1.0, 1.0, True)
    for t in range(3):
        # Sign is RELATIVE to the mover: (-1)^(l - t).
        mc = sum(((-1.0) ** (l - t)) * r[l] for l in range(t, 3)) - v[t]
        assert a[t] == pytest.approx(mc)
    assert a == pytest.approx([0.7, -0.9, 0.7])


def test_gae_alternating_sign_is_mandatory():
    # r=[1,1,1], V=[0,0,0], terminal: the mover-perspective return of ply 0
    # is 1 - 1 + 1 = 1 (alternating), NOT 3 (naive positive sum).
    a = gae([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 1.0, 1.0, True)
    assert a == pytest.approx([1.0, 0.0, 1.0])


def test_gae_lambda0_equals_td_error():
    r = [0.2, -0.1, 0.5]
    v = [0.1, 0.3, -0.2]
    d = td_errors(r, v, 1.0, False, 0.25)
    a = gae(r, v, 1.0, 0.0, False, 0.25)
    assert a == pytest.approx(d)


def test_gae_discounted_terminal_manual():
    # gamma=0.5, terminal, r=[1,1,1], V=0:
    #   A_2 = 1
    #   A_1 = 1 - 0.5*1       = 0.5
    #   A_0 = 1 - 0.5*0.5     = 0.75
    # Monte-Carlo check: 1 - 0.5 + 0.25 = 0.75.
    a = gae([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.5, 1.0, True)
    assert a == pytest.approx([0.75, 0.5, 1.0])


def test_return_targets_are_advantage_plus_old_value():
    r = [0.2, -0.1, 0.5]
    v = [0.1, 0.3, -0.2]
    a = gae(r, v, 1.0, 0.5, False, 0.25)
    tgt = return_targets(r, v, 1.0, 0.5, False, 0.25)
    assert tgt == pytest.approx([ai + vi for ai, vi in zip(a, v)])
    assert tgt == pytest.approx([0.1125, -0.125, 0.25])


def test_return_targets_use_training_not_game_semantics():
    # Dense training rewards may exceed [-1, 1]; the training-value target
    # must follow the training reward (spec Section 30), so a return target
    # outside [-1, 1] is legitimate and must not be clipped.
    tgt = return_targets([2.5, -2.0, 3.0], [0.0, 0.0, 0.0], 1.0, 1.0, True)
    assert tgt[0] == pytest.approx(2.5 - (-2.0) + 3.0)


# ------------------------------------------------------------------ errors
def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        td_errors([0.1, 0.2], [0.1], 1.0, True)


def test_terminal_with_bootstrap_raises():
    with pytest.raises(ValueError):
        td_errors([0.1], [0.0], 1.0, True, 0.5)


def test_truncated_without_bootstrap_raises():
    with pytest.raises(ValueError):
        td_errors([0.1], [0.0], 1.0, False, None)


def test_nan_reward_raises():
    with pytest.raises(ValueError):
        td_errors([float("nan")], [0.0], 1.0, True)


def test_inf_value_raises():
    with pytest.raises(ValueError):
        gae([0.1], [float("inf")], 1.0, 0.95, True)


def test_nan_bootstrap_raises():
    with pytest.raises(ValueError):
        td_errors([0.1], [0.0], 1.0, False, float("nan"))


def test_all_outputs_finite_on_finite_inputs():
    r = [0.3, -1.7, 0.9, 0.0]
    v = [0.4, -0.2, 1.2, -0.9]
    for term, boot in ((True, None), (False, 0.33)):
        d = td_errors(r, v, 0.99, term, boot)
        a = gae(r, v, 0.99, 0.95, term, boot)
        t = return_targets(r, v, 0.99, 0.95, term, boot)
        for seq in (d, a, t):
            assert all(math.isfinite(x) for x in seq)
