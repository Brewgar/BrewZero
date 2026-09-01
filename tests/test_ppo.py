"""Gate 6 -- PPO tests (spec Sections 14, 31-37, 77, 80).

Verifies: ratio identity (old == new => rho == 1), clipping on both sides,
masked-probability invariants (illegal = 0, legal sum = 1), entropy over
legal actions only, finite gradients, and the hard-fail behavior on NaN /
no-legal-action inputs.
"""

from __future__ import annotations

import math

import chess
import numpy as np
import pytest
import torch

from env.action_space import ACTION_SPACE_SIZE, ActionCodec
from env.encoding import encode_state
from model.network import ChessNet
from train.losses import (
    entropy,
    masked_log_softmax,
    ppo_policy_loss,
    total_loss,
    value_loss,
)
from train.ppo import NonFiniteLossError, PPOConfig, ppo_update


# ------------------------------------------------------- masked softmax
def test_illegal_actions_zero_prob_and_legal_sum_one():
    torch.manual_seed(0)
    logits = torch.randn(7, ACTION_SPACE_SIZE)
    mask = torch.zeros(7, ACTION_SPACE_SIZE, dtype=torch.bool)
    for i in range(7):
        legal = torch.randperm(ACTION_SPACE_SIZE)[: torch.randint(1, 50, (1,)).item()]
        mask[i, legal] = True
    lp = masked_log_softmax(logits, mask)
    probs = lp.exp()
    assert (probs[~mask] == 0.0).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(7), atol=1e-6)


def test_masked_softmax_starting_position_sum_one():
    board = chess.Board()  # canonical: White to move at start
    legal = ActionCodec.encode_legal_moves(board)
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, legal] = True
    logits = torch.randn(1, ACTION_SPACE_SIZE)
    probs = masked_log_softmax(logits, mask).exp()
    assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-6)
    assert (probs[0, legal] > 0).all()
    assert probs[0].sum().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------- ratio / clipping
def test_ratio_one_when_policies_equal():
    logp = torch.tensor([-1.2, -0.4, -3.1])
    adv = torch.tensor([0.5, -0.2, 1.0])
    loss, clip_frac, kl = ppo_policy_loss(logp, logp.clone(), adv, 0.2)
    assert loss.item() == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert clip_frac.item() == pytest.approx(0.0)
    assert kl.item() == pytest.approx(0.0, abs=1e-7)


def test_clipping_when_ratio_above_upper_bound():
    # ratio = 2 > 1 + eps, positive advantage -> clipped branch wins:
    # loss = -(1 + eps) * mean(A)
    logp_old = torch.tensor([-1.0, -1.0])
    logp_new = logp_old + math.log(2.0)
    adv = torch.tensor([1.0, 1.0])
    loss, clip_frac, _ = ppo_policy_loss(logp_new, logp_old, adv, 0.2)
    assert loss.item() == pytest.approx(-1.2, abs=1e-6)
    assert clip_frac.item() == pytest.approx(1.0)


def test_clipping_when_ratio_below_lower_bound():
    # ratio = 0.5 < 1 - eps, negative advantage -> the clipped branch is
    # the min: min(0.5A, 0.8A) = 0.8A = -0.8 -> loss = -(-0.8) = +0.8.
    # (The clipped branch has zero gradient, which is the point of PPO.)
    logp_old = torch.tensor([-1.0, -1.0])
    logp_new = logp_old - math.log(2.0)
    adv = torch.tensor([-1.0, -1.0])
    loss, clip_frac, _ = ppo_policy_loss(logp_new, logp_old, adv, 0.2)
    assert loss.item() == pytest.approx(0.8, abs=1e-6)
    assert clip_frac.item() == pytest.approx(1.0)


def test_ratio_uses_stored_old_log_probs():
    # The old log-prob must be the STORED behavior-policy value (Sec. 52):
    # feeding a deliberately different old value must change the result.
    logp_new = torch.tensor([-1.0])
    adv = torch.tensor([1.0])
    l1, _, _ = ppo_policy_loss(logp_new, torch.tensor([-1.0]), adv, 0.2)
    l2, _, _ = ppo_policy_loss(logp_new, torch.tensor([-2.0]), adv, 10.0)
    assert l1.item() != pytest.approx(l2.item())


def test_nonfinite_ratio_raises():
    bad = torch.tensor([float("inf")])
    with pytest.raises(ValueError):
        ppo_policy_loss(bad, torch.tensor([-1.0]), torch.tensor([1.0]), 0.2)


# --------------------------------------------------------------- entropy
def test_entropy_uniform_over_legal_actions():
    k = 20
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, torch.randperm(ACTION_SPACE_SIZE)[:k]] = True
    logits = torch.zeros(1, ACTION_SPACE_SIZE)
    h = entropy(masked_log_softmax(logits, mask), mask)
    assert h.item() == pytest.approx(math.log(k), rel=1e-5)


def test_entropy_ignores_illegal_logits():
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, torch.randperm(ACTION_SPACE_SIZE)[:15]] = True
    base = torch.zeros(1, ACTION_SPACE_SIZE)
    h1 = entropy(masked_log_softmax(base, mask), mask)
    h2 = entropy(masked_log_softmax(base + 5.0, mask), mask)
    # Same legal logits (uniform) -> same entropy regardless of illegal
    # logit values (masked out before normalization).
    assert h1.item() == pytest.approx(h2.item(), rel=1e-6)


# ----------------------------------------------------------- value losses
def test_value_losses_are_mse_with_separate_targets():
    v = torch.tensor([1.0, 2.0])
    t_train = torch.tensor([0.0, 4.0])
    t_game = torch.tensor([0.5, 0.5])
    assert value_loss(v, t_train).item() == pytest.approx(2.5)
    assert value_loss(v, t_game).item() == pytest.approx(1.25)


def test_value_loss_shape_mismatch_raises():
    with pytest.raises(ValueError):
        value_loss(torch.zeros(3), torch.zeros(4))


# ------------------------------------------------------------ total loss
def test_total_loss_excludes_sf_term_when_coef_zero():
    pl, vt, vg, h, sf = (torch.tensor(x) for x in (1.0, 2.0, 3.0, 0.5, 10.0))
    a = total_loss(pl, vt, vg, h, sf, 0.5, 0.5, 0.01, 0.0)
    b = total_loss(pl, vt, vg, h, None, 0.5, 0.5, 0.01, 0.0)
    assert a.item() == pytest.approx(b.item())
    assert a.item() == pytest.approx(1.0 + 1.0 + 1.5 - 0.005, abs=1e-6)


def test_total_loss_includes_sf_term_when_enabled():
    args = (torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
    without = total_loss(*args, None, 0.5, 0.5, 0.01, 0.0)
    with_sf = total_loss(*args, torch.tensor(4.0), 0.5, 0.5, 0.01, 0.25)
    assert without.item() == pytest.approx(1.0)
    assert with_sf.item() == pytest.approx(1.0 + 1.0)


# ------------------------------------- masked softmax error path (Rule 4)
def test_no_legal_actions_raises():
    logits = torch.randn(1, ACTION_SPACE_SIZE)
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    with pytest.raises(ValueError):
        masked_log_softmax(logits, mask)


# ------------------------------------------------------------ full update
def _synthetic_batch(n: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    states, masks, actions = [], [], []
    for i in range(n):
        board = chess.Board()
        for _ in range(i % 6):  # a few plies of real play
            moves = list(board.legal_moves)
            board.push(moves[rng.integers(len(moves))])
        states.append(encode_state(board))
        legal = ActionCodec.encode_legal_moves(board)
        m = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        m[legal] = True
        masks.append(m)
        actions.append(int(legal[rng.integers(len(legal))]))
    return {
        "states": np.stack(states).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": rng.normal(-3.0, 0.2, size=n).astype(np.float32),
        "legal_masks": np.stack(masks),
        "advantages": rng.normal(0.0, 1.0, size=n).astype(np.float32),
        "return_targets": rng.normal(0.0, 0.5, size=n).astype(np.float32),
        "game_targets": rng.choice([-1.0, 0.0, 1.0], size=n).astype(np.float32),
        "sf_targets": rng.uniform(-1.0, 1.0, size=n).astype(np.float32),
    }


def test_ppo_update_runs_with_finite_metrics_and_gradients():
    torch.manual_seed(3)
    net = ChessNet(channels=16, num_blocks=1, use_sf_head=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    metrics = ppo_update(net, _synthetic_batch(), PPOConfig(epochs=2, batch_size=4), opt)
    expected = {
        "ppo_loss", "value_train_loss", "value_game_loss", "sf_aux_loss",
        "entropy", "clip_fraction", "approx_kl", "grad_norm",
    }
    assert set(metrics) == expected
    for key, val in metrics.items():
        assert math.isfinite(val), key


def test_ppo_update_accepts_batch_without_sf_targets():
    torch.manual_seed(4)
    net = ChessNet(channels=16, num_blocks=1, use_sf_head=False)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    batch = _synthetic_batch()
    batch["sf_targets"] = None
    metrics = ppo_update(net, batch, PPOConfig(epochs=1), opt)
    assert metrics["sf_aux_loss"] == 0.0


def test_ppo_update_sf_coef_requires_targets():
    net = ChessNet(channels=16, num_blocks=1, use_sf_head=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    batch = _synthetic_batch()
    batch["sf_targets"] = None
    with pytest.raises(ValueError):
        ppo_update(net, batch, PPOConfig(stockfish_value_coef=0.1), opt)


def test_ppo_update_nan_advantage_hard_fails():
    torch.manual_seed(5)
    net = ChessNet(channels=16, num_blocks=1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    batch = _synthetic_batch()
    batch["advantages"][0] = float("nan")
    with pytest.raises(NonFiniteLossError):
        ppo_update(net, batch, PPOConfig(epochs=1), opt)


def test_ppo_update_rejects_illegal_action():
    torch.manual_seed(6)
    net = ChessNet(channels=16, num_blocks=1)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    batch = _synthetic_batch()
    # Pick an action index that is ILLEGAL in position 0: its masked
    # log-probability is -inf -> select_log_probs must hard-fail.
    illegal = int(np.flatnonzero(~batch["legal_masks"][0])[0])
    batch["actions"][0] = illegal
    with pytest.raises(ValueError):
        ppo_update(net, batch, PPOConfig(epochs=1), opt)


def test_gradient_clipping_makes_update_scale_invariant():
    # With SGD and max_grad_norm=1.0, the parameter update depends only on
    # the gradient DIRECTION (norm is clipped to 1.0).  If the ENTIRE loss
    # is scaled (policy advantages + all value targets, entropy coef = 0),
    # gradients are proportional; when BOTH norms exceed the clip threshold
    # the clipped updates must be identical.  Both scales are chosen large
    # enough that clipping engages in each run.
    base = _synthetic_batch(n=4, seed=11)
    loss_keys = ("advantages", "return_targets", "game_targets", "sf_targets")
    batch_a, batch_b = dict(base), dict(base)
    for key in loss_keys:
        batch_a[key] = base[key] * 1e2
        batch_b[key] = base[key] * 1e4
    final = []
    for batch in (batch_a, batch_b):
        torch.manual_seed(42)
        net = ChessNet(channels=16, num_blocks=1, use_sf_head=True)
        opt = torch.optim.SGD(net.parameters(), lr=0.5)
        metrics = ppo_update(
            net, batch, PPOConfig(epochs=1, max_grad_norm=1.0, entropy_coef=0.0), opt
        )
        assert metrics["grad_norm"] > 1.0  # precondition: clipping engaged
        final.append(torch.cat([p.detach().flatten() for p in net.parameters()]))
    # max |diff| ~ 4e-5 (fp rounding in the scaled loss), mean ~ 1e-8
    assert torch.allclose(final[0], final[1], atol=1e-4)

