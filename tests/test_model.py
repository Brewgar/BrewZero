"""Gate 5 -- model architecture tests.

Verifies (spec Sections 38-42): exact policy output size, value-head
ranges (tanh-bounded game/SF values, unbounded training value), disjoint
head parameters, finite outputs, and gradient flow.  A tiny configuration
is used so the suite stays fast; dimensions are identical to the serious
configuration.
"""

from __future__ import annotations

import pytest
import torch

from env.action_space import ACTION_SPACE_SIZE
from env.encoding import NUM_CHANNELS
from model.network import ChessNet


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ChessNet(channels=32, num_blocks=2, use_sf_head=True)


@pytest.fixture(scope="module")
def outputs(net):
    torch.manual_seed(1)
    x = torch.randn(4, NUM_CHANNELS, 8, 8)
    with torch.no_grad():
        return net(x)


# ------------------------------------------------------------ dimensions
def test_policy_head_output_is_exact_action_space(outputs):
    assert outputs["policy_logits"].shape == (4, ACTION_SPACE_SIZE)


def test_action_space_size_is_not_4672():
    # The implemented mapping is 64 * 76 = 4864 (spec Section 13 forbids
    # assuming 4672 without implementing the exact mapping).
    assert ACTION_SPACE_SIZE == 4864


def test_value_heads_are_scalar_per_state(outputs):
    for key in ("v_game", "v_train", "v_sf"):
        assert outputs[key].shape == (4,)


def test_input_channels_match_encoder(net):
    assert net.stem[0].in_channels == NUM_CHANNELS == 18


# ---------------------------------------------------------- value ranges
def test_game_value_head_bounded(outputs):
    assert (outputs["v_game"].abs() <= 1.0).all()


def test_sf_value_head_bounded(outputs):
    assert (outputs["v_sf"].abs() <= 1.0).all()


def test_training_value_head_is_unbounded(net):
    # No tanh on the training-value head (Sec. 41).  GroupNorm normalizes
    # input magnitude away, so the functional proof is to scale the FINAL
    # linear layer: without a saturating nonlinearity the output scales
    # proportionally and must exceed 1.
    with torch.no_grad():
        net.training_value.fc2.weight.mul_(100.0)
        net.training_value.fc2.bias.mul_(100.0)
        x = torch.randn(4, NUM_CHANNELS, 8, 8)
        v = net(x)["v_train"]
    assert (v.abs() > 1.0).any()
    assert torch.isfinite(v).all()


def test_bounded_heads_stay_bounded_on_extreme_inputs(net):
    x = torch.randn(4, NUM_CHANNELS, 8, 8) * 50.0
    with torch.no_grad():
        out = net(x)
    assert (out["v_game"].abs() <= 1.0).all()
    assert (out["v_sf"].abs() <= 1.0).all()


# --------------------------------------------------------- independence
def test_heads_have_disjoint_parameters(net):
    groups = {
        "policy": set(net.policy.parameters()),
        "game": set(net.game_value.parameters()),
        "train": set(net.training_value.parameters()),
        "sf": set(net.sf_value.parameters()),
    }
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (groups[a] & groups[b]), f"{a} and {b} share parameters"


def test_sf_head_absent_when_disabled():
    net = ChessNet(channels=32, num_blocks=1, use_sf_head=False)
    assert net.sf_value is None
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    with torch.no_grad():
        out = net(x)
    assert "v_sf" not in out


# ------------------------------------------------------------ numerics
def test_all_outputs_finite(outputs):
    for key, value in outputs.items():
        assert torch.isfinite(value).all(), key


def test_gradients_flow_to_all_heads(net):
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    out = net(x)
    loss = (
        out["policy_logits"].pow(2).mean()
        + out["v_game"].pow(2).mean()
        + out["v_train"].pow(2).mean()
        + out["v_sf"].pow(2).mean()
    )
    loss.backward()
    for name, param in net.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


def test_deterministic_given_same_input(net):
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    net.eval()
    with torch.no_grad():
        a = net(x)
        b = net(x)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_train_eval_modes_agree_groupnorm(net):
    # GroupNorm has no running statistics: identical outputs in train and
    # eval mode for the same input (needed for the PPO ratio identity).
    x = torch.randn(2, NUM_CHANNELS, 8, 8)
    net.eval()
    with torch.no_grad():
        ev = net(x)
    net.train()
    with torch.no_grad():
        tr = net(x)
    for key in ev:
        assert torch.allclose(ev[key], tr[key], atol=1e-6), key
