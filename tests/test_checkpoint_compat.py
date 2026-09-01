"""Tests for the architecture vs reward-schedule split in checkpoint compat.

The assisted -> autonomous transition (lambda_SF -> 0) requires resuming a
checkpoint with a deliberately changed reward schedule; architecture keys
must remain strictly checked.
"""

from __future__ import annotations

import pytest

from train.checkpoint import check_config_compatibility
from train.config import load_config


def _payload(lambda_game=1.0, lambda_stockfish=0.10, channels=128):
    """A checkpoint payload (contains 'config')."""
    return {
        "config": {
            "model": {"channels": channels, "residual_blocks": 6,
                      "use_sf_head": False, "norm_groups": 8},
            "rl": {"gamma": 1.0, "gae_lambda": 0.95,
                   "lambda_game": lambda_game,
                   "lambda_stockfish": lambda_stockfish},
        }
    }


def _cfg(lambda_game=1.0, lambda_stockfish=0.10, channels=128,
         **train_overrides):
    """A live config (the second argument of check_config_compatibility)."""
    cfg = load_config(None, None)
    cfg["train"].update(train_overrides)
    cfg["rl"]["lambda_game"] = lambda_game
    cfg["rl"]["lambda_stockfish"] = lambda_stockfish
    cfg["model"]["channels"] = channels
    return cfg


def test_matching_config_passes():
    check_config_compatibility(_payload(), _cfg())


def test_lambda_mismatch_rejected_by_default():
    with pytest.raises(ValueError, match="lambda_stockfish"):
        check_config_compatibility(
            _payload(lambda_stockfish=0.10), _cfg(lambda_stockfish=0.0)
        )


def test_lambda_transition_allowed_when_opted_in():
    check_config_compatibility(
        _payload(lambda_stockfish=0.10),
        _cfg(lambda_stockfish=0.0, allow_reward_schedule_change=True),
    )


def test_architecture_always_strict_even_when_opted_in():
    with pytest.raises(ValueError, match="channels"):
        check_config_compatibility(
            _payload(channels=128),
            _cfg(channels=64, allow_reward_schedule_change=True),
        )


def test_gamma_always_strict_even_when_opted_in():
    bad = _cfg(allow_reward_schedule_change=True)
    bad["rl"]["gae_lambda"] = 0.5
    with pytest.raises(ValueError, match="gae_lambda"):
        check_config_compatibility(_payload(), bad)
