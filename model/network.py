"""Shared residual trunk for the chess policy/value network.

Architecture (spec Section 38):

* input  : (N, 18, 8, 8) canonical state tensors (see ``env.encoding``)
* stem   : 3x3 conv, GroupNorm, ReLU   (spatial structure preserved)
* trunk  : ``num_blocks`` residual blocks ``h_{l+1} = ReLU(h_l + F_l(h_l))``
  with ``F_l = [3x3 conv, GN, ReLU, 3x3 conv, GN]``
* heads  : policy / game-value / training-value / optional SF-value
  (separate modules in their own files)

NORMALIZATION CHOICE -- GroupNorm, NOT BatchNorm (documented decision):

PPO stores log-probabilities from a frozen behavior policy (spec Sections
31, 52) and requires ``theta == theta_old  =>  ratio == 1`` (Section 80).
BatchNorm behaves differently in train mode (batch statistics) and eval
mode (running statistics), so re-evaluating the same parameters would give
different outputs depending on the mode/batch -- silently violating the
on-policy ratio identity.  GroupNorm depends only on the sample itself and
is identical in train/eval mode, keeping the PPO semantics exact.

ACTIVATION: ReLU throughout the trunk and hidden head layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from env.action_space import ACTION_SPACE_SIZE
from env.encoding import NUM_CHANNELS

BOARD_SIZE = 8


class ConvBlock(nn.Sequential):
    """3x3 conv + GroupNorm + ReLU (padding=1, spatial size preserved)."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    """``h_{l+1} = ReLU(h_l + F_l(h_l))`` with a two-conv ``F_l``."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.relu(x + self.fc(x))


class ChessNet(nn.Module):
    """Shared residual trunk + separate value-semantics heads.

    Heads are separate nn.Modules with disjoint parameters:

    * ``policy_logits``  -- (N, A) raw logits, A == ACTION_SPACE_SIZE (4864)
    * ``v_game``         -- (N,) in [-1, 1]: expected final chess result,
      side-to-move perspective (tanh-bounded)
    * ``v_train``        -- (N,) UNBOUNDED: expected PPO training return
      (no tanh -- dense Stockfish shaping makes returns unbounded)
    * ``v_sf``           -- (N,) in [-1, 1], only when ``use_sf_head``:
      auxiliary Stockfish centered-score predictor
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        channels: int = 128,
        num_blocks: int = 6,
        action_size: int = ACTION_SPACE_SIZE,
        use_sf_head: bool = False,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if channels % norm_groups != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by norm_groups ({norm_groups})"
            )
        self.action_size = action_size
        self.use_sf_head = use_sf_head

        self.stem = ConvBlock(in_channels, channels, norm_groups)
        self.trunk = nn.Sequential(
            *(ResidualBlock(channels, norm_groups) for _ in range(num_blocks))
        )

        # Imported here to keep a single source of truth for head modules
        # and to avoid circular imports at package init time.
        from model.policy_head import PolicyHead
        from model.game_value_head import GameValueHead
        from model.training_value_head import TrainingValueHead
        from model.sf_value_head import SFValueHead

        self.policy = PolicyHead(channels, action_size=action_size)
        self.game_value = GameValueHead(channels)
        self.training_value = TrainingValueHead(channels)
        self.sf_value = SFValueHead(channels) if use_sf_head else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(self.stem(x))
        out: dict[str, torch.Tensor] = {
            "policy_logits": self.policy(h),
            "v_game": self.game_value(h),
            "v_train": self.training_value(h),
        }
        if self.sf_value is not None:
            out["v_sf"] = self.sf_value(h)
        return out
