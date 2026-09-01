"""Policy head: exactly ``A == ACTION_SPACE_SIZE`` raw logits (spec Sec. 39).

The output size is bound to ``env.action_space.ACTION_SPACE_SIZE`` (4864) at
construction time -- the same constant used by the action encoder, the legal
mask, trajectory storage, and the decoder.  A mismatch between any of them
is therefore impossible by construction.

Layout: 1x1 conv (channels -> ``hidden``) -> flatten -> linear
(``hidden * 8 * 8`` -> A).  Raw logits, no normalization, no activation:
illegal-action handling is done exclusively through the legal-action mask
(spec Section 14), never by clamping logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from env.action_space import ACTION_SPACE_SIZE

BOARD_SIZE = 8


class PolicyHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        action_size: int = ACTION_SPACE_SIZE,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_channels * BOARD_SIZE * BOARD_SIZE, action_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.relu(self.conv(h))
        return self.fc(x.flatten(1))
