"""Game-value head: ``v_game = tanh(f(h))`` in [-1, +1] (spec Section 40).

Semantics: expected FINAL CHESS RESULT from the side-to-move perspective.
Supervised only by actual game outcomes (``z`` targets from
``train.gae.game_value_targets``) -- never by dense shaped returns.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BOARD_SIZE = 8


class GameValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, mlp: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(hidden_channels * BOARD_SIZE * BOARD_SIZE, mlp)
        self.fc2 = nn.Linear(mlp, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = h
        x = self.relu(self.conv(x))
        x = self.relu(self.fc1(x.flatten(1)))
        return torch.tanh(self.fc2(x)).squeeze(-1)
