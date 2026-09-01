"""Stockfish-value head: ``v_sf = tanh(f(h))`` in [-1, +1] (spec Sec. 42).

Semantics: Stockfish's centered expected score ``S_sf`` from the
side-to-move perspective.  Purely auxiliary; independent of the game-value,
training-value, and policy code paths (separate parameters, separate loss,
separate log entry).  Optional -- constructed only when ``use_sf_head``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BOARD_SIZE = 8


class SFValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, mlp: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(hidden_channels * BOARD_SIZE * BOARD_SIZE, mlp)
        self.fc2 = nn.Linear(mlp, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.relu(self.conv(h))
        x = self.relu(self.fc1(x.flatten(1)))
        return torch.tanh(self.fc2(x)).squeeze(-1)
