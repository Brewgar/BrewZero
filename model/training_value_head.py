"""Training-value head: ``v_train = f(h)`` UNBOUNDED (spec Section 41).

Semantics: expected return under the exact configured PPO training reward.
When dense Stockfish shaping is active, that return is NOT bounded to
[-1, +1], so NO tanh is applied (spec Section 41 explicitly forbids it
unless the return is mathematically bounded).  Used for TD errors, GAE, and
PPO return targets -- never for the game-result critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

BOARD_SIZE = 8


class TrainingValueHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32, mlp: int = 256) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(hidden_channels * BOARD_SIZE * BOARD_SIZE, mlp)
        self.fc2 = nn.Linear(mlp, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = self.relu(self.conv(h))
        x = self.relu(self.fc1(x.flatten(1)))
        return self.fc2(x).squeeze(-1)
