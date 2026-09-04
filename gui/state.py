"""Shared mutable state observed by the GUI thread from the worker thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class State(Enum):
    IDLE = "Idle"
    TRAINING = "Training"
    STOPPING = "Stopping"
    EVALUATING = "Evaluating"
    PLAYING = "Playing"
    LOADING = "Loading"
    ERROR = "Error"


@dataclass
class TrainingStatus:
    state: State = State.IDLE
    experiment: str = "—"
    checkpoint: str = "—"
    iteration: int = 0
    games: int = 0
    plies: int = 0
    elapsed_seconds: float = 0.0
    mean_reward: float = 0.0
    mean_dense_reward: float = 0.0
    mean_delta_s: float = 0.0
    mean_engine_regret: float = 0.0
    entropy: float = 0.0
    value_loss: float = 0.0
    ppo_loss: float = 0.0
    eval_score: float = 0.0
    elo: float | None = None
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_games: int = 0
    total_plies: int = 0
    total_engine_evals: int = 0
    model_params: str = "—"
    device: str = "—"
    gpu: str = "—"
    cuda_available: bool = False
    stockfish_path: str = "—"
    stockfish_enabled: bool = False
    train_hours: float | None = None
    max_iterations: int | None = None
    log_messages: list[str] = field(default_factory=list)

    def format_elapsed(self) -> str:
        s = int(self.elapsed_seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def progress_label(self) -> str:
        if self.max_iterations is not None and self.max_iterations > 0:
            return f"{self.iteration} / {self.max_iterations}"
        return "∞"

    def win_draw_loss(self) -> str:
        return f"{self.wins}W / {self.draws}D / {self.losses}L"


@dataclass
class GUIConfig:
    config_path: str = "configs/smoke.yaml"
    checkpoint_path: str | None = None
    eval_games: int = 4
    play_as_white: bool = True
    play_temperature: float = 0.0
    train_hours: float | None = None
    max_iterations: int | None = None
    window_width: int = 860
    window_height: int = 620