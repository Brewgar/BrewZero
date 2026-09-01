"""Trajectory storage (CPU-side, compact; spec Sections 44, 52, 54).

Everything needed by PPO is kept per transition:

    state           (18, 8, 8) float32 canonical encoding
    action          canonical action index
    legal_indices   the legal action indices (compact; expanded to a dense
                    mask only at batch-build time)
    log_prob        log pi_{theta_old}(a_t | s_t)  (frozen behavior policy)
    v_game          V_game,old(s_t)   (game-value head, stm perspective)
    v_train         V_train,old(s_t)  (training-value head, stm perspective)
    v_sf_target     S_sf(s_t) engine centered score (aux-head target), or None
    reward          dict with every reward component separately observable
    terminal        true if the game actually ended on this move
    truncated       true if the max-ply boundary ended the game

Rewards are finalized one step late (see ``selfplay``): the dense Stockfish
signal of step t needs the engine score of the child position s_{t+1}, which
is produced by the NEXT step's root analysis (halving engine calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from env.encoding import NUM_CHANNELS


@dataclass
class Step:
    state: np.ndarray                      # (18, 8, 8) float32
    action: int
    legal_indices: list[int]
    log_prob: float
    v_game: float
    v_train: float
    v_sf_target: float | None
    reward: dict = field(default_factory=dict)   # finalized post-hoc
    terminal: bool = False
    truncated: bool = False


@dataclass
class Trajectory:
    steps: list[Step]
    z_white: float | None            # None for truncated games
    truncated: bool
    bootstrap_v_train: float | None  # V_train,old(s_T) stm at boundary
    bootstrap_v_game: float | None   # V_game,old(s_T)  stm at boundary
    engine_evals: int
    game_plies: int

    @property
    def mover_results(self) -> list[float]:
        """Per-step game result from the MOVER's perspective (0 mid-game)."""
        out = []
        for i, s in enumerate(self.steps):
            if s.terminal:
                z_white = self.z_white
                z_mover = z_white if i % 2 == 0 else -z_white
                out.append(float(z_mover))
            else:
                out.append(0.0)
        return out
