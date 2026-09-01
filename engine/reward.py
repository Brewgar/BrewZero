"""Reward construction from Stockfish signals and game results.

Every component requested in operating Rule 3 is kept separately observable:

    delta_score              -- ``Delta S_t``  position change (mover POV)
    engine_regret            -- ``G_t``: prospective regret, implemented as
                                ``S_t - S_{t+1}^{mover} = -Delta S_t``.  NOTE: this is
                                NOT the counterfactual ``S_best - S_actual``; the
                                engine's best move is deliberately not searched (one
                                search per ply).  As a result ``regret_reward`` is
                                monotonically increasing in ``Delta S`` (saturating
                                tanh duplicate of the position-change signal).
    delta_reward             -- clipped scaled position change
    regret_reward            -- scaled tanh regret (negative)
    stockfish_dense_reward   -- delta_reward + regret_reward
    terminal_game_reward     -- ``z`` for the moving player (0 mid-game)
    total_training_reward    -- lambda-weighted combination
"""

from __future__ import annotations

from dataclasses import dataclass


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class RewardComponents:
    delta_score: float
    engine_regret: float
    delta_reward: float
    regret_reward: float
    stockfish_dense_reward: float
    terminal_game_reward: float
    total_training_reward: float


def dense_reward_components(
    delta_score: float,
    engine_regret: float,
    delta_coef: float,
    regret_coef: float,
    regret_tau: float,
    rmax: float,
) -> tuple[float, float]:
    """Return ``(delta_reward, regret_reward)`` per spec Sections 23.

    Parameters are raw floats for testability; the trainer passes the
    configured values.
    """
    from math import tanh

    delta_reward = clip(delta_coef * delta_score, -rmax, rmax)
    regret_reward = -regret_coef * tanh(engine_regret / regret_tau)
    return delta_reward, regret_reward


def stockfish_dense_reward(delta_reward: float, regret_reward: float) -> float:
    """Combined dense Stockfish reward ``r^SF = r^Delta + r^regret``."""
    return delta_reward + regret_reward


def terminal_game_reward(z_terminal: float) -> float:
    """Terminal game reward: exactly ``z`` (see spec Section 24)."""
    if z_terminal not in (-1.0, 0.0, 1.0):
        raise ValueError(f"invalid game reward z={z_terminal}; must be in {{-1,0,1}}")
    return z_terminal


def training_reward(
    stockfish_component: float,
    game_component: float,
    lambda_stockfish: float,
    lambda_game: float,
) -> float:
    """``r_train = lambda_sf * r^SF + lambda_game * r^game``."""
    return lambda_stockfish * stockfish_component + lambda_game * game_component


def compute_reward_components(
    delta_score: float,
    engine_regret: float,
    terminal_z_mover: float | None,
    delta_coef: float,
    regret_coef: float,
    regret_tau: float,
    rmax: float,
    lambda_stockfish: float,
    lambda_game: float,
) -> RewardComponents:
    """Full per-transition reward package (all components logged).

    ``terminal_z_mover`` is ``None`` for non-terminal transitions (game
    component = 0) and ``z`` in {-1,0,1} at the true terminal transition.
    """
    delta_reward, regret_reward = dense_reward_components(
        delta_score, engine_regret, delta_coef, regret_coef, regret_tau, rmax
    )
    sf_component = stockfish_dense_reward(delta_reward, regret_reward)
    z = terminal_game_reward(terminal_z_mover) if terminal_z_mover is not None else 0.0
    total = training_reward(sf_component, z, lambda_stockfish, lambda_game)
    return RewardComponents(
        delta_score=float(delta_score),
        engine_regret=float(engine_regret),
        delta_reward=float(delta_reward),
        regret_reward=float(regret_reward),
        stockfish_dense_reward=float(sf_component),
        terminal_game_reward=float(z),
        total_training_reward=float(total),
    )