"""TD errors, GAE, and value targets under alternating player perspectives.

PERSPECTIVE CONVENTION (project-wide; spec Sections 2, 18, 27, 29, 50):

* ``rewards[t]``      -- reward for the transition ``s_t -> s_{t+1}``, from
  the perspective of the player who made the decision at ``s_t`` (the mover).
* ``values[t]``       -- critic output at ``s_t`` from the perspective of the
  side to move at ``s_t`` (which IS the mover at step ``t``).
* ``bootstrap_value`` -- critic output at the truncation boundary ``s_T``
  from the perspective of the side to move there, i.e. the *opponent* of the
  last mover.

Because the next state's critic value is expressed from the *opponent's*
perspective, the Bellman operator carries an explicit negative sign
(spec Section 27 -- mandatory):

    V(s_t)     = r_t - gamma * V(s_{t+1})
    delta_t    = r_t - gamma * V(s_{t+1}) - V(s_t)          (spec Sec. 29)

and the GAE recursion alternates sign accordingly (spec Section 29):

    A_t = sum_{l=0}^{T-t-1} (-gamma * lambda)^l * delta_{t+l}

At a TRUE terminal state the next-state value is defined as exactly 0 (no
bootstrap): ``delta_{T-1} = r_{T-1} - V(s_{T-1})``.

At a TRUNCATION boundary the next state's stm-value bootstraps in with the
same negative sign (spec Section 50): truncation is NOT scored as a draw.

Two value semantics are kept strictly separate (spec Sections 15, 26):

* TRAINING value: critic of the configured PPO training reward.  Used with
  :func:`td_errors` / :func:`gae` / :func:`return_targets`.  Unbounded.
* GAME value: expected final chess result in [-1, +1] from the side-to-move
  perspective.  Its targets come from :func:`game_value_targets` -- the
  actual terminal result for completed games, or a bootstrapped alternating
  projection of the game-value critic at truncation.  NEVER trained toward
  dense shaped returns.

All functions raise ``ValueError`` on NaN/Inf inputs, length mismatches,
terminal+bootstrap contradictions, or out-of-range game values (operating
Rule 4: never silently repair invalid data).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence


def _check_finite(name: str, x: float) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value in '{name}': {x}")
    return v


def _check_sequence(name: str, seq: Sequence[float]) -> list[float]:
    return [_check_finite(f"{name}[{i}]", v) for i, v in enumerate(seq)]


def _validate_trajectory(
    rewards: Sequence[float],
    values: Sequence[float],
    terminal: bool,
    bootstrap_value: Optional[float],
) -> tuple[list[float], list[float]]:
    if len(rewards) != len(values):
        raise ValueError(
            f"rewards and values length mismatch: {len(rewards)} vs {len(values)}"
        )
    if len(rewards) == 0:
        raise ValueError("empty trajectory")
    if terminal and bootstrap_value is not None:
        raise ValueError(
            "bootstrap_value must be None at a true terminal state "
            "(spec Section 29: delta = r - V at terminal)"
        )
    if not terminal and bootstrap_value is None:
        raise ValueError(
            "truncated trajectory requires bootstrap_value "
            "(spec Section 50: bootstrap at truncation, do not invent a draw)"
        )
    r = _check_sequence("rewards", rewards)
    v = _check_sequence("values", values)
    if bootstrap_value is not None:
        _check_finite("bootstrap_value", bootstrap_value)
    return r, v
def td_errors(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    terminal: bool,
    bootstrap_value: Optional[float] = None,
) -> list[float]:
    """One-step TD errors of the TRAINING-value critic.

    ``delta_t = r_t - gamma * V_stm(s_{t+1}) - V_stm(s_t)``

    where ``V_stm(s_{t+1})`` is the *opponent-perspective* value of the next
    state -- hence the negative sign (spec Sections 27/29).  At a true
    terminal state ``V_stm(s_T) := 0``; at truncation it is
    ``bootstrap_value``.
    """
    r, v = _validate_trajectory(rewards, values, terminal, bootstrap_value)
    g = _check_finite("gamma", gamma)
    boot = 0.0 if terminal else _check_finite("bootstrap_value", bootstrap_value)

    deltas: list[float] = []
    n = len(r)
    for t in range(n):
        v_next = v[t + 1] if t < n - 1 else boot
        deltas.append(r[t] - g * v_next - v[t])
    return deltas


def gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    gae_lambda: float,
    terminal: bool,
    bootstrap_value: Optional[float] = None,
) -> list[float]:
    """Generalized advantage estimates, alternating-perspective form.

    ``A_t = sum_{l=0}^{T-t-1} (-gamma*lambda)^l * delta_{t+l}``  (Sec. 29)

    Computed backward via ``A_t = delta_t + (-gamma*lambda) * A_{t+1}``.
    The alternating sign converts the opponent-perspective TD errors of odd
    steps into the mover's perspective at ``t``.

    Property (verified in tests): with ``lambda = 1`` this equals the
    alternating Monte-Carlo return minus ``V(s_t)``:
    ``A_t = sum_l (-gamma)^l r_{t+l} - V(s_t)``.
    """
    r, v = _validate_trajectory(rewards, values, terminal, bootstrap_value)
    g = _check_finite("gamma", gamma)
    lam = _check_finite("gae_lambda", gae_lambda)
    deltas = td_errors(r, v, g, terminal, bootstrap_value)

    n = len(deltas)
    adv = [0.0] * n
    acc = 0.0
    decay = -g * lam
    for t in range(n - 1, -1, -1):
        acc = deltas[t] + decay * acc
        adv[t] = acc
    return adv


def return_targets(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    gae_lambda: float,
    terminal: bool,
    bootstrap_value: Optional[float] = None,
) -> list[float]:
    """PPO value targets for the TRAINING-value critic (spec Section 30).

    ``R_hat_t = A_t + V_train_old(s_t)``

    These targets correspond to the configured training reward (including
    any dense Stockfish shaping) -- NOT to the final chess result.  Use
    :func:`game_value_targets` for the game-value head.
    """
    v = _check_sequence("values", values)
    adv = gae(rewards, v, gamma, gae_lambda, terminal, bootstrap_value)
    return [a + val for a, val in zip(adv, v)]


def game_value_targets(
    n_steps: int,
    terminal: bool,
    gamma: float,
    z_white: Optional[float] = None,
    bootstrap_game_value: Optional[float] = None,
) -> list[float]:
    """Per-step targets for the GAME-value critic (spec Sections 24, 30, 50).

    Completed (terminal) trajectory:
        target_t = z from the perspective of the player who moved at step t.
        Ply ``t`` is played by White iff ``t`` is even (chess always starts
        with White), so ``target_t = +z_white`` for even ``t`` and
        ``-z_white`` for odd ``t``.  Zero-sum holds between consecutive
        plies by construction.

    Truncated trajectory (spec Section 50 -- bootstrap, never a draw):
        target_t = (-gamma)^(T - t) * V_game_stm(s_T)
        where ``V_game_stm(s_T)`` is ``bootstrap_game_value``, the
        game-value critic's stm-perspective output at the truncation
        boundary.  The alternating sign maps the boundary value back into
        each mover's perspective.  Bounds: |target_t| <= 1 for
        gamma in [0, 1] and |bootstrap| <= 1.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    g = _check_finite("gamma", gamma)
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"gamma out of [0, 1]: {gamma}")

    if terminal:
        if bootstrap_game_value is not None:
            raise ValueError(
                "bootstrap_game_value must be None for a terminal trajectory"
            )
        if z_white is None:
            raise ValueError("terminal trajectory requires z_white")
        z = _check_finite("z_white", z_white)
        if z not in (-1.0, 0.0, 1.0):
            raise ValueError(
                f"invalid terminal result z_white={z}; must be in {{-1,0,1}}"
            )
        # Ply t played by White iff t even (documented assumption).
        return [z if t % 2 == 0 else -z for t in range(n_steps)]

    # Truncated: bootstrap boundary.
    if z_white is not None:
        raise ValueError("z_white must be None for a truncated trajectory")
    if bootstrap_game_value is None:
        raise ValueError(
            "truncated trajectory requires bootstrap_game_value "
            "(spec Section 50)"
        )
    vb = _check_finite("bootstrap_game_value", bootstrap_game_value)
    if not -1.0 <= vb <= 1.0:
        raise ValueError(f"game value bootstrap out of [-1, 1]: {vb}")
    return [((-g) ** (n_steps - t)) * vb for t in range(n_steps)]

