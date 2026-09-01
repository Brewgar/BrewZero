"""PPO batch construction from self-play trajectories (spec Sections 30, 64).

Responsibilities, in order:
  1. training-value GAE advantages + return targets (from the configured
     training reward, which includes any dense Stockfish shaping);
  2. game-value targets (actual terminal results / truncation bootstrap) --
     a SEPARATE target for the SEPARATE game-value head;
  3. optional Stockfish-value auxiliary targets S_sf(s_t);
  4. dense batch arrays (states, actions, old log-probs, legal masks).

All GAE inputs use the frozen behavior policy's stored values
(V_train,old / V_game,old) -- never recomputed (spec Sections 31/52).
"""

from __future__ import annotations

import numpy as np

from env.action_space import ACTION_SPACE_SIZE
from train.gae import gae, game_value_targets, return_targets
from selfplay.trajectory import Trajectory


def build_ppo_batch(
    trajectories: list[Trajectory],
    rl: dict,
    use_sf_head: bool,
) -> tuple[dict, dict]:
    """Build the on-policy PPO batch and aggregate logging statistics."""
    states, actions, old_log_probs, masks = [], [], [], []
    advantages, ret_targets, game_targets, sf_targets = [], [], [], []
    stats = {
        "games": len(trajectories),
        "plies": 0,
        "engine_evals": 0,
        "truncated_games": 0,
        "terminated_games": 0,
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "mean_game_reward": 0.0,
        "mean_stockfish_reward": 0.0,
        "mean_delta_s": 0.0,
        "mean_regret": 0.0,
        "mean_return": 0.0,
        "n_steps": 0,
    }
    n_reward_steps = 0

    for traj in trajectories:
        rewards = [s.reward["total_training_reward"] for s in traj.steps]
        values = [s.v_train for s in traj.steps]
        terminal = not traj.truncated
        bootstrap = None if terminal else traj.bootstrap_v_train
        adv = gae(rewards, values, rl["gamma"], rl["gae_lambda"], terminal, bootstrap)
        targets = return_targets(
            rewards, values, rl["gamma"], rl["gae_lambda"], terminal, bootstrap
        )
        game_t = game_value_targets(
            n_steps=len(traj.steps),
            terminal=terminal,
            gamma=rl["gamma"],
            z_white=traj.z_white if terminal else None,
            bootstrap_game_value=None if terminal else traj.bootstrap_v_game,
        )

        for i, step in enumerate(traj.steps):
            states.append(step.state)
            actions.append(step.action)
            old_log_probs.append(step.log_prob)
            mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            mask[step.legal_indices] = True
            masks.append(mask)
            advantages.append(adv[i])
            ret_targets.append(targets[i])
            game_targets.append(game_t[i])
            if use_sf_head:
                if step.v_sf_target is None:
                    raise ValueError("missing Stockfish aux target for a step")
                sf_targets.append(step.v_sf_target)

        stats["plies"] += traj.game_plies
        stats["engine_evals"] += traj.engine_evals
        stats["truncated_games"] += int(traj.truncated)
        stats["terminated_games"] += int(not traj.truncated)
        if traj.z_white is not None:
            stats["white_wins"] += int(traj.z_white == 1.0)
            stats["black_wins"] += int(traj.z_white == -1.0)
            stats["draws"] += int(traj.z_white == 0.0)
        stats["mean_return"] += float(traj.z_white or 0.0)
        for s in traj.steps:
            r = s.reward
            stats["mean_game_reward"] += r.get("terminal_game_reward", 0.0)
            stats["mean_stockfish_reward"] += r.get("stockfish_dense_reward", 0.0)
            stats["mean_delta_s"] += r.get("delta_score", 0.0)
            stats["mean_regret"] += r.get("engine_regret", 0.0)
            n_reward_steps += 1

    stats["n_steps"] = len(actions)
    n = max(1, stats["n_steps"])
    n_rew = max(1, n_reward_steps)
    stats["mean_game_reward"] /= n_rew
    stats["mean_stockfish_reward"] /= n_rew
    stats["mean_delta_s"] /= n_rew
    stats["mean_regret"] /= n_rew
    stats["mean_return"] /= max(1, stats["terminated_games"])
    stats["mean_abs_advantage"] = float(np.abs(advantages).mean())
    stats["advantage_std"] = float(np.std(advantages))

    batch = {
        "states": np.stack(states).astype(np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": np.asarray(old_log_probs, dtype=np.float32),
        "legal_masks": np.stack(masks),
        "advantages": np.asarray(advantages, dtype=np.float32),
        "return_targets": np.asarray(ret_targets, dtype=np.float32),
        "game_targets": np.asarray(game_targets, dtype=np.float32),
        "sf_targets": np.asarray(sf_targets, dtype=np.float32) if use_sf_head else None,
    }
    return batch, stats