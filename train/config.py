"""Configuration loading with documented defaults (spec Sections 4, 87).

YAML files under ``configs/`` only override values that differ from the
defaults below; everything else inherits.  All important values are
accessible through configuration.  Unknown keys are rejected (Rule 1: no
silent redefinition).

Engine path default resolves to the repository-local Stockfish binary; it is
never hard-coded inside the modules themselves.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENGINE_PATH = str(_REPO_ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe")

DEFAULTS: dict = {
    "project": {
        "seed": 12345,
        "name": "run",
    },
    "engine": {
        "enabled": True,
        "path": _DEFAULT_ENGINE_PATH,
        "depth": 16,
        "threads": 1,
        "hash_mb": 256,
        "workers": 12,
    },
    "model": {
        "channels": 128,
        "residual_blocks": 6,
        "use_sf_head": False,
        "norm_groups": 8,
    },
    # Reward semantics (spec Sections 23-25, 72):
    #   A terminal_only      -> lambda_stockfish: 0.0, engine.enabled: false
    #   B stockfish_dense    -> lambda_game: 0.0
    #   C combined           -> defaults below
    #   D combined_aux       -> stockfish_value_coef > 0, use_sf_head: true
    "rl": {
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "ppo_clip": 0.2,
        "lambda_game": 1.0,
        "lambda_stockfish": 0.10,
        "delta_coef": 0.10,
        "regret_coef": 0.10,
        "regret_tau": 0.5,   # documented choice: tanh saturation at |G|~1
        "r_max": 1.0,
        "entropy_coef": 0.01,
        "training_value_coef": 0.5,
        "game_value_coef": 0.5,
        "stockfish_value_coef": 0.0,
    },
    "ppo": {
        "epochs": 4,
        "batch_size": 512,
    },
    "optimizer": {
        "type": "adamw",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
    },
    "selfplay": {
        "games_per_iteration": 48,
        "max_plies": 300,
        "temperature": 1.0,
        "threads": 12,         # concurrent games (bounded by engine workers)
    },
    # Performance wiring (spec Section 53).  Defaults preserve the exact
    # pre-optimization numerical behavior except inference batching, which
    # only changes GEMM batch size (last-bit float effects).
    "performance": {
        "inference_batch_size": 8,   # >1 enables the microbatch inference server
        "max_inference_wait_ms": 2.0,
        "pinned_memory": False,      # pinned host->device batches for PPO
        "torch_compile": False,      # checkpoint-compatible compile (off: CPU-bound)
        "amp": False,                # autocast+GradScaler for the PPO update
    },
    "train": {
        "hours": None,
        "max_iterations": None,
        "eval_every": 10,
        "eval_games": 20,
        "checkpoint_every": 1,
        # Evaluation-only ply budget; None -> fall back to selfplay.max_plies.
        # Kept separate so training truncation semantics and match-scoring
        # truncation are independently configurable (audit H-4).
        "eval_max_plies": None,
        # When True, resume() may load a checkpoint whose lambda_game /
        # lambda_stockfish differ from this config (deliberate operator-driven
        # reward-schedule change).  Architecture keys remain strictly checked.
        "allow_reward_schedule_change": False,
        # DISK POLICY: keep only ONE checkpoint file ({name}_latest.pt).
        # Named per-iteration archive copies are OFF by default (they caused
        # ~134 MB x N accumulation); enable only for short diagnostic runs.
        "archive_checkpoints": False,
        # Keep one previous-generation checkpoint ({name}_prev.pt) as a
        # corruption fallback.  The atomic write (tmp -> validate -> replace)
        # makes this nearly redundant, so it is OFF by default.
        "keep_previous_checkpoint": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _check_unknown(merged: dict, reference: dict, prefix: str = "") -> None:
    for key in merged:
        if key not in reference:
            raise ValueError(f"unknown config key: '{prefix}{key}'")
        if isinstance(merged[key], dict) and isinstance(reference[key], dict):
            _check_unknown(merged[key], reference[key], prefix=f"{prefix}{key}.")


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config over :data:`DEFAULTS`.

    ``overrides`` (e.g. from the command line) is applied last.  Unknown keys
    raise ``ValueError``.  PROJECT POLICY: Stockfish assistance is permanent
    -- any configuration that enables the engine must keep the Stockfish
    reward coefficient strictly positive, and a positive coefficient requires
    the engine (spec Sections 15-18: no autonomous phase, no hidden disable).
    """
    merged = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        _check_unknown(user, DEFAULTS)
        merged = _deep_merge(merged, user)
    if overrides:
        _check_unknown(overrides, DEFAULTS)
        merged = _deep_merge(merged, overrides)
    _validate_assistance_policy(merged)
    return merged


def _validate_assistance_policy(cfg: dict) -> None:
    """Stockfish assistance is permanent (spec Sections 15-18).

    * engine enabled  =>  lambda_stockfish > 0 (no engine without assistance);
    * lambda_stockfish > 0  =>  engine enabled (no silent assistance loss).
    There is no rating threshold and no automatic phase anywhere in the
    project; Stockfish assistance is permanent during training.  A change of
    lambda_* on resume is only permitted when the operator explicitly opts in
    via train.allow_reward_schedule_change (deliberate schedule change, never
    triggered by a rating threshold).
    """
    engine_on = bool(cfg["engine"]["enabled"])
    lam_sf = float(cfg["rl"]["lambda_stockfish"])
    if engine_on and lam_sf <= 0.0:
        raise ValueError(
            "invalid configuration: engine.enabled=true requires "
            f"rl.lambda_stockfish > 0 (permanent Stockfish assistance); got {lam_sf}"
        )
    if lam_sf > 0.0 and not engine_on:
        raise ValueError(
            "invalid configuration: rl.lambda_stockfish > 0 requires "
            f"engine.enabled=true; got lambda_stockfish={lam_sf}, "
            "engine.enabled=false"
        )


def config_overrides(hours: float | None = None, max_iterations: int | None = None) -> dict:
    """Command-line overrides that map onto the config tree."""
    out: dict = {}
    if hours is not None:
        out["train"] = {"hours": float(hours)}
    if max_iterations is not None:
        out.setdefault("train", {})["max_iterations"] = int(max_iterations)
    return out
