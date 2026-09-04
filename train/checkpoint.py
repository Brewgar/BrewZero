"""Atomic checkpoint writes and resume (spec Sections 58-60).

* Never overwrite the last known-good checkpoint directly: write to a temp
  file, validate by re-loading, then ``os.replace``.
* Maintain one previous valid checkpoint (``*_prev.pt``) alongside
  ``*_latest.pt``.
* Payloads carry model, optimizer, iteration counters, configuration,
  engine configuration/identity, seeds, statistics, and environment
  metadata (Python/PyTorch/CUDA/GPU/CPU/Stockfish version + hash, git
  commit, architecture).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch

from engine.stockfish_pool import binary_sha256, engine_version_line


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_metadata(cfg: dict) -> dict:
    meta = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cpu": platform.processor(),
        "git_commit": _git_commit(),
        "model_architecture": {
            "channels": cfg["model"]["channels"],
            "residual_blocks": cfg["model"]["residual_blocks"],
            "use_sf_head": cfg["model"]["use_sf_head"],
        },
        "rl_parameters": {k: v for k, v in cfg["rl"].items()},
    }
    if cfg["engine"]["enabled"] and os.path.exists(cfg["engine"]["path"]):
        meta["stockfish_version"] = engine_version_line(cfg["engine"]["path"])
        meta["stockfish_binary_sha256"] = binary_sha256(cfg["engine"]["path"])
        meta["engine_configuration"] = {
            k: v for k, v in cfg["engine"].items() if k != "path"
        }
    return meta


def save_checkpoint_atomic(path: str | Path, payload: dict) -> None:
    """Write temp -> validate by re-loading -> atomic replace (Sec. 59)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    # Validate: a corrupt temp file must never replace a good checkpoint.
    probe = torch.load(tmp, map_location="cpu", weights_only=False)
    if "model" not in probe or "optimizer" not in probe:
        tmp.unlink(missing_ok=True)
        raise ValueError("checkpoint validation failed: missing model/optimizer")
    del probe
    os.replace(tmp, path)


def save_training_state(
    ckpt_dir: str | Path,
    name: str,
    payload: dict,
    archive_every: int,
    iteration: int,
    keep_previous: bool = False,
    keep_archives: bool = False,
) -> None:
    """Save the current training state as the SINGLE latest checkpoint.

    DISK POLICY (default): only ``{name}_latest.pt`` is kept.  Previous-
    generation copies and named per-iteration archives are opt-in via
    ``keep_previous`` / ``keep_archives``; when disabled, stale files from
    older runs are actively deleted so the checkpoints directory cannot
    accumulate gigabytes.

    * ``latest``      -- always written (atomically).
    * ``prev``        -- only when ``keep_previous``: the previous latest,
                         copied before the new write (corruption fallback).
    * ``{name}_iter{iteration:06d}.pt`` -- only when ``keep_archives`` and
                         ``archive_every > 0`` and ``iteration % archive_every == 0``.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / f"{name}_latest.pt"
    prev = ckpt_dir / f"{name}_prev.pt"

    if keep_previous and latest.exists():
        shutil.copy2(latest, prev)
    save_checkpoint_atomic(latest, payload)

    if keep_archives and archive_every > 0 and iteration % archive_every == 0:
        save_checkpoint_atomic(ckpt_dir / f"{name}_iter{iteration:06d}.pt", payload)

    # Enforce the disk policy: remove stale files from older runs/configs.
    if not keep_previous and prev.exists():
        prev.unlink()
    if not keep_archives:
        for stale in ckpt_dir.glob(f"{name}_iter*.pt"):
            stale.unlink()


def find_latest_checkpoint(ckpt_dir: str | Path, name: str) -> Path | None:
    ckpt_dir = Path(ckpt_dir)
    for candidate in (
        ckpt_dir / f"{name}_latest.pt",
        ckpt_dir / f"{name}_prev.pt",
    ):
        if candidate.exists():
            return candidate
    return None


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    """Load and structurally validate a checkpoint (Sec. 60/81)."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    required = {"model", "optimizer", "iteration", "config", "seeds", "meta"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"corrupt checkpoint {path}: missing {sorted(missing)}")
    return payload


def check_config_compatibility(payload: dict, cfg: dict) -> None:
    """Shape-relevant architecture and reward-semantics must match (Sec. 60).

    Architecture keys are ALWAYS strict.  The reward-schedule keys
    ``lambda_game`` / ``lambda_stockfish`` are strict unless the config sets
    ``train.allow_reward_schedule_change: true`` -- an explicit,
    operator-driven reward-schedule change on resume.  There is NO automatic
    transition: Stockfish assistance is permanent during training and is
    never disabled by a rating threshold.
    """
    old = payload.get("config", {})
    for section, keys in (
        ("model", ("channels", "residual_blocks", "use_sf_head", "norm_groups")),
        ("rl", ("gamma", "gae_lambda")),
    ):
        for key in keys:
            a, b = old.get(section, {}).get(key), cfg[section][key]
            if a is not None and a != b:
                raise ValueError(
                    f"checkpoint/config mismatch: {section}.{key} "
                    f"checkpoint={a} config={b}"
                )
    allow_schedule_change = bool(
        cfg.get("train", {}).get("allow_reward_schedule_change", False)
    )
    if not allow_schedule_change:
        for key in ("lambda_game", "lambda_stockfish"):
            a, b = old.get("rl", {}).get(key), cfg["rl"][key]
            if a is not None and a != b:
                raise ValueError(
                    f"checkpoint/config mismatch: rl.{key} "
                    f"checkpoint={a} config={b} "
                    "(set train.allow_reward_schedule_change: true to permit "
                    "a deliberate reward-schedule transition, e.g. lambda_SF -> 0)"
                )
