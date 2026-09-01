"""PPO update procedure (spec Sections 31, 52, 53, 55, 56, 64).

BEHAVIOR-POLICY CONTRACT: every sample in ``batch`` was generated under one
frozen behavior policy ``theta_old`` and carries its stored
``log pi_{theta_old}(a_t | s_t)``.  Old log-probabilities are NEVER
recomputed (spec Section 52); the ratio compares the live policy against
exactly that stored policy (spec Section 53).

NaN/Inf at any point raises ``NonFiniteLossError`` (Rule 4 / Section 55):
the caller must save the failing batch + checkpoint and stop.  Values are
never silently replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from train.losses import (
    entropy,
    masked_log_softmax,
    ppo_policy_loss,
    select_log_probs,
    sf_auxiliary_loss,
    total_loss,
    value_loss,
)


class NonFiniteLossError(RuntimeError):
    """Hard stop: NaN/Inf loss or gradient during a PPO update (Sec. 55)."""


@dataclass
class PPOConfig:
    clip_eps: float = 0.2
    epochs: int = 4
    batch_size: int = 256
    entropy_coef: float = 0.01
    training_value_coef: float = 0.5
    game_value_coef: float = 0.5
    stockfish_value_coef: float = 0.0
    max_grad_norm: float | None = 1.0
    # Performance options (spec Sections 36/35).  Both default OFF; enabling
    # them does NOT change the loss mathematics: autocast only lowers the
    # trunk convs to fp16 while every loss term is explicitly cast back to
    # fp32, and all hard-fail finite checks run on UNSCALED gradients.
    amp: bool = False
    pinned_memory: bool = False

def ppo_update(net, batch: dict, config: PPOConfig, optimizer) -> dict:
    """Run ``config.epochs`` epochs of PPO on one on-policy batch.

    ``batch`` keys (all length N, one frozen behavior policy):
        states         (N, C, 8, 8) float32 canonical encodings
        actions        (N,)         int64 canonical action indices
        old_log_probs  (N,)         float32  log pi_{theta_old}(a_t|s_t)
        legal_masks    (N, A)       bool
        advantages     (N,)         float32  GAE (mover perspective)
        return_targets (N,)         float32  V_train targets (Sec. 30)
        game_targets   (N,)         float32  V_game targets in [-1, 1]
        sf_targets     (N,) or None float32  S_sf in [-1, 1] (aux head)

    Returns a dict of scalar metrics averaged over all minibatch updates.
    """
    device = next(net.parameters()).device
    n = len(batch["actions"])
    if n == 0:
        raise ValueError("empty PPO batch")

    pin = bool(config.pinned_memory) and str(device).startswith("cuda")
    states = _to_tensor(batch["states"], torch.float32, device, pin=pin)
    actions = _to_tensor(batch["actions"], torch.int64, device)
    old_log_probs = _to_tensor(batch["old_log_probs"], torch.float32, device)
    legal_masks = _to_tensor(batch["legal_masks"], torch.bool, device)
    advantages = _to_tensor(batch["advantages"], torch.float32, device)
    return_targets = _to_tensor(batch["return_targets"], torch.float32, device)
    game_targets = _to_tensor(batch["game_targets"], torch.float32, device)
    sf_targets = (
        _to_tensor(batch["sf_targets"], torch.float32, device)
        if batch.get("sf_targets") is not None
        else None
    )
    if config.stockfish_value_coef != 0.0 and sf_targets is None:
        raise ValueError("stockfish_value_coef > 0 requires sf_targets")

    metrics_sum: dict[str, float] = {}
    metrics_count = 0
    indices = torch.arange(n, device=device)
    minibatch = max(1, int(config.batch_size))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config.amp) and str(device).startswith("cuda")
    )
    return _run_epochs(
        net,
        config,
        optimizer,
        states,
        actions,
        old_log_probs,
        legal_masks,
        advantages,
        return_targets,
        game_targets,
        sf_targets,
        indices,
        minibatch,
        metrics_sum,
        metrics_count,
        scaler,
    )


def _run_epochs(
    net,
    config,
    optimizer,
    states,
    actions,
    old_log_probs,
    legal_masks,
    advantages,
    return_targets,
    game_targets,
    sf_targets,
    indices,
    minibatch,
    metrics_sum,
    metrics_count,
    scaler=None,
) -> dict:
    n = states.shape[0]
    use_amp = scaler is not None and scaler.is_enabled()
    device_type = "cuda" if str(next(net.parameters()).device).startswith("cuda") else "cpu"
    for _ in range(int(config.epochs)):
        perm = indices[torch.randperm(n, device=indices.device)]
        for start in range(0, n, minibatch):
            mb = perm[start : start + minibatch]
            with torch.autocast(device_type, enabled=use_amp):
                out = net(states[mb])
                # All loss math in fp32 regardless of autocast: the trunk
                # convs run in fp16, every head output is cast back so the
                # PPO ratio, entropy, and value losses keep fp32 semantics.
                logits = out["policy_logits"].float()
                _require_finite(logits, "policy_logits")
                v_train = out["v_train"].float()
                v_game = out["v_game"].float()
                v_sf = out["v_sf"].float() if "v_sf" in out else None

                log_probs = masked_log_softmax(logits, legal_masks[mb])
                logp_new = select_log_probs(log_probs, actions[mb], legal_masks[mb])

                p_loss, clip_frac, approx_kl = ppo_policy_loss(
                    logp_new, old_log_probs[mb], advantages[mb], config.clip_eps
                )
                vt_loss = value_loss(v_train, return_targets[mb])
                vg_loss = value_loss(v_game, game_targets[mb])
                sf_loss = (
                    sf_auxiliary_loss(v_sf, sf_targets[mb])
                    if sf_targets is not None and "v_sf" in out
                    else None
                )
                ent = entropy(log_probs, legal_masks[mb]).mean()

                loss = total_loss(
                    p_loss,
                    vt_loss,
                    vg_loss,
                    ent,
                    sf_loss,
                    config.training_value_coef,
                    config.game_value_coef,
                    config.entropy_coef,
                    config.stockfish_value_coef,
                )
            _require_finite(loss, "total PPO loss")

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                # Hard-fail finite checks must see UNSCALED gradients.
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if config.max_grad_norm is not None:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), config.max_grad_norm
                    )
                )
                if not math.isfinite(grad_norm):
                    raise NonFiniteLossError(
                        "non-finite gradient norm during PPO update"
                    )
            else:
                total_sq = sum(
                    p.grad.pow(2).sum().item()
                    for p in net.parameters()
                    if p.grad is not None
                )
                grad_norm = math.sqrt(total_sq)

            # Validate gradients BEFORE stepping (Rule 4: never repair).
            for name, p in net.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise NonFiniteLossError(f"non-finite gradient in '{name}'")

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            metrics_count += 1
            for key, val in (
                ("ppo_loss", p_loss),
                ("value_train_loss", vt_loss),
                ("value_game_loss", vg_loss),
                ("sf_aux_loss", sf_loss if sf_loss is not None else torch.tensor(0.0)),
                ("entropy", ent),
                ("clip_fraction", clip_frac),
                ("approx_kl", approx_kl),
            ):
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(val.item())
            metrics_sum["grad_norm"] = metrics_sum.get("grad_norm", 0.0) + grad_norm

    return {k: v / metrics_count for k, v in metrics_sum.items()}


def _to_tensor(x, dtype, device, pin: bool = False):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    t = torch.as_tensor(x, dtype=dtype)
    if pin and t.device.type == "cpu" and t.is_floating_point():
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device=device)


def _require_finite(t: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(t).all()):
        raise NonFiniteLossError(f"non-finite values in '{name}' during PPO update")
