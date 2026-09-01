"""PPO loss mathematics (spec Sections 14, 31-37, 80).

PERSPECTIVE: advantages and values are mover-perspective per the project
convention; this module makes no sign changes -- it consumes GAE outputs
from ``train.gae`` as-is.

Masking rule (spec Section 14): illegal logits are set to the smallest
finite float of the dtype (NOT ``-inf``, which would turn ``0 * -inf``
into NaN inside the entropy sum) before a numerically stable log-softmax.
This guarantees exactly:

    pi(a|s) = 0                for illegal a
    sum_{a in L(s)} pi(a|s) = 1

A position with zero legal actions is an invalid input (Rule 4) and raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PPOLossComponents:
    """All loss parts, kept separately observable (operating Rule 3)."""

    policy_loss: torch.Tensor
    value_train_loss: torch.Tensor
    value_game_loss: torch.Tensor
    sf_aux_loss: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor
    clip_fraction: torch.Tensor
    approx_kl: torch.Tensor


def masked_log_softmax(
    logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Log-probabilities with illegal actions forced to exactly zero prob.

    ``logits``: (N, A) raw policy logits.  ``legal_mask``: (N, A) boolean.
    Raises ``ValueError`` for any row without a legal action.
    """
    if logits.shape != legal_mask.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != mask shape {tuple(legal_mask.shape)}"
        )
    mask = legal_mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError(
            "invalid state: a position in the batch has no legal actions"
        )
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    # Stable log-softmax: subtract the per-row max before exp.
    row_max = masked.max(dim=-1, keepdim=True).values
    lse = torch.logsumexp(masked - row_max, dim=-1, keepdim=True) + row_max
    return masked - lse


def select_log_probs(
    log_probs: torch.Tensor, actions: torch.Tensor, legal_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Gather ``log pi(a_t | s_t)`` for taken actions; validates legality.

    ``legal_mask`` (optional, (N, A) bool) is the authoritative legality
    check: masked-out actions carry a finite but astronomically negative
    log-probability, so finiteness alone cannot detect them (Rule 4:
    invalid inputs must hard-fail, never flow through).
    """
    idx = actions.long().unsqueeze(1)
    if legal_mask is not None:
        legal = legal_mask.bool().gather(1, idx).squeeze(1)
        if not bool(legal.all()):
            raise ValueError("taken action is illegal in its state")
    chosen = log_probs.gather(1, idx).squeeze(1)
    if not bool(torch.isfinite(chosen).all()):
        raise ValueError("non-finite log-probability for a taken action")
    return chosen


def ppo_policy_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped surrogate objective (spec Section 33), computed in log space.

    Returns ``(loss, clip_fraction, approx_kl)`` where

        loss           = -mean( min(rho*A, clip(rho, 1-e, 1+e)*A) )
        clip_fraction  = mean(|rho - 1| > eps)
        approx_kl      = mean(log pi_old - log pi_new)   (biased estimator)
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("non-finite PPO ratio (check stored old log-probs)")
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.mean(torch.min(surr1, surr2))
    clip_fraction = (ratio - 1.0).abs().gt(clip_eps).float().mean()
    approx_kl = torch.mean(log_probs_old - log_probs_new)
    return loss, clip_fraction, approx_kl


def entropy(log_probs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Per-state entropy over LEGAL actions only (spec Section 34).

    ``H = -sum_{a in L(s)} pi(a|s) log pi(a|s)``.  Illegal entries carry
    probability exactly 0 and contribute exactly 0.
    """
    mask = legal_mask.bool()
    probs = log_probs.exp().masked_fill(~mask, 0.0)
    return -(probs * log_probs.masked_fill(~mask, 0.0)).sum(dim=-1)


def value_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE value loss (used for both V_train and V_game with their own
    targets -- never one target for both heads, spec Section 35)."""
    if values.shape != targets.shape:
        raise ValueError(
            f"value/target shape mismatch: {tuple(values.shape)} vs {tuple(targets.shape)}"
        )
    return F.mse_loss(values, targets)


def sf_auxiliary_loss(v_sf: torch.Tensor, s_sf: torch.Tensor) -> torch.Tensor:
    """MSE loss of the optional Stockfish-value head (spec Section 36)."""
    return value_loss(v_sf, s_sf)


def total_loss(
    policy_loss: torch.Tensor,
    value_train_loss: torch.Tensor,
    value_game_loss: torch.Tensor,
    entropy_mean: torch.Tensor,
    sf_aux_loss: torch.Tensor | None,
    training_value_coef: float,
    game_value_coef: float,
    entropy_coef: float,
    stockfish_value_coef: float,
) -> torch.Tensor:
    """``L = -L^PPO + c_Vtrain L_Vtrain + c_Vgame L_Vgame - c_H H + c_SF L_sf``

    (spec Section 37).  ``stockfish_value_coef == 0`` (or ``sf_aux_loss
    is None``) excludes the auxiliary term entirely.
    """
    loss = (
        policy_loss
        + training_value_coef * value_train_loss
        + game_value_coef * value_game_loss
        - entropy_coef * entropy_mean
    )
    if sf_aux_loss is not None and stockfish_value_coef != 0.0:
        loss = loss + stockfish_value_coef * sf_aux_loss
    return loss
