"""Temperature sampling (spec Section 45).

``pi_T(a|s)  propto  pi(a|s)^(1/T)``.  With raw logits this is exactly a
softmax over ``logits / T`` restricted to legal actions:

    exp(z_a / T) / sum_{a' in L(s)} exp(z_a'/T)  =  pi(a|s)^(1/T) / Z

``T = 0`` degenerates to the deterministic argmax (used in evaluation).
Invalid probabilities (NaN / negative / sum != 1) are hard errors (Rule 4).
"""

from __future__ import annotations

import math

import numpy as np


def temperature_probs(
    logits: np.ndarray, legal_indices: list[int], temperature: float
) -> np.ndarray:
    """Return the temperature-scaled legal-action distribution."""
    if not legal_indices:
        raise ValueError("no legal actions to sample from")
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        # Deterministic argmax over legal logits (ties -> lowest index).
        probs = np.zeros(len(logits), dtype=np.float64)
        best = max(legal_indices, key=lambda a: logits[a])
        probs[best] = 1.0
        return probs
    z = logits[legal_indices] / float(temperature)
    z = z - z.max()  # stable softmax
    e = np.exp(z)
    legal_probs = e / e.sum()
    probs = np.zeros(len(logits), dtype=np.float64)
    probs[legal_indices] = legal_probs
    if not np.isfinite(probs).all() or probs.min() < 0.0:
        raise ValueError("invalid probability distribution from temperature sampling")
    if not math.isclose(probs.sum(), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"probabilities sum to {probs.sum()}, expected 1")
    return probs


def sample_action_from_probs(
    legal_indices: list[int],
    probs: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Sample an action index from a pre-computed legal-action distribution."""
    return int(rng.choice(legal_indices, p=probs[legal_indices]))


def sample_action(
    logits: np.ndarray,
    legal_indices: list[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Sample an action index from the temperature-scaled legal policy."""
    probs = temperature_probs(logits, legal_indices, temperature)
    return sample_action_from_probs(legal_indices, probs, rng)


def log_prob_of(probs: np.ndarray, action: int) -> float:
    """log of the sampled probability (stored as old log-prob, Sec. 31)."""
    p = float(probs[action])
    if p <= 0.0:
        raise ValueError("sampled action has zero probability")
    return math.log(p)