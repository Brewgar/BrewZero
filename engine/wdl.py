"""WDL -> probability -> expected-score math.

All functions assert the document-level invariants:

    p_w + p_d + p_l == 1
    0 <= E <= 1
    -1 <= S <= 1
"""

from __future__ import annotations


def wdl_to_probs(wins: float, draws: float, losses: float) -> tuple[float, float, float]:
    """Normalize WDL raw counts into probabilities.

    Raises ``ValueError`` when raw values are negative, NaN, inf, or sum to
    zero (invalid engine response per operating Rule 4).
    """
    import math

    for name, v in (("wins", wins), ("draws", draws), ("losses", losses)):
        if not math.isfinite(v):
            raise ValueError(f"non-finite WDL count: {name}={v}")
        if v < 0:
            raise ValueError(f"negative WDL count: {name}={v}")
    total = wins + draws + losses
    if total <= 0:
        raise ValueError(f"WDL counts sum to zero: {wins},{draws},{losses}")
    return wins / total, draws / total, losses / total


def expected_from_probs(p_w: float, p_d: float, p_l: float) -> float:
    """Expected game score ``E = p_w + p_d / 2`` in [0, 1]."""
    e = p_w + 0.5 * p_d
    if not -1e-9 <= e <= 1 + 1e-9:
        raise ValueError(f"expected score out of range: {e}")
    return float(e)


def centered_from_probs(p_w: float, p_d: float, p_l: float) -> float:
    """Centered score ``S = 2E - 1`` in [-1, 1]."""
    return 2.0 * expected_from_probs(p_w, p_d, p_l) - 1.0


def centered_from_wdl(wins: float, draws: float, losses: float) -> float:
    """One-shot centered score from raw WDL counts."""
    return centered_from_probs(*wdl_to_probs(wins, draws, losses))


def expected_from_wdl(wins: float, draws: float, losses: float) -> float:
    """One-shot expected score from raw WDL counts."""
    return expected_from_probs(*wdl_to_probs(wins, draws, losses))