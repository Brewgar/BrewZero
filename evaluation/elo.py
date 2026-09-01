"""Rating-difference estimation (spec Section 68).

A raw win percentage is NOT an Elo rating.  We report the Elo difference to
the fixed opponent implied by the match score, with a normal-approximation
95% CI propagated from the score's binomial standard error.
"""

from __future__ import annotations

import math


def elo_difference(wins: int, draws: int, losses: int) -> tuple[float | None, float | None, float | None]:
    """Elo difference from the match score plus a 95% CI.

    Returns ``(elo_diff, ci_low, ci_high)``; ``None`` values when the score
    is degenerate (0.0 or 1.0) or the sample is too small.
    """
    n = wins + draws + losses
    if n == 0:
        return None, None, None
    score = (wins + 0.5 * draws) / n
    if score <= 0.0 or score >= 1.0:
        return None, None, None
    diff = -400.0 * math.log10(1.0 / score - 1.0)
    # Binomial SE of the score, propagated through d(diff)/d(score).
    se_score = math.sqrt(score * (1.0 - score) / n)
    d_elo = 400.0 / (math.log(10.0) * score * (1.0 - score))
    se_diff = se_score * d_elo
    return diff, diff - 1.96 * se_diff, diff + 1.96 * se_diff
