"""Relative Elo rating for the fixed evaluation opponent pool.

METHODOLOGY (documented per project spec Sections 4-6, 14):

The displayed rating is a **Relative Elo** on the project's internal scale.
It is NOT an official FIDE/online rating and must not be read as an absolute
playing-strength claim beyond the fixed opponent pool.

Model (batch estimator over each fixed opponent matchup):

* For opponent ``i`` with match score ``s_i = (wins + 0.5*draws) / games``,
  the Elo expected-score model gives

      E[score] = 1 / (1 + 10^((R_opponent - R_model) / 400))

  so the maximum-likelihood rating DIFFERENCE to that opponent is

      d_i = -400 * log10(1 / s_i - 1)

  (see :func:`evaluation.elo.elo_difference`; draws count as S = 0.5 and are
  included; color balance is enforced upstream by ``play_match``, which
  alternates colors and aggregates results from the NET's perspective).

* Each matchup's standard error is propagated from the binomial score SE
  through d(diff)/d(score) (delta method), matching the 95% CI returned by
  :func:`evaluation.elo.elo_difference`.

* The pool aggregate is the inverse-variance weighted mean of per-opponent
  differences:

      R_model = BASE + sum_i w_i d_i / sum_i w_i,     w_i = 1 / se_i^2
      se_agg  = sqrt(1 / sum_i w_i)

  Opponents with degenerate scores (exactly 0.0 or 1.0 -- the MLE is at
  infinity) or too few games are EXCLUDED, never fabricated.  If no opponent
  qualifies, the rating is ``None`` and the GUI legitimately shows N/A.

Reference system: the pool is defined by the evaluation suite (random,
material-greedy, fixed-depth Stockfish opponents).  ``BASE = 1500`` anchors
the scale to that pool; only rating DIFFERENCES across checkpoints are
scientifically meaningful.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE_RATING = 1500.0
Z95 = 1.96
MIN_GAMES = 4          # a matchup needs this many games to enter the aggregate
RATING_LABEL = "Relative Elo"

_REPO = Path(__file__).resolve().parent.parent


@dataclass
class EvaluationResult:
    """Structured evaluation outcome, including the pool rating."""

    wins: int
    draws: int
    losses: int
    games: int
    truncated: int
    score: float
    rating: float | None
    rating_uncertainty: float | None
    rating_label: str = RATING_LABEL
    opponents: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "rating": self.rating,
            "rating_uncertainty": self.rating_uncertainty,
            "rating_label": self.rating_label,
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "truncated": self.truncated,
            "score": self.score,
        }


def _se_from_ci(lo: float | None, hi: float | None) -> float | None:
    """Recover the delta-method SE from the 95% CI returned by elo_difference."""
    if lo is None or hi is None:
        return None
    se = (hi - lo) / (2.0 * Z95)
    if not math.isfinite(se) or se <= 0.0:
        return None
    return se


def relative_elo(opponent_rows: list[dict]) -> tuple[float | None, float | None, list[str]]:
    """Pool rating from per-opponent rows (inverse-variance weighted).

    Rows are the dicts returned by ``play_match`` (need ``games``,
    ``elo_diff``, ``elo_ci95``).  Returns ``(rating, se, excluded_notes)``;
    ``rating is None`` when no matchup qualifies.
    """
    weights: list[float] = []
    diffs: list[float] = []
    excluded: list[str] = []
    for row in opponent_rows:
        name = row.get("opponent", "?")
        games = int(row.get("games", 0))
        diff = row.get("elo_diff")
        lo, hi = row.get("elo_ci95", (None, None))
        if diff is None:
            excluded.append(
                f"{name}: degenerate score (all wins or all losses) - excluded"
            )
            continue
        if games < MIN_GAMES:
            excluded.append(f"{name}: only {games} games (< {MIN_GAMES}) - excluded")
            continue
        se = _se_from_ci(lo, hi)
        if se is None:
            excluded.append(f"{name}: invalid CI - excluded")
            continue
        weights.append(1.0 / (se * se))
        diffs.append(float(diff))
    if not weights:
        return None, None, excluded
    total_w = sum(weights)
    rating = BASE_RATING + sum(w * d for w, d in zip(weights, diffs)) / total_w
    se_agg = math.sqrt(1.0 / total_w)
    return rating, se_agg, excluded


def build_evaluation_result(results: list[dict]) -> EvaluationResult:
    """Aggregate per-opponent match rows into one :class:`EvaluationResult`."""
    wins = draws = losses = truncated = 0
    for row in results:
        wins += int(row.get("wins", 0))
        draws += int(row.get("draws", 0))
        losses += int(row.get("losses", 0))
        truncated += int(row.get("truncated", 0))
    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games if games else 0.0
    rating, se, _excluded = relative_elo(results)
    return EvaluationResult(
        wins=wins, draws=draws, losses=losses, games=games, truncated=truncated,
        score=score, rating=rating, rating_uncertainty=se,
        opponents=list(results),
    )


class RatingHistory:
    """JSONL rating history per experiment (reports/ratings_<name>.jsonl).

    The GUI reads the latest entry at startup so a restarted application
    still shows the most recent evaluation (spec Sections 12/13).
    """

    def __init__(self, experiment: str, directory: str | Path | None = None) -> None:
        self.experiment = experiment
        d = Path(directory) if directory is not None else _REPO / "reports"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"ratings_{experiment}.jsonl"

    def append(self, result: EvaluationResult, iteration: int) -> dict:
        record = {
            "iteration": int(iteration),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **result.summary(),
            "opponents": [
                {
                    "opponent": r.get("opponent"),
                    "games": r.get("games"),
                    "elo_diff": r.get("elo_diff"),
                    "elo_ci95": list(r["elo_ci95"]) if r.get("elo_ci95") else None,
                }
                for r in result.opponents
            ],
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record

    def latest(self) -> dict | None:
        if not self.path.exists():
            return None
        last = None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return last


def load_latest_rating(experiment: str) -> dict | None:
    """Convenience accessor for the GUI: latest rating record or None."""
    return RatingHistory(experiment).latest()
