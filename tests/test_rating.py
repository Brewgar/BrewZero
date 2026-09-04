"""Rating-system tests (Relative Elo pool estimator, history persistence).

The rating must (1) originate from a defined likelihood model, (2) count
draws as 0.5, (3) respect per-opponent uncertainty, (4) be None (not
fabricated) when no matchup qualifies, and (5) persist for the GUI.
"""

from __future__ import annotations

import json

import pytest

from evaluation.rating import (
    BASE_RATING,
    EvaluationResult,
    RatingHistory,
    build_evaluation_result,
    relative_elo,
)


def _row(name, games, wins, draws, losses, score, diff, lo, hi):
    return {
        "opponent": name, "games": games, "wins": wins, "draws": draws,
        "losses": losses, "score": score, "truncated": 0,
        "elo_diff": diff, "elo_ci95": (lo, hi),
    }


def test_relative_elo_single_clean_matchup():
    # One opponent, 0.75 score => diff = -400*log10(1/0.75-1) = -400*log10(1/3)
    rows = [_row("sf1", 8, 6, 0, 2, 0.75, -190.85, -350.0, -31.7)]
    rating, se, _ = relative_elo(rows)
    assert rating == pytest.approx(BASE_RATING - 190.85, abs=4.5)
    assert se is not None and se > 0


def test_relative_elo_degenerate_excluded_then_none():
    # All-wins => elo_diff None => no qualified matchup => rating None.
    rows = [_row("sf1", 8, 8, 0, 0, 1.0, None, None, None)]
    rating, se, excluded = relative_elo(rows)
    assert rating is None
    assert se is None
    assert len(excluded) == 1


def test_relative_elo_mixed_pool_weighting():
    # Two opponents with different variances; the narrow-CI one dominates.
    rows = [
        _row("material", 30, 20, 5, 5, 0.75, -190.85, -220.0, -161.7),
        _row("sf_d4",     4,  2, 1, 1, 0.625, -88.74, -300.0, 122.5),
    ]
    rating, se, _ = relative_elo(rows)
    assert BASE_RATING - 200.0 < rating < BASE_RATING - 100.0
    assert se is not None and se > 0


def test_relative_elo_too_few_games_excluded():
    rows = [_row("material", 2, 1, 1, 0, 0.75, -190.85, -500.0, 118.0)]
    rating, se, excluded = relative_elo(rows)
    assert rating is None
    assert any("2 games" in e for e in excluded)


def test_build_evaluation_result_aggregates():
    rows = [
        _row("random", 4, 0, 4, 0, 0.5, -0.0, -340.0, 340.0),
        _row("material", 4, 0, 3, 1, 0.375, -88.74, -440.0, 262.9),
    ]
    res = build_evaluation_result(rows)
    assert isinstance(res, EvaluationResult)
    assert res.games == 8
    assert res.wins == 0 and res.draws == 7 and res.losses == 1
    assert res.score == pytest.approx(7 / 16)
    assert res.rating is not None and res.rating_uncertainty is not None
    assert res.rating_label == "Relative Elo"
    assert len(res.opponents) == 2


def test_rating_history_append_and_latest(tmp_path):
    hist = RatingHistory("smoke_test", directory=tmp_path)
    res = EvaluationResult(
        wins=0, draws=12, losses=4, games=16, truncated=11, score=0.375,
        rating=1431.53, rating_uncertainty=94.56,
        opponents=[{**_row("material", 4, 0, 3, 1, 0.375, -88.7, -440.0, 262.9)}],
    )
    rec = hist.append(res, 7)
    assert rec["iteration"] == 7
    assert rec["rating"] == 1431.53
    latest = hist.latest()
    assert latest["iteration"] == 7
    assert latest["games"] == 16
    # The GUI consumes the persisted summary directly.
    assert latest["rating_label"] == "Relative Elo"
    assert latest["opponents"][0]["elo_diff"] == pytest.approx(-88.7, abs=1.0)


def test_rating_history_missing_file_is_none(tmp_path):
    assert RatingHistory("nope", directory=tmp_path).latest() is None