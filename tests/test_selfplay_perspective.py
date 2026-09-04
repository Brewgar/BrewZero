"""Perspective regression tests for the self-play / Stockfish interface.

These tests exist because the engine tests exercised ``centered_score_stm``
correctly (passing the ANALYZED board) while the self-play code passed the
CANONICAL board (whose turn is always WHITE).  The result was a silent sign
inversion of the engine score at every Black-to-move ply, corrupting Delta S,
regret, and the SF aux targets.  A scripted multi-ply self-play game with a
stub pool is the only test shape that catches this class of bug.

Every expected number below is computed BY HAND from the project convention:

    S_stm(P)          centered engine score from the side-to-move perspective
    S_mover(child)    = -S_stm(child)   (zero-sum: mover = -opponent)
    Delta S_t         = S_mover(child) - S_stm(pre-move)
"""

from __future__ import annotations

import numpy as np
import pytest

import selfplay.selfplay as sp
from engine.evaluation import EngineInfo
from env.action_space import ActionCodec
from model.network import ChessNet


def _info_from_white_score(s_white: float) -> EngineInfo:
    """EngineInfo whose centered White-perspective score equals ``s_white``.

    Uses a WDL with no draws: S = 2E - 1 with E = wins/1000.
    """
    wins = max(0, min(1000, int(round((s_white + 1.0) * 500))))
    return EngineInfo(
        best_move=None, depth=8, seldepth=8, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=None, mate_n=None, wdl=(wins, 0, 1000 - wins),
    )


class StubPool:
    """Pool stub: maps 'placement turn' -> White-perspective centered score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.requested: list[str] = []

    @staticmethod
    def _key(fen: str) -> str:
        parts = fen.split(" ")
        return f"{parts[0]} {parts[1]}"

    def analyse(self, fen: str, depth=None) -> EngineInfo:
        key = self._key(fen)
        self.requested.append(key)
        return _info_from_white_score(self.scores[key])


RL = {
    "delta_coef": 1.0,       # delta_reward == Delta S (unclipped: r_max=10)
    "regret_coef": 0.0,
    "regret_tau": 1.0,
    "r_max": 10.0,
    "lambda_stockfish": 1.0,
    "lambda_game": 0.0,
}

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"


def _play_scripted(canonical_moves_uci: list[str], scores: dict[str, float],
                   max_plies: int):
    net = ChessNet(channels=8, num_blocks=1, use_sf_head=False)
    queue = list(canonical_moves_uci)
    original = sp.sample_action_from_probs

    def fake_sample_action(legal_indices, probs, rng):
        uci = queue.pop(0)
        for a in legal_indices:
            if ActionCodec.decode(a).uci() == uci:
                return a
        raise AssertionError(f"scripted move {uci} not legal")

    sp.sample_action_from_probs = fake_sample_action
    try:
        pool = StubPool(scores)
        traj = sp.play_single_game(
            net, RL, pool, engine_depth=8, max_plies=max_plies,
            temperature=1.0, device="cpu",
            rng=np.random.default_rng(0), use_sf_head=False,
        )
    finally:
        sp.sample_action_from_probs = original
    return traj, pool


def test_white_mover_perspective_multi_ply():
    """White improves from equal to clearly better: Delta S must be POSITIVE.

    Positions (White-perspective scores chosen by the stub):
      s0 start (White to move)                     S_white = 0.0
      s1 after 1.e4 (Black to move)                S_white = +0.6  -> S_stm = -0.6
      s2 after 1...e5 (White to move, truncation)  S_white = +0.2
    Expected:
      step0: S_mover(s1) = -S_stm(s1) = +0.6; Delta S = +0.6 - 0.0 = +0.6
      step1 (truncation boundary): S_mover(s2) = -S_stm(s2) = -0.2;
             Delta S = -0.2 - (-0.6) = +0.4
    (The pre-fix code returned -0.6 and -0.4 respectively.)
    """
    scores = {
        START: 0.0,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": 0.6,
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w": 0.2,
    }
    # Canonical UCIs: White e2e4, and Black's e7e5 is canonical e2e4 (mirror).
    traj, _ = _play_scripted(["e2e4", "e2e4"], scores, max_plies=2)
    assert traj.truncated
    assert traj.steps[0].reward["delta_score"] == pytest.approx(0.6)
    assert traj.steps[1].reward["delta_score"] == pytest.approx(0.4)
    # Step 0 was rewarded for IMPROVING White's position.
    assert traj.steps[0].reward["delta_reward"] > 0


def test_black_mover_perspective_multi_ply():
    """The same sign discipline must hold when Black is the mover.

      s0 start (White to move)                       S_white = 0.0
      s1 after 1.e4 (Black to move)                  S_white = +0.3 -> S_stm = -0.3
      s2 after 1...d5 (White to move, truncation)    S_white = +0.1
    Expected:
      step0 (White moves): S_mover(s1) = -(-0.3) = +0.3; Delta S = +0.3
      step1 (Black moves): S_stm(s1) = -0.3; S_mover(s2) = -0.1;
             Delta S = -0.1 - (-0.3) = +0.2
    """
    scores = {
        START: 0.0,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b": 0.3,
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w": 0.1,
    }
    # Canonical UCIs: White e2e4, Black d7d5 == canonical d2d4 (mirror).
    traj, _ = _play_scripted(["e2e4", "d2d4"], scores, max_plies=2)
    assert traj.truncated
    assert traj.steps[0].reward["delta_score"] == pytest.approx(0.3)
    assert traj.steps[1].reward["delta_score"] == pytest.approx(0.2)


def test_terminal_mate_perspective_and_no_terminal_search():
    """Fool's mate: the terminal child's mover-relative score is exactly z.

      s0 start (White)                    S_white = 0.0
      s1 after 1.f3 (Black)               S_white = +0.1 -> S_stm = -0.1
      s2 after 1...e5 (White)             S_white = +0.2
      s3 after 2.g4 (Black)               S_white = +0.4 -> S_stm = -0.4
      s4 terminal after Qh4#  (NOT searched)
    Expected deltas: [+0.1, -0.1, +0.2, 1.0 - (-0.4) = +1.4];
    the mover at the terminal ply is Black -> z_mover = +1, z_white = -1.
    """
    scores = {
        START: 0.0,
        "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b": 0.1,
        "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w": 0.2,
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b": 0.4,
    }
    moves = ["f2f3", "e2e4", "g2g4", "d1h5"]  # canonical UCIs
    traj, pool = _play_scripted(moves, scores, max_plies=10)
    assert not traj.truncated
    assert traj.z_white == pytest.approx(-1.0)
    deltas = [s.reward["delta_score"] for s in traj.steps]
    assert deltas == pytest.approx([0.1, -0.1, 0.2, 1.4])
    assert traj.steps[3].reward["terminal_game_reward"] == pytest.approx(1.0)
    # The terminal position itself must never be searched (z is used instead).
    assert len(pool.requested) == 4


def test_training_temperature_contract():
    """Rollout temperature != 1 must hard-fail (PPO ratio identity)."""
    net = ChessNet(channels=8, num_blocks=1, use_sf_head=False)
    with pytest.raises(ValueError, match="temperature"):
        sp.play_single_game(
            net, RL, None, engine_depth=0, max_plies=4,
            temperature=0.75, device="cpu",
            rng=np.random.default_rng(0), use_sf_head=False,
        )

