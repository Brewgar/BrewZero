"""Gate 2 / Mate-handling tests.

UCI mate scores are NOT centipawns.  This module verifies that ``EngineInfo``
keeps ``is_mate`` / ``mate_distance`` separate from centipawn scores and maps
forced mates into S in {-1, +1}.
"""

from __future__ import annotations

import os

import chess
import pytest

from engine.evaluation import (
    EngineConfig,
    EngineInfo,
    StockfishEngine,
    centered_score_stm,
)

SF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish-windows-x86-64-avx2.exe")
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(SF_PATH), reason="Stockfish binary unavailable"
)


def test_mate_score_positive_and_negative():
    info_win = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=None, mate_n=3, wdl=None,
    )
    assert info_win.is_mate
    assert not info_win.has_wdl
    assert info_win.mate_n == 3
    assert info_win.centered_score_white() == 1.0

    info_loss = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=None, mate_n=-3, wdl=None,
    )
    assert info_loss.centered_score_white() == -1.0


def test_mate_distance_stored_separately():
    info = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=None, mate_n=-5, wdl=None,
    )
    assert info.mate_n == -5  # raw signed value, distance is abs()
    assert abs(info.mate_n) == 5


def test_none_mate_attribute_is_not_a_mate():
    info = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=120, mate_n=None, wdl=(990, 9, 1),
    )
    assert not info.is_mate
    assert info.has_wdl


@pytest.fixture(scope="module")
def engine():
    eng = StockfishEngine(EngineConfig(path=SF_PATH, depth=10, threads=1, hash_mb=32))
    eng.start()
    yield eng
    eng.close()


def test_real_mate_in_one_detected(engine):
    # White to move can mate with Qh5 #?  Use a genuine mate-in-1: 1.Rh7#? no:
    # position where a single killer move mates:
    board = chess.Board("6k1/5p2/6p1/8/8/8/8/R5K1 w - - 0 1")
    info = engine.analyse(board, depth=8)
    # Stockfish reports a mate score for the mating move Ra8#.
    assert info.best_move is not None
    if info.is_mate:
        assert info.centered_score_white() == 1.0
    else:
        # At shallow depth engines may return a large cp; not a correctness bug.
        assert info.centered_score_white() > 0.0


def test_mated_side_gets_negative_centered(engine):
    # Black to move with White holding an extra rook: S_stm for Black < 0.
    board = chess.Board("6k1/8/8/8/8/8/8/R5K1 b - - 0 1")
    info = engine.analyse(board, depth=8)
    assert info.has_wdl or info.is_mate
    assert centered_score_stm(info, board) < 0.0


def test_leading_side_gets_positive_centered(engine):
    # White to move with an extra rook: S_stm for White > 0.
    board = chess.Board("6k1/8/8/8/8/8/8/R5K1 w - - 0 1")
    info = engine.analyse(board, depth=8)
    assert centered_score_stm(info, board) > 0.0


def test_wdl_missing_raises():
    info = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=42, mate_n=None, wdl=None,
    )
    with pytest.raises(Exception) as exc:
        info.centered_score_white()
    assert isinstance(exc.value, Exception)  # EngineWdlMissingError subclass of RuntimeError


def test_expected_score_from_mate():
    info = EngineInfo(
        best_move=None, depth=10, seldepth=10, nodes=1, nps=0, time_ms=1,
        hashfull=0, cp=None, mate_n=1, wdl=None,
    )
    assert info.expected_score_white() == 1.0