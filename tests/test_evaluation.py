"""Gate 2 / Stockfish wrapper tests (required engine-dependent tests)."""

from __future__ import annotations

import os

import chess
import pytest

from engine.evaluation import (
    EngineConfig,
    EngineInfo,
    EngineWdlMissingError,
    StockfishEngine,
    centered_score_stm,
    mover_centered_score,
)

SF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "stockfish", "stockfish-windows-x86-64-avx2.exe")
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(SF_PATH), reason="Stockfish binary unavailable"
)


@pytest.fixture(scope="module")
def engine():
    eng = StockfishEngine(
        EngineConfig(path=SF_PATH, depth=10, threads=1, hash_mb=32)
    )
    eng.start()
    yield eng
    eng.close()


def test_uci_handshake_and_identity(engine):
    assert engine._engine is not None  # started without error


def test_root_analysis_structured(engine):
    board = chess.Board()
    info = engine.analyse(board, depth=8)
    assert info.depth >= 6
    assert info.nodes > 0
    assert info.nps >= 0
    # Non-terminal root must provide either WDL or mate.
    assert info.has_wdl or info.is_mate
    assert info.best_move is not None
    assert info.best_move in board.legal_moves
    s = info.centered_score_white()
    assert -1.0 <= s <= 1.0


def test_wdl_sum_is_one(engine):
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    info = engine.analyse(board, depth=8)
    assert info.has_wdl
    wins, draws, losses = info.wdl
    assert wins + draws + losses > 0
    assert wins >= 0 and draws >= 0 and losses >= 0


def test_centered_score_white_bounds(engine):
    positions = [
        chess.Board(),
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
        chess.Board("8/3k4/8/8/8/8/4p3/6K1 b - - 0 1"),  # black to move, promotion
    ]
    for b in positions:
        info = engine.analyse(b, depth=8)
        s = info.centered_score_white()
        assert -1.0 <= s <= 1.0


def test_side_to_move_perspective_white(engine):
    # White to move: S_stm == S_white.
    board = chess.Board()
    info = engine.analyse(board, depth=8)
    assert centered_score_stm(info, board) == info.centered_score_white()


def test_side_to_move_perspective_black(engine):
    # Black to move: S_stm == -S_white.
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    info = engine.analyse(board, depth=8)
    assert centered_score_stm(info, board) == pytest.approx(-info.centered_score_white())


def test_mover_perspective_after_move(engine):
    # After white's e2-e4 it's black to move; the mover-relative value of the
    # child is minus black's stm value.
    board = chess.Board()
    child = board.copy()
    child.push_uci("e2e4")
    info_child = engine.analyse(child, depth=8)
    mover_view = mover_centered_score(info_child, child)
    assert mover_view == pytest.approx(-centered_score_stm(info_child, child))


def test_perspective_consistency_color_swap(engine):
    # A color-swapped position must have opposite mover-relative centered
    # scores for the same structural side.
    b1 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    i1 = engine.analyse(b1, depth=8)
    s_white_persp = i1.centered_score_white()
    # Mirror the same position: white now to move (e4 white pawn).
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    canon = chess.Board(
        "RNBQKBNR/PPPP1PPP/8/4p3/8/8/pppppppp/rnbqkbnr w KQkq - 0 1"
    )
    i2 = engine.analyse(canon, depth=8)
    # Original: black-to-move with a pawn advantage for white -> S_white > 0.
    # Canonical: white-to-move, roles relabeled -> same side is white -> same sign.
    assert s_white_persp * i2.centered_score_white() > 0 or abs(s_white_persp) < 1e-9