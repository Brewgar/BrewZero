"""Gate 1 / Terminal-state and environment tests, incl. edge-case chess states."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from env.action_space import ACTION_SPACE_SIZE, ActionCodec
from env.chess_env import (
    ChessEnv,
    is_terminal,
    result_for_player,
    result_for_mover,
)

START = chess.STARTING_FEN


def _board(*moves: str) -> chess.Board:
    b = chess.Board()
    for m in moves:
        b.push_san(m)
    return b


def test_starting_position_not_terminal():
    assert not is_terminal(chess.Board())


def test_checkmate_white_wins():
    b = chess.Board()
    for move in ("f3", "e5", "g4", "Qh4"):
        b.push_san(move)  # fool's mate
    assert b.is_checkmate()
    assert is_terminal(b)
    assert result_for_player(b, chess.BLACK) == 1
    assert result_for_player(b, chess.WHITE) == -1


def test_stalemate():
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert b.is_stalemate()
    assert is_terminal(b)
    assert result_for_player(b, chess.WHITE) == 0
    assert result_for_player(b, chess.BLACK) == 0


def test_insufficient_material():
    b = chess.Board("8/8/8/8/8/8/4k3/4K3 w - - 0 1")
    assert b.is_insufficient_material()
    assert is_terminal(b)
    assert result_for_player(b, chess.WHITE) == 0


def test_fifty_move_claim_is_terminal():
    # 50 moves of shuffling: hard to construct; check the rule directly.
    b = chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
    b.halfmove_clock = 100
    assert b.can_claim_fifty_moves()
    assert is_terminal(b)
    assert result_for_player(b, chess.WHITE) == 0


def test_seventyfive_move_is_auto_terminal():
    b = chess.Board("8/8/8/4k3/8/4K3/8/8 w - - 0 1")
    b.halfmove_clock = 150
    assert b.is_seventyfive_moves()
    assert is_terminal(b)


def test_threefold_claim_is_terminal():
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    b = _board(*moves)
    assert b.can_claim_threefold_repetition()
    assert is_terminal(b)


def test_fivefold_is_terminal():
    moves = ["Nf3", "Nf6", "Ng1", "Ng8"] * 5
    b = _board(*moves)
    assert b.is_fivefold_repetition()
    assert is_terminal(b)


# ---------------------------------------------------------------- env behavior


def test_env_reset_and_step():
    env = ChessEnv()
    mask = env.legal_mask()
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask.dtype == np.int8
    assert int(mask.sum()) == 20
    idx = ActionCodec.encode(chess.Move.from_uci("e2e4"))
    assert mask[idx] == 1
    env.step(idx)
    assert env.board.fen().split(" ")[0] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
    assert env.turn == chess.BLACK


def test_env_step_illegal_raises():
    env = ChessEnv()
    with pytest.raises(ValueError):
        env.step(ActionCodec.encode(chess.Move.from_uci("e2e5")))  # illegal pawn double-jump
        env.step(0)  # knight jump from a1 is also illegal


def test_env_black_step_mapping():
    env = ChessEnv()
    env.step(ActionCodec.encode(chess.Move.from_uci("e2e4")))
    # Black to move: canonical coordinate for black's e7-e5 is 'e2e4'.
    idx = ActionCodec.encode(chess.Move.from_uci("e2e4"))
    env.step(idx)
    assert env.board.fen().split(" ")[0] == "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"
    assert env.board.fullmove_number == 2
    assert env.turn == chess.WHITE


def test_env_castling_both_sides():
    env = ChessEnv("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    env.step(ActionCodec.encode(chess.Move.from_uci("e1g1")))
    assert env.board.piece_at(chess.G1).piece_type == chess.KING

    env2 = ChessEnv("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1")
    # Black to move: canonical coordinate for black's e8-g8 is e1-g1.
    env2.step(ActionCodec.encode(chess.Move.from_uci("e1g1")))
    assert env2.board.piece_at(chess.G8).piece_type == chess.KING


def test_env_en_passant():
    env = ChessEnv()
    env.step(ActionCodec.encode(chess.Move.from_uci("e2e4")))  # white
    env.step(ActionCodec.encode(chess.Move.from_uci("a2a3")))  # canonical black a7-a6
    env.step(ActionCodec.encode(chess.Move.from_uci("e4e5")))  # white
    env.step(ActionCodec.encode(chess.Move.from_uci("d2d4")))  # canonical black d7-d5
    # White captures en passant e5xd6 (canonical == original; white to move).
    env.step(ActionCodec.encode(chess.Move.from_uci("e5d6")))
    hist = list(env.board.move_stack)
    assert hist[-1].uci() == "e5d6"
    assert env.board.piece_at(chess.D6) is not None
    assert env.board.piece_at(chess.D6).color == chess.WHITE
    assert env.board.piece_at(chess.D5) is None
    assert env.board.piece_at(chess.E5) is None


def test_env_promotion_and_underpromotion():
    env = ChessEnv("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    env.step(ActionCodec.encode(chess.Move.from_uci("e7e8q")))
    assert env.board.piece_at(chess.E8).piece_type == chess.QUEEN

    env2 = ChessEnv("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    env2.step(ActionCodec.encode(chess.Move.from_uci("e7e8n")))
    assert env2.board.piece_at(chess.E8).piece_type == chess.KNIGHT


def test_env_terminal_result_for_mover():
    b = chess.Board()
    b.push_san("f3")
    b.push_san("e5")
    b.push_san("g4")
    prev = b.copy()
    b.push_san("Qh4")
    assert is_terminal(b)
    assert result_for_mover(prev, b) == 1  # black just mated


def test_terminal_result_via_move():
    mate = chess.Board("7k/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert not is_terminal(mate)
    mate.push_san("Ra8")
    assert is_terminal(mate)
    assert result_for_player(mate, chess.WHITE) == 1
    assert result_for_player(mate, chess.BLACK) == -1