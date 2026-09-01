"""Gate 1 / State-encoding tests."""

from __future__ import annotations

import chess
import numpy as np

from env.encoding import (
    HALFMOVE_NORM,
    NUM_CHANNELS,
    encode_state,
    encode_states,
)
from env.symmetry import canonicalize


def test_shape_and_dtype():
    b = chess.Board()
    s = encode_state(b)
    assert s.shape == (NUM_CHANNELS, 8, 8)
    assert s.dtype == np.float32


def test_starting_position_pieces():
    b = chess.Board()
    s = encode_state(b)
    # 8 white pawns -> channel 0 (OUR pawns) has 8 ones.
    assert s[0].sum() == 8.0
    # 2 white knights -> channel 1 has 2 ones.
    assert s[1].sum() == 2.0
    # 2 white bishops / rooks, 1 queen, 1 king.
    assert s[2].sum() == 2.0
    assert s[3].sum() == 2.0
    assert s[4].sum() == 1.0
    assert s[5].sum() == 1.0
    # Opponent (black) mirror.
    assert s[6].sum() == 8.0


def test_piece_plane_positions():
    b = chess.Board()
    s = encode_state(b)
    # OUR pawn plane a2 has a 1.
    assert s[0, 0, 1] == 1.0  # a-file, rank 2
    assert s[0, 4, 1] == 1.0  # e-file, rank 2
    # OUR king on e1.
    assert s[5, 4, 0] == 1.0
    # OPPONENT king on e8 -> opponent king channel index 11.
    assert s[11, 4, 7] == 1.0


def test_castling_channels():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    s = encode_state(b)
    assert s[12].mean() == 1.0  # our kingside
    assert s[13].mean() == 1.0  # our queenside
    assert s[14].mean() == 1.0  # opponent kingside
    assert s[15].mean() == 1.0  # opponent queenside

    b2 = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    canon = canonicalize(b2)
    s2 = encode_state(canon)
    # After canonicalization the canonical side (white) should own all rights.
    assert s2[12].mean() == 1.0
    assert s2[13].mean() == 1.0
    assert s2[14].mean() == 1.0
    assert s2[15].mean() == 1.0


def test_en_passant_channel():
    b = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    s = encode_state(b)
    assert s[16, 3, 5] == 1.0  # d-file, rank 6
    assert s[16].sum() == 1.0

    b2 = chess.Board("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1")
    canon = canonicalize(b2)
    s2 = encode_state(canon)
    assert s2[16, 3, 5] == 1.0  # canonical d6


def test_halfmove_channel():
    b = chess.Board()
    # Six consecutive piece moves (knights only) => halfmove clock = 6.
    for mv in ("g1f3", "b8c6", "b1c3", "g8f6", "f3g1", "c6b8"):
        b.push_uci(mv)
    assert b.halfmove_clock == 6
    s = encode_state(b)
    assert s[17].mean() == 6.0 / HALFMOVE_NORM


def test_black_canonical_pieces():
    b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    canon = canonicalize(b)
    s = encode_state(canon)
    # Canonical side is the original black: its king is on e8 -> canonical e1
    assert s[5, 4, 0] == 1.0  # OUR king on e1 (originally black king on e8)
    assert s[11, 4, 7] == 1.0  # OPPONENT king (originally white king on e1)


def test_encode_states_batch():
    boards = []
    b = chess.Board()
    boards.append(b.copy())
    b.push_uci("e2e4")
    boards.append(b.copy())
    s = encode_states(boards)
    assert s.shape == (2, NUM_CHANNELS, 8, 8)
    assert s.dtype == np.float32