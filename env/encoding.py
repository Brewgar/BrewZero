"""Spatial state encoding of canonical boards.

Canonical boards only (see ``env.symmetry``): the side to move is always
White.  Encoding table (18 channels, each an 8x8 board):

====================  ========================================================
Channel index         Meaning
====================  ========================================================
0..5                  OUR (side-to-move) pieces:  P, N, B, R, Q, K
6..11                 OPPONENT pieces:           P, N, B, R, Q, K
12                    OUR kingside castling right (available: 1.0)
13                    OUR queenside castling right
14                    OPPONENT kingside castling right
15                    OPPONENT queenside castling right
16                    En-passant target square (1.0 on the square)
17                    Halfmove clock (constant fill, normalized by 100)
====================  ========================================================

Tensor layout: ``array[c, f, r]`` with file index ``f`` (0..7 = a..h) and rank
index ``r`` (0..7 = rank 1..rank 8).  A plane is therefore indexed as
``[file, rank]`` along the last two axes.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_CHANNELS = 18
BOARD_SIZE = 8
NUM_PIECE_TYPES = 6
HALFMOVE_NORM = 100.0

_PIECE_ORDER: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def encode_state(board: chess.Board) -> np.ndarray:
    """Encode one canonical board into an (NUM_CHANNELS, 8, 8) float32 array."""
    channels = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    stm = board.turn

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        f, r = chess.square_file(square), chess.square_rank(square)
        try:
            k = _PIECE_ORDER.index(piece.piece_type)
        except ValueError:
            raise ValueError(f"unhandled piece {piece.symbol()}") from None
        if piece.color == stm:
            channels[k, f, r] = 1.0
        else:
            channels[NUM_PIECE_TYPES + k, f, r] = 1.0

    castling = board.castling_rights
    # Castling-right features reflect whether the right still exists (neither
    # king nor relevant rook has moved / is pinned into it), not whether the
    # move is immediately executable (a blocking piece may sit between).
    if stm == chess.WHITE:
        channels[12] = 1.0 if bool(castling & chess.BB_H1) else 0.0
        channels[13] = 1.0 if bool(castling & chess.BB_A1) else 0.0
        channels[14] = 1.0 if bool(castling & chess.BB_H8) else 0.0
        channels[15] = 1.0 if bool(castling & chess.BB_A8) else 0.0
    else:
        # Stm is Black (defensive; canonical boards normally have White to move).
        channels[12] = 1.0 if bool(castling & chess.BB_H8) else 0.0
        channels[13] = 1.0 if bool(castling & chess.BB_A8) else 0.0
        channels[14] = 1.0 if bool(castling & chess.BB_H1) else 0.0
        channels[15] = 1.0 if bool(castling & chess.BB_A1) else 0.0

    ep = board.ep_square
    if ep is not None:
        f, r = chess.square_file(ep), chess.square_rank(ep)
        channels[16, f, r] = 1.0

    channels[17] = min(board.halfmove_clock, HALFMOVE_NORM) / HALFMOVE_NORM

    return channels


def encode_states(boards) -> np.ndarray:
    """Encode a sequence of canonical boards into (N, NUM_CHANNELS, 8, 8)."""
    if not isinstance(boards, (list, tuple)):
        boards = list(boards)
    if len(boards) == 0:
        return np.zeros((0, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    stack = np.stack([encode_state(b) for b in boards], axis=0)
    return stack.astype(np.float32)