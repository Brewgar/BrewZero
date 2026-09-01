"""Canonical state transformation (perspective handling).

Convention
----------
The network and all derived quantities (values, rewards, GAE advantages) are
expressed from the perspective of the **side to move**.  A *canonical* board is
a board in which:

    * the side to move is represented as White;
    * the mover's pawns advance toward rank 8 ("up" the board).

Transformation
--------------
* If the side to move is White, the state is already canonical: identity.
* If the side to move is Black, we apply a **vertical mirror** (rank 1 <-> rank
  8, files unchanged) together with a **color swap** (White <-> Black), preserving
  the side-to-move player's identity.  Because files are unchanged, left/right
  capture geometry is preserved for the mover, so the transformation is an
  isometry with respect to the mover's perspective.

The transformation is an involution: applying it twice restores the original
piece placement, colors, castling rights, en-passant state, and move history.

Implementation
--------------
The transform is performed at the FEN level to guarantee that every facet
(piece colors, pawn direction, king placement, castling rights, en-passant
square, halfmove/fullmove counters) is transformed consistently.  ``chess.Board``
validates the result.
"""

from __future__ import annotations

import chess


def _vertical_mirror_and_swap_fen(fen: str) -> str:
    """Return the FEN of the vertical-mirrored, color-swapped position of ``fen``.

    The returned FEN always has White to move, because the same player remains
    the side to move but is relabeled White.
    """
    parts = fen.split(" ")
    if len(parts) != 6:
        raise ValueError(f"malformed FEN: {fen!r}")

    placement = parts[0]
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError(f"malformed FEN placement: {placement!r}")

    new_rows = []
    for row in reversed(rows):
        out = []
        for ch in row:
            if ch.isalpha():
                out.append(ch.swapcase())
            else:
                out.append(ch)
        new_rows.append("".join(out))
    new_placement = "/".join(new_rows)

    castling = "".join(
        letter for letter in ("K", "Q", "k", "q") if letter in parts[2].swapcase()
    )
    if not castling:
        castling = "-"
    ep = parts[3]
    if ep != "-":
        f = ep[0]
        r = int(ep[1])
        ep = f + str(9 - r)

    return f"{new_placement} w {castling} {ep} {parts[4]} {parts[5]}"


def canonicalize(board: chess.Board, copy: bool = True) -> chess.Board:
    """Return the canonical board for ``board`` (side to move is White).

    ``board`` is treated as read-only.  When ``copy=False`` and the board is
    already canonical, the same object is returned.
    """
    if board.turn == chess.WHITE:
        return board.copy() if copy else board
    return chess.Board(_vertical_mirror_and_swap_fen(board.fen()))


def uncanonicalize(canonical_board: chess.Board, original_turn: chess.Color) -> chess.Board:
    """Invert :func:`canonicalize`.

    ``canonical_board`` must have White to move.  ``original_turn`` is the
    color of the side to move in the original, non-canonical position.
    """
    if canonical_board.turn != chess.WHITE:
        raise ValueError("uncanonicalize requires a canonical (White-to-move) board")
    if original_turn == chess.WHITE:
        return canonical_board.copy()
    mirrored = _vertical_mirror_and_swap_fen(canonical_board.fen())
    parts = mirrored.split(" ")
    parts[1] = "b"
    return chess.Board(" ".join(parts))


def map_move_to_original(canonical_move: chess.Move, original_turn: chess.Color) -> chess.Move:
    """Map a move expressed on the canonical board to the original board.

    If the original side to move was Black, the canonical board is a vertical
    mirror, so every target square must be remapped ``(file, rank) ->
    (file, 9 - rank)``.  The promotion piece is unchanged (UCI notation is
    color-agnostic).
    """
    if original_turn == chess.WHITE:
        return canonical_move
    from_square = _vertical_mirror_square(canonical_move.from_square)
    to_square = _vertical_mirror_square(canonical_move.to_square)
    return chess.Move(from_square, to_square, promotion=canonical_move.promotion)


def _vertical_mirror_square(square: int) -> int:
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return chess.square(f, 7 - r)