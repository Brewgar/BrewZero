"""Fixed, globally indexed action space for canonical chess boards.

Action space design
-------------------
The mapping ``f : M -> {0, ..., A-1}`` maps a ``chess.Move`` to an integer.
Total size is ``A = 64 * 76 = 4864``.

Each from-square ``f`` (0..63) owns 76 consecutive slots ``[76*f, 76*f+75]``.

====================  =========================================================
Slot range            Meaning
====================  =========================================================
``0..7``              Knight jumps in 8 fixed canonical offsets.
``8..63``             Sliding moves: 8 compass directions x 7 step lengths.
                      slot = 8 + dir*7 + (step-1)
``64..75``            Promotions: 3 destination categories (left/straight/right)
                      x 4 promotion pieces (Q, R, B, N).
                      Promotion category is determined by the *file* delta of
                      the move (so the encoding is color-agnostic):
                        dfile=-1 -> category 0
                        dfile= 0 -> category 1
                        dfile=+1 -> category 2
                      slot = 64 + cat*4 + piece_idx
                      with piece_idx: Q=0, R=1, B=2, N=3.
====================  =========================================================

Decoding assumes the canonical board orientation in which the side to move is
White and its pawns advance toward rank 8.  For a canonical board:

    category 0 -> delta (-1, +1)   (forward-left)
    category 1 -> delta ( 0, +1)   (forward)
    category 2 -> delta (+1, +1)   (forward-right)

Injectivity / reversibility property
------------------------------------
* Every move with a promotion component is routed to the promotion slots.
* Every move without a promotion component is routed to knight/sliding slots.
* Therefore two distinct moves can never share a slot index, and any encoded
  legal canonical move can be decoded back to the identical ``chess.Move``.

These properties are enforced exhaustively by ``tests/test_action_space.py``.
"""

from __future__ import annotations

import chess

# Number of slots per origin square, total action count.
SLOTS_PER_SQUARE = 76
ACTION_SPACE_SIZE = 64 * SLOTS_PER_SQUARE

# Canonical knight offsets (dfile, drank).  These are absolute board
# directions, valid for either color.
KNIGHT_OFFSETS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

# Canonical compass direction offsets, index 0..7.
# Fixed, documented ordering: N, NE, E, SE, S, SW, W, NW.
SLIDE_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),    # 0 N
    (1, 1),    # 1 NE
    (1, 0),    # 2 E
    (1, -1),   # 3 SE
    (0, -1),   # 4 S
    (-1, -1),  # 5 SW
    (-1, 0),   # 6 W
    (-1, 1),   # 7 NW
)
_MAX_STEPS = 7

# Promotion-piece slot ordering within a category.
PROMOTION_PIECES: tuple[chess.PieceType, ...] = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)

_PROMOTION_PIECE_INDEX: dict[int, int] = {
    chess.QUEEN: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
    chess.KNIGHT: 3,
}


def _file_rank(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


class ActionCodec:
    """Bidirectional explicit move <-> index mapping."""

    @staticmethod
    def encode(move: chess.Move) -> int:
        """Map a ``chess.Move`` to its global index.

        Raises ``ValueError`` for moves that cannot be represented
        (non-adjacent squares that are neither a knight jump nor collinear,
        or promotions moving in an impossible direction).
        """
        if move.promotion is not None:
            return _encode_promotion(move)
        return _encode_quiet(move)

    @staticmethod
    def decode(action: int) -> chess.Move:
        """Invert :meth:`encode` in canonical (side-to-move is White) space.

        Decoding is purely geometric.  It assumes the canonical orientation in
        which promotion categories map to forward moves.
        """
        if not 0 <= action < ACTION_SPACE_SIZE:
            raise ValueError(f"action index {action} out of range [0, {ACTION_SPACE_SIZE})")
        from_square, slot = divmod(action, SLOTS_PER_SQUARE)
        if slot < 64:
            to_square = _quiet_target(from_square, slot)
            return chess.Move(from_square, to_square)
        cat = (slot - 64) // 4
        piece_idx = (slot - 64) % 4
        if cat == 0:
            dfile, drank = -1, +1
        elif cat == 1:
            dfile, drank = 0, +1
        else:
            dfile, drank = +1, +1
        to_square = _square_plus(from_square, dfile, drank)
        if to_square is None:
            raise ValueError(f"action {action} decodes to an off-board promotion")
        return chess.Move(from_square, to_square, promotion=PROMOTION_PIECES[piece_idx])

    @staticmethod
    def encode_legal_moves(board: chess.Board) -> list[int]:
        """Encode every legal move of ``board`` (must be a canonical board)."""
        return [ActionCodec.encode(m) for m in board.legal_moves]

    @staticmethod
    def legal_mask(board: chess.Board) -> list[int]:
        """Return a dense 0/1 mask of shape ``ACTION_SPACE_SIZE``.

        mask[i] == 1 iff the move with index ``i`` is legal in ``board``.
        """
        mask = [0] * ACTION_SPACE_SIZE
        for m in board.legal_moves:
            mask[ActionCodec.encode(m)] = 1
        return mask


def _encode_quiet(move: chess.Move) -> int:
    """Encode a non-promotion move through knight or sliding slots."""
    from_square = move.from_square
    to_square = move.to_square
    f0, r0 = _file_rank(from_square)
    f1, r1 = _file_rank(to_square)
    dfile, drank = f1 - f0, r1 - r0

    # Knight slot?
    k_idx = -1
    for i, (dd, dr) in enumerate(KNIGHT_OFFSETS):
        if dd == dfile and dr == drank:
            k_idx = i
            break
    if k_idx >= 0:
        return from_square * SLOTS_PER_SQUARE + k_idx

    # Sliding slot: must be collinear along one of the 8 compass directions.
    for d_idx, (dd, dr) in enumerate(SLIDE_DIRECTIONS):
        if dd == 0 and dfile != 0:
            continue
        if dr == 0 and drank != 0:
            continue
        if dd != 0 and dfile % dd != 0:
            continue
        if dr != 0 and drank % dr != 0:
            continue
        steps_file = dfile // dd if dd != 0 else None
        steps_rank = drank // dr if dr != 0 else None
        if steps_file is None:
            if steps_rank is None:
                continue
            steps = steps_rank
        elif steps_rank is None:
            steps = steps_file
        else:
            if steps_file != steps_rank:
                continue
            steps = steps_file
        if 1 <= steps <= _MAX_STEPS:
            return from_square * SLOTS_PER_SQUARE + 8 + d_idx * _MAX_STEPS + (steps - 1)
    raise ValueError(f"cannot encode quiet move {move.uci()} into action space")


def _encode_promotion(move: chess.Move) -> int:
    """Encode a promotion move (which always carries a promotion piece)."""
    from_square = move.from_square
    dfile = chess.square_file(move.to_square) - chess.square_file(from_square)
    if dfile == -1:
        cat = 0
    elif dfile == 0:
        cat = 1
    elif dfile == 1:
        cat = 2
    else:
        raise ValueError(f"promotion move {move.uci()} has invalid file delta {dfile}")
    try:
        piece_idx = _PROMOTION_PIECE_INDEX[move.promotion]
    except KeyError:
        raise ValueError(f"unknown promotion piece {move.promotion}") from None
    return from_square * SLOTS_PER_SQUARE + 64 + cat * 4 + piece_idx


def _quiet_target(from_square: int, slot: int) -> int:
    """Geometric target square for a knight/sliding slot (no board lookup)."""
    if slot < 8:
        dfile, drank = KNIGHT_OFFSETS[slot]
        to_square = _square_plus(from_square, dfile, drank)
        if to_square is None:
            raise ValueError(f"knight slot {slot} from {from_square} is off-board")
        return to_square
    within = slot - 8
    d_idx, step = divmod(within, _MAX_STEPS)
    step += 1
    dfile, drank = SLIDE_DIRECTIONS[d_idx]
    to_square = _square_plus(from_square, dfile * step, drank * step)
    if to_square is None:
        raise ValueError(f"sliding slot {slot} from {from_square} is off-board")
    return to_square


def _square_plus(from_square: int, dfile: int, drank: int) -> int | None:
    f0 = chess.square_file(from_square)
    r0 = chess.square_rank(from_square)
    f1 = f0 + dfile
    r1 = r0 + drank
    if not (0 <= f1 < 8 and 0 <= r1 < 8):
        return None
    return chess.square(f1, r1)