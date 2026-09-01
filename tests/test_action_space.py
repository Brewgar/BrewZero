"""Gate 1 / Property tests for the explicit action-space mapping."""

from __future__ import annotations

import chess
import pytest

from env.action_space import (
    ACTION_SPACE_SIZE,
    PROMOTION_PIECES,
    SLOTS_PER_SQUARE,
    ActionCodec,
)

FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # midgame
    "8/4P3/8/8/8/8/8/4K2k w - - 0 1",  # promotion (white)
    "8/8/8/3k4/8/8/4p3/4K3 b - - 0 1",  # promotion (black)
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",  # kingside castling right (white)
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",  # queenside castling right (white)
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # castling (black to move)
    "rnbqkbnr/ppp1pppp/8/8/8/8/PPPPpPPP/RNBQKBNR w - - 0 1",  # ep target e3? no: see below
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # en passant (white)
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",  # en passant (black)
    "r1bqkbnr/pppp1ppp/2n5/8/4p1P1/2N2N2/PPPPPP1P/R1BQKB1R b KQkq g3 0 3",  # ep + check-ish
    "8/8/8/8/8/2k5/8/4K3 b - - 0 1",  # forced (near-king endgame)
]


def _all_legal_moves_encode_decode(board: chess.Board) -> None:
    """Every legal move must encode to a unique index and decode back exactly."""
    seen: set[int] = set()
    for move in board.legal_moves:
        a = ActionCodec.encode(move)
        assert 0 <= a < ACTION_SPACE_SIZE
        assert a not in seen, f"collision for {move.uci()} -> {a}"
        seen.add(a)
        assert ActionCodec.decode(a) == move, (
            f"round-trip failed: {move.uci()} -> {a} -> {ActionCodec.decode(a)}"
        )
    # The mask must mark exactly the legal indices.
    mask = ActionCodec.legal_mask(board)
    assert len(mask) == ACTION_SPACE_SIZE
    assert sum(mask) == len(seen)
    assert set(i for i, v in enumerate(mask) if v) == seen


@pytest.mark.parametrize("fen", FENS)
def test_encode_decode_all_legal_moves(fen: str) -> None:
    board = chess.Board(fen)
    _all_legal_moves_encode_decode(board)


def test_starting_position_mask() -> None:
    board = chess.Board()
    mask = ActionCodec.legal_mask(board)
    assert sum(mask) == 20
    # 8 white pawn double/push moves + 4 knight moves = 20.
    for move in board.legal_moves:
        assert mask[ActionCodec.encode(move)] == 1


def test_promotion_distinguishes_pieces() -> None:
    board = chess.Board("8/4P3/8/8/8/8/8/4K2k w - - 0 1")
    moves = list(board.legal_moves)
    promoted = [m for m in moves if m.promotion is not None]
    assert len(promoted) == 4
    assert {m.promotion for m in promoted} == {
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    }
    indices = {ActionCodec.encode(m): m for m in promoted}
    assert len(indices) == 4
    # Each piece type must land in the same category (straight) with distinct slot.
    straight_slots = {action % SLOTS_PER_SQUARE for action in indices}
    # 4 promotion slots for the straight category: 68..71
    assert straight_slots == {68, 69, 70, 71}


def test_promotion_underpromotion_black_perspective() -> None:
    board = chess.Board("8/3k4/8/8/8/8/4p3/6K1 b - - 0 1")
    moves = list(board.legal_moves)
    promoted = [m for m in moves if m.promotion is not None]
    assert len(promoted) == 4
    indices = {ActionCodec.encode(m) for m in promoted}
    assert len(indices) == 4
    # Black promotion straight (dfile=0) also maps to the straight category.
    straight_slots = {action % SLOTS_PER_SQUARE for action in indices}
    assert straight_slots == {68, 69, 70, 71}


def test_geometric_closure_of_valid_actions() -> None:
    """Every decodable action must re-encode to itself.

    Some actions describe geometrically off-board targets (e.g. a knight jump
    from a1 leaving the board); decoding those raises ``ValueError`` by design,
    and they can never appear as legal moves.
    """
    decodable = 0
    for action in range(ACTION_SPACE_SIZE):
        try:
            move = ActionCodec.decode(action)
        except ValueError:
            continue
        decodable += 1
        assert ActionCodec.encode(move) == action, (
            f"closure failed for action {action} -> {move.uci()} -> "
            f"{ActionCodec.encode(move)}"
        )
    # The vast majority of slots are decodable.  Only boundary geometry
    # (knight jumps and slid/se/promotion targets leaving the board) raises.
    assert 2000 <= decodable <= 4800


def test_decode_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        ActionCodec.decode(ACTION_SPACE_SIZE)
    with pytest.raises(ValueError):
        ActionCodec.decode(-1)


def test_action_space_size() -> None:
    assert ACTION_SPACE_SIZE == 64 * SLOTS_PER_SQUARE == 4864
    assert SLOTS_PER_SQUARE == 76


def test_castling_encodes_as_king_double_step() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    kingside = chess.Move.from_uci("e1g1")
    assert kingside in board.legal_moves
    a = ActionCodec.encode(kingside)
    assert snake_slide(a, kingside)  # helper below
    assert ActionCodec.decode(a) == kingside


def snake_slide(action: int, move: chess.Move) -> bool:
    """The castle move e1g1 is a 2-step East slide (E, step 2)."""
    from_square, slot = divmod(action, SLOTS_PER_SQUARE)
    assert from_square == move.from_square
    within = slot - 8
    d_idx, step = divmod(within, 7)
    # Direction 2 is E, step 2 for e1->g1.
    return d_idx == 2 and (step + 1) == 2