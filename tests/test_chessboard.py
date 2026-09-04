"""Graphical chess board widget tests (torch-free).

The widget is a pure view: it must select legal moves for the side to move,
show legal targets, emit exactly the clicked move (auto-queen on
promotions), and map coordinates correctly under board flipping.
"""

from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

tk = pytest.importorskip("tkinter", reason="tkinter required for board widget")

from gui.chessboard import ChessBoardWidget  # noqa: E402


@pytest.fixture(scope="module")
def root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


def _click(widget: ChessBoardWidget, square: int) -> None:
    x, y = widget._square_center(square)
    widget._on_click(SimpleNamespace(x=int(x), y=int(y)))


def test_select_and_move_emits_correct_move(root):
    received = []
    w = ChessBoardWidget(root, on_move=received.append)
    w.set_position(chess.Board())
    _click(w, chess.E2)
    assert w.selected == chess.E2
    assert chess.E3 in w._targets and chess.E4 in w._targets
    _click(w, chess.E4)
    assert received == [chess.Move(chess.E2, chess.E4)]
    assert w.selected is None


def test_cannot_select_opponent_piece(root):
    received = []
    w = ChessBoardWidget(root, on_move=received.append)
    w.set_position(chess.Board())
    _click(w, chess.E7)  # black pawn, not side to move
    assert w.selected is None
    assert received == []


def test_illegal_destination_rejected(root):
    received = []
    w = ChessBoardWidget(root, on_move=received.append)
    w.set_position(chess.Board())
    _click(w, chess.E2)
    _click(w, chess.E5)  # three squares forward: illegal
    assert received == []
    # selection cleared after an attempted move
    assert w.selected is None


def test_auto_queen_on_promotion(root):
    received = []
    w = ChessBoardWidget(root, on_move=received.append)
    # Black king safely on h8; the white a-pawn can promote on a8.
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    w.set_position(board)
    _click(w, chess.A7)
    root.update()  # pump event loop so the widget finishes its click
    _click(w, chess.A8)
    assert received == [chess.Move(chess.A7, chess.A8, promotion=chess.QUEEN)]


def test_capture_targets_show_ring_and_move(root):
    received = []
    w = ChessBoardWidget(root, on_move=received.append)
    # White rook a1 can capture the black pawn a7 (a8 blocked by the king,
    # so the position is legal: the black king is not in check).
    w.set_position(chess.Board("k7/p7/8/8/8/8/8/R6K w - - 0 1"))
    _click(w, chess.A1)
    assert chess.A7 in w._targets      # enemy piece -> ring marker
    assert chess.A4 in w._targets      # empty square -> dot marker
    _click(w, chess.A7)
    assert received == [chess.Move(chess.A1, chess.A7)]


def test_flipped_orientation_maps_squares(root):
    w = ChessBoardWidget(root, white_bottom=False)
    # With Black at the bottom, square a1 sits in the TOP-left data cell.
    x, y = w._square_origin(chess.A1)
    assert w._xy_to_square(x + 5, y + 5) == chess.A1
    # and h8 in the bottom-right cell
    x, y = w._square_origin(chess.H8)
    assert w._xy_to_square(x + 5, y + 5) == chess.H8


def test_set_position_clears_selection(root):
    w = ChessBoardWidget(root)
    w.set_position(chess.Board())
    _click(w, chess.E2)
    assert w.selected is not None
    w.set_position(chess.Board())
    assert w.selected is None
    assert w._targets == set()