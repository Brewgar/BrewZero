"""Graphical chess board widget (pure Tkinter + python-chess).

A readable, click-to-move chess board rendered on a ``tk.Canvas``:

* lichess-style square colors (open-source AGPL project palette):
  light ``#F0D9B5`` / dark ``#B58863``;
* pieces drawn with the Unicode SOLID chess glyphs (U+265A..U+265F) for
  BOTH colors, colored white/black with a drop shadow -- far more readable
  than the outline "white" glyphs on light squares;
* rank/file coordinate labels; last-move and check highlighting;
* click-to-select with legal-destination dots (rings on capturable squares);
* click a destination to emit the move through the ``on_move`` callback.
  The widget never mutates game state: the owner validates/applies the move
  and calls :meth:`set_position` again.

Promotion: auto-queen (the common choice for casual play); the text-mode
console supports under-promotion via SAN (``e8=N``) or UCI.
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable

import chess

LIGHT_SQUARE = "#F0D9B5"
DARK_SQUARE = "#B58863"
HIGHLIGHT_LAST = "#F6F669"
HIGHLIGHT_SELECTED = "#F1E06E"
HIGHLIGHT_CHECK = "#E8736B"
DOT_COLOR = "#15781B"
RING_COLOR = "#15781B"
COORD_COLOR = "#8A6A44"

# Solid (black) glyphs are used for both colors; readability comes from fill.
PIECE_GLYPH = {
    chess.PAWN: "\u265F",
    chess.KNIGHT: "\u265E",
    chess.BISHOP: "\u265D",
    chess.ROOK: "\u265C",
    chess.QUEEN: "\u265B",
    chess.KING: "\u265A",
}

FILES = "abcdefgh"


class ChessBoardWidget(tk.Frame):
    """Pure-view chess board.  The owner owns the game state."""

    def __init__(
        self,
        master,
        cell: int = 64,
        margin: int = 26,
        on_move: Callable[[chess.Move], None] | None = None,
        white_bottom: bool = True,
    ) -> None:
        super().__init__(master)
        self.cell = int(cell)
        self.margin = int(margin)
        self.on_move = on_move
        self.white_bottom = bool(white_bottom)

        self.board: chess.Board = chess.Board()
        self.selected: int | None = None
        self.last_move: chess.Move | None = None
        self._targets: set[int] = set()

        size = self.cell * 8 + self.margin * 2
        self.canvas = tk.Canvas(self, width=size, height=size,
                                highlightthickness=0, bg="#3B3937")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self._font = ("Segoe UI Symbol", int(self.cell * 0.60))
        self._coord_font = ("Segoe UI", max(9, int(self.margin * 0.45)))

    # ------------------------------------------------------------ public API
    def set_position(self, board: chess.Board,
                     last_move: chess.Move | None = None) -> None:
        """Render ``board`` (a copy is kept; the widget never mutates it)."""
        self.board = board.copy()
        self.last_move = last_move
        self.selected = None
        self._targets = set()
        self.render()

    def flip(self) -> None:
        self.white_bottom = not self.white_bottom
        self.render()

    # ------------------------------------------------------------ geometry
    def _square_origin(self, square: int) -> tuple[int, int]:
        """Canvas (x, y) of the square's top-left corner, orientation-aware."""
        f = chess.square_file(square)
        r = chess.square_rank(square)
        col = f if self.white_bottom else 7 - f
        row = 7 - r if self.white_bottom else r
        return self.margin + col * self.cell, self.margin + row * self.cell

    def _xy_to_square(self, x: int, y: int) -> int | None:
        fx = (x - self.margin) // self.cell
        fy = (y - self.margin) // self.cell
        if not (0 <= fx < 8 and 0 <= fy < 8):
            return None
        col, row = int(fx), int(fy)
        file = col if self.white_bottom else 7 - col
        rank = 7 - row if self.white_bottom else row
        return chess.square(file, rank)

    def _square_center(self, square: int) -> tuple[int, int]:
        x, y = self._square_origin(square)
        return x + self.cell // 2, y + self.cell // 2

    # ------------------------------------------------------------- rendering
    def render(self) -> None:
        c = self.canvas
        c.delete("all")
        last = self.last_move
        check_square = (
            self.board.king(self.board.turn) if self.board.is_check() else None
        )
        for sq in chess.SQUARES:
            x, y = self._square_origin(sq)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            color = DARK_SQUARE if dark else LIGHT_SQUARE
            if last is not None and sq in (last.from_square, last.to_square):
                color = HIGHLIGHT_LAST
            elif sq == self.selected:
                color = HIGHLIGHT_SELECTED
            elif sq == check_square:
                color = HIGHLIGHT_CHECK
            c.create_rectangle(x, y, x + self.cell, y + self.cell,
                               fill=color, outline="", width=0)

        # coordinates
        for i in range(8):
            f_letter = FILES[i] if self.white_bottom else FILES[7 - i]
            x, _ = self._square_center(chess.square(i, 0))
            c.create_text(x, self.margin + 8 * self.cell + self.margin // 2,
                          text=f_letter, fill=COORD_COLOR, font=self._coord_font)
            r_number = str(8 - i) if self.white_bottom else str(i + 1)
            _, y = self._square_center(chess.square(0, 7 - i))
            c.create_text(self.margin // 2, y,
                          text=r_number, fill=COORD_COLOR, font=self._coord_font)

        # legal-target markers for the current selection
        if self.selected is not None:
            for sq in self._targets:
                cx, cy = self._square_center(sq)
                if self.board.piece_at(sq) is not None:
                    ring_r = self.cell * 0.48
                    c.create_oval(cx - ring_r, cy - ring_r,
                                  cx + ring_r, cy + ring_r,
                                  outline=RING_COLOR,
                                  width=max(3, self.cell // 16))
                else:
                    r = self.cell * 0.16
                    c.create_oval(cx - r, cy - r, cx + r, cy + r,
                                  fill=DOT_COLOR, outline="", width=0)

        # pieces (shadow pass keeps white pieces readable on light squares)
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            cx, cy = self._square_center(sq)
            glyph = PIECE_GLYPH[piece.piece_type]
            if piece.color == chess.WHITE:
                c.create_text(cx + 2, cy + 2, text=glyph, font=self._font,
                              fill="#4A4A4A")
                c.create_text(cx, cy, text=glyph, font=self._font, fill="#FFFFFF")
            else:
                c.create_text(cx + 2, cy + 2, text=glyph, font=self._font,
                              fill="#666666")
                c.create_text(cx, cy, text=glyph, font=self._font, fill="#141414")

    # ------------------------------------------------------------ interaction
    def _legal_targets(self, from_square: int) -> set[int]:
        return {
            m.to_square
            for m in self.board.legal_moves
            if m.from_square == from_square
        }

    @staticmethod
    def _promotion_piece_type(board: chess.Board, move: chess.Move):
        """Auto-queen: return QUEEN if from->to has any promotion, else None."""
        for m in board.legal_moves:
            if (m.from_square == move.from_square
                    and m.to_square == move.to_square
                    and m.promotion is not None):
                return chess.QUEEN
        return None

    def _on_click(self, event) -> None:
        square = self._xy_to_square(event.x, event.y)
        if square is None:
            return
        if self.selected is not None and square in self._targets:
            move = chess.Move(self.selected, square)
            promo = self._promotion_piece_type(self.board, move)
            if promo is not None:
                move = chess.Move(self.selected, square, promotion=promo)
            self.selected = None
            self._targets = set()
            if self.on_move is not None:
                self.on_move(move)
            return
        piece = self.board.piece_at(square)
        if piece is not None and piece.color == self.board.turn:
            if self.selected == square:
                self.selected = None
                self._targets = set()
            else:
                self.selected = square
                self._targets = self._legal_targets(square)
        else:
            self.selected = None
            self._targets = set()
        self.render()