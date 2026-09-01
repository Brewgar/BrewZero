"""Chess environment for self-play.

The environment keeps the *original* board (as the authoritative chess state)
and exposes a *canonical* view for the policy (side to move relabeled as
White).  All decisions are expressed in canonical action indices (see
``env.action_space``); execution converts them to original-board moves.

Terminal rules
--------------
``python-chess`` computes legality.  A state is terminal when:

* checkmate (win for the side that gave mate),
* stalemate, insufficient material, 75-move rule, or fivefold repetition
  (draw),
* a *claimable* 50-move rule draw or *claimable* threefold repetition draw.
  In self-play we adopt the convention that claimable draws are automatically
  claimed (documented choice; keeps game-value targets well defined).

Terminal results are always expressed from a specified player's perspective
(+1 win, 0 draw, -1 loss) and honor zero-sum symmetry.
"""

from __future__ import annotations

import chess
import numpy as np

from env.action_space import ACTION_SPACE_SIZE, ActionCodec
from env.symmetry import canonicalize, map_move_to_original

WIN = 1
DRAW = 0
LOSS = -1


class ChessEnv:
    """Minimal self-play chess environment wrapper."""

    def __init__(self, fen: str | None = None) -> None:
        self._board = chess.Board(fen if fen is not None else chess.STARTING_FEN)
        self._ply = 0

    # ------------------------------------------------------------------ state
    @property
    def board(self) -> chess.Board:
        return self._board

    @property
    def ply(self) -> int:
        return self._ply

    @property
    def turn(self) -> chess.Color:
        return self._board.turn

    def reset(self, fen: str = chess.STARTING_FEN) -> None:
        self._board = chess.Board(fen)
        self._ply = 0

    def canonical_board(self) -> chess.Board:
        return canonicalize(self._board, copy=False)

    # ------------------------------------------------------------- action API
    def legal_mask(self) -> np.ndarray:
        """Dense binary mask of canonical legal moves, shape (ACTION_SPACE_SIZE,)."""
        return np.asarray(ActionCodec.legal_mask(self.canonical_board()), dtype=np.int8)

    def legal_action_indices(self) -> list[int]:
        return ActionCodec.encode_legal_moves(self.canonical_board())

    def canonical_action_to_move(self, action: int) -> chess.Move:
        """Decode a canonical action into the corresponding *original* move."""
        canonical_move = ActionCodec.decode(action)
        return map_move_to_original(canonical_move, self._board.turn)

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> None:
        """Apply a canonical action index.

        Raises ``ValueError`` if the decoded move is not legal (hard error,
        per operating Rule 4).
        """
        move = self.canonical_action_to_move(action)
        if move not in self._board.legal_moves:
            raise ValueError(
                f"illegal action {action} -> move {move.uci()} in position "
                f"{self._board.fen()}"
            )
        self._board.push(move)
        self._ply += 1

    def step_direct(self, move: chess.Move) -> None:
        """Apply a ``chess.Move`` directly on the original board (for opponents).

        Hard error on illegal moves (Rule 4). Does not go through the canonical
        action encoding since the opponent plays on the real board.
        """
        if move not in self._board.legal_moves:
            raise ValueError(
                f"illegal move {move.uci()} in position {self._board.fen()}"
            )
        self._board.push(move)
        self._ply += 1

    # -------------------------------------------------------------- terminals
    def is_terminal(self) -> bool:
        b = self._board
        if b.is_checkmate():
            return True
        if b.is_stalemate() or b.is_insufficient_material():
            return True
        if b.is_seventyfive_moves() or b.is_fivefold_repetition():
            return True
        if b.can_claim_fifty_moves() or b.can_claim_threefold_repetition():
            return True
        return False

    def terminal_result_for_side_to_move(self) -> int:
        """Result from the perspective of the side to move on a terminal board.

        +1 win, 0 draw, -1 loss.  If not terminal, raises ``ValueError``.
        """
        if not self.is_terminal():
            raise ValueError("terminal_result called on a non-terminal board")
        if self._board.is_checkmate():
            return LOSS
        return DRAW

    def result_for_player(self, player: chess.Color) -> int:
        """Zero-sum game result from ``player``'s perspective."""
        z = self.terminal_result_for_side_to_move()
        if self._board.turn == player:
            return z
        return -z


def result_for_mover(prev_board: chess.Board, board: chess.Board) -> int:
    """Terminal game result from the perspective of the player who just moved.

    To be called after ``prev_board``'s mover has pushed a move producing
    ``board``.  ``prev_board.turn`` is the side that moved. Returns ``None`` if
    ``board`` is not terminal.
    """
    if not is_terminal(board):
        return None
    mover = prev_board.turn
    return result_for_player(board, mover)


def is_terminal(board: chess.Board) -> bool:
    """Module-level terminal check (see class docstring for the rules)."""
    if board.is_checkmate():
        return True
    if board.is_stalemate() or board.is_insufficient_material():
        return True
    if board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return True
    if board.can_claim_fifty_moves() or board.can_claim_threefold_repetition():
        return True
    return False


def result_for_player(board: chess.Board, player: chess.Color) -> int:
    """Zero-sum result from ``player``'s perspective on a terminal board."""
    if not is_terminal(board):
        raise ValueError("result_for_player called on a non-terminal board")
    if board.is_checkmate():
        # Side to move is mated and loses.
        if board.turn == player:
            return LOSS
        return WIN
    return DRAW