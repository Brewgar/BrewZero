"""Fixed evaluation opponents (spec Section 65).

Opponent move choice is expressed over CANONICAL action indices so the same
env/codec machinery applies to both sides.  All configurations stay fixed
across checkpoints during a comparison (Sec. 83).
"""

from __future__ import annotations

import numpy as np

import chess

from env.action_space import ActionCodec

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def random_opponent(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    """Uniform-random legal move."""
    moves = list(board.legal_moves)
    return moves[int(rng.integers(len(moves)))]


def _material(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, color))
        total -= value * len(board.pieces(piece_type, not color))
    return total


def material_greedy_opponent(
    board: chess.Board, rng: np.random.Generator
) -> chess.Move:
    """1-ply material-gain greedy with random tie-breaking (heuristic)."""
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError("material_greedy called on a terminal position")
    mover = board.turn
    best, best_gain = [], None
    for move in moves:
        board.push(move)
        gain = _material(board, mover)
        board.pop()
        if best_gain is None or gain > best_gain:
            best, best_gain = [move], gain
        elif gain == best_gain:
            best.append(move)
    return best[int(rng.integers(len(best)))]


def opponent_by_name(name: str, board: chess.Board, rng: np.random.Generator) -> chess.Move:
    if name == "random":
        return random_opponent(board, rng)
    if name == "material":
        return material_greedy_opponent(board, rng)
    raise ValueError(f"unknown opponent '{name}'")


class StockfishDepthOpponent:
    """Fixed-depth Stockfish evaluation opponent.

    Settings are frozen at construction (Threads=1, fixed Hash, fixed depth)
    so every checkpoint in a comparison faces an identical opponent (spec
    Section 83).  Fixed-depth, single-thread Stockfish is deterministic in
    practice but not contractually; the match RNG does not influence it.

    Lifecycle: callers MUST call :meth:`close` when finished (see
    ``evaluation.matches.evaluate_suite``, which owns the instances).
    """

    def __init__(self, path: str, depth: int, hash_mb: int = 64) -> None:
        import chess.engine

        if depth < 1:
            raise ValueError(f"Stockfish opponent depth must be >= 1, got {depth}")
        self.depth = depth
        self.name = f"stockfish_d{depth}"
        self._limit = chess.engine.Limit(depth=depth)
        self._engine = chess.engine.SimpleEngine.popen_uci(path, debug=False)
        self._engine.configure({"Threads": 1, "Hash": hash_mb})

    def choose(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        result = self._engine.play(board, self._limit)
        if result.move is None:
            raise RuntimeError(f"engine returned no move in {board.fen()}")
        return result.move

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass



def legal_indices_of(board: chess.Board) -> list[int]:
    """Canonical legal action indices (board must be canonical)."""
    return ActionCodec.encode_legal_moves(board)