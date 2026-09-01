"""Model package: residual trunk + policy / game-value / training-value /
optional Stockfish-value heads (Gate 5)."""

from model.network import ChessNet

__all__ = ["ChessNet"]
