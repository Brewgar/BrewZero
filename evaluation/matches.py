"""Color-balanced evaluation matches (spec Sections 65-68).

For every opponent matchup: even game count, alternating colors, fixed
configuration, fixed search budget.  The network plays with low temperature
(0.25, documented; deterministic play can stall by repetition).  Records
wins/draws/losses from the NET's perspective, game score (Sec. 67), game
lengths, and an Elo-difference estimate with CI (Sec. 68).
"""

from __future__ import annotations

import chess
import numpy as np

from env.chess_env import ChessEnv, result_for_player
from evaluation.elo import elo_difference
from evaluation.opponents import StockfishDepthOpponent, opponent_by_name
from selfplay.selfplay import _infer
from selfplay.sampling import sample_action
from env.action_space import ActionCodec


def encode_state_(board):
    from env.encoding import encode_state
    return encode_state(board)


def play_match(
    net,
    opponent,
    n_games: int,
    max_plies: int,
    temperature: float,
    device: str,
    seed: int,
) -> dict:
    """Play ``n_games`` (even, color-alternating) games vs a fixed opponent.

    ``opponent`` is either a name understood by :func:`opponent_by_name` or an
    object exposing ``choose(board, rng) -> chess.Move`` (e.g.
    ``StockfishDepthOpponent``).
    """
    if n_games % 2 != 0:
        raise ValueError("n_games must be even for color balance")
    rng = np.random.default_rng(seed)
    wins = draws = losses = truncated = 0
    lengths: list[int] = []

    def _opponent_move(board: chess.Board) -> chess.Move:
        if isinstance(opponent, str):
            return opponent_by_name(opponent, board, rng)
        return opponent.choose(board, rng)

    half = n_games // 2
    for game in range(n_games):
        net_is_white = game < half  # first half White, second half Black
        env = ChessEnv()
        while True:
            canon = env.canonical_board()
            net_to_move = (env.turn == chess.WHITE) == net_is_white
            if net_to_move:
                logits, _, _, _ = _infer(net, encode_state_(canon), device)
                legal = ActionCodec.encode_legal_moves(canon)
                action = sample_action(logits, legal, temperature, rng)
                env.step(action)
            else:
                move = _opponent_move(env.board)
                env.step_direct(move)
            if env.is_terminal():
                break
            if env.ply >= max_plies:
                break

        lengths.append(env.ply)
        if env.is_terminal():
            z_net = result_for_player(env.board, net_is_white)
        else:
            z_net = 0.0  # truncation counts as a draw for match scoring (Sec. 50)
            truncated += 1
        wins += int(z_net == 1.0)
        draws += int(z_net == 0.0)
        losses += int(z_net == -1.0)

    total = wins + draws + losses
    score = (wins + 0.5 * draws) / max(1, total)
    elo, lo, hi = elo_difference(wins, draws, losses)
    return {
        "opponent": (
            opponent if isinstance(opponent, str)
            else getattr(opponent, "name", type(opponent).__name__)
        ),
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        # Truncated games are scored as draws but reported separately: the
        # draw count conflates real draws with unfinished games (audit H-4).
        "truncated": truncated,
        "score": score,
        "avg_game_length": float(np.mean(lengths)) if lengths else 0.0,
        "elo_diff": elo,
        "elo_ci95": (lo, hi),
    }


def _make_opponent(spec, engine_path: str | None):
    """Expand an opponent spec into a usable opponent (or engine instance)."""
    if isinstance(spec, str) and spec.startswith("stockfish_d"):
        if engine_path is None:
            raise ValueError(
                f"opponent '{spec}' requires an engine path (engine_path=None)"
            )
        depth = int(spec[len("stockfish_d"):])
        return StockfishDepthOpponent(engine_path, depth)
    return spec


def evaluate_suite(net, opponents: list, n_games: int, max_plies: int,
                   temperature: float, device: str, seed: int,
                   engine_path: str | None = None) -> list[dict]:
    """Run the fixed opponent suite (same suite for every checkpoint).

    ``opponents`` entries are names (``"random"``, ``"material"``) or
    ``"stockfish_d<depth>"`` specs (requires ``engine_path``).  Engine-backed
    opponents are created here and closed before returning.
    """
    instances: list = []
    specs: list = []
    for opp in opponents:
        created = _make_opponent(opp, engine_path)
        specs.append(created)
        if hasattr(created, "close"):
            instances.append(created)
    try:
        return [
            play_match(net, spec, n_games, max_plies, temperature, device, seed + i)
            for i, spec in enumerate(specs)
        ]
    finally:
        for inst in instances:
            inst.close()

