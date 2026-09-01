"""Play a game against a trained checkpoint from the command line.

Usage:
    python play.py --checkpoint checkpoints/combined_latest.pt [--human white|black] [--temp 0.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from env.action_space import ActionCodec
from env.chess_env import ChessEnv, result_for_player
from env.encoding import encode_state
from model.network import ChessNet
from selfplay.selfplay import _infer
from selfplay.sampling import sample_action
from train.checkpoint import check_config_compatibility, load_checkpoint


def build_net(checkpoint_path: str, device: str):
    payload = load_checkpoint(checkpoint_path, map_location=device)
    cfg = payload["config"]
    net = ChessNet(
        channels=cfg["model"]["channels"],
        num_blocks=cfg["model"]["residual_blocks"],
        use_sf_head=cfg["model"]["use_sf_head"],
        norm_groups=cfg["model"]["norm_groups"],
    ).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net, cfg, int(payload["iteration"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--human", choices=("white", "black"), default="white")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="net sampling temperature (0 = greedy)")
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    net, cfg, iteration = build_net(args.checkpoint, device)
    rng = np.random.default_rng(0)
    human_is_white = args.human == "white"

    env = ChessEnv()
    print(f"Checkpoint iteration: {iteration}")
    print("Enter moves as SAN (e4, Nf3, O-O) or UCI (e2e4). 'quit' exits.\n")

    while True:
        print(env.board.unicode(borders=True, empty_square="."))
        turn = "White" if env.turn == chess.WHITE else "Black"
        print(f"\n{turn} to move. Ply {env.ply}.")

        human_turn = (env.turn == chess.WHITE) == human_is_white
        if human_turn:
            raw = input("your move: ").strip()
            if raw.lower() in ("quit", "exit"):
                return
            try:
                try:
                    move = env.board.parse_san(raw)
                except ValueError:
                    move = chess.Move.from_uci(raw)
                action = ActionCodec.encode(move)
            except ValueError:
                print("could not parse move; try again.\n")
                continue
        else:
            canon = env.canonical_board()
            logits, _, _, _ = _infer(net, encode_state(canon), device)
            legal = ActionCodec.encode_legal_moves(canon)
            action = sample_action(logits, legal, args.temp, rng)
            print(f"net plays: {env.canonical_action_to_move(action).uci()}\n")

        try:
            env.step(action)
        except ValueError as exc:
            print(f"illegal action: {exc}\n")
            continue

        if env.is_terminal():
            z = result_for_player(env.board, human_is_white)
            print(env.board.unicode(borders=True, empty_square="."))
            print("RESULT:", "you win" if z == 1 else "you lose" if z == -1 else "draw")
            return
        if env.ply >= 300:
            print("Game truncated at 300 plies (draw for scoring).")
            return


if __name__ == "__main__":
    main()