"""Benchmark end-to-end self-play throughput (spec Sections 62-63).

Reports plies/hour and games/hour with and without engine assistance.
Usage:
    python benchmarks/bench_selfplay.py [--config configs/combined.yaml] [--minutes 5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from engine.stockfish_pool import StockfishPool
from model.network import ChessNet
from selfplay.selfplay import play_games
from train.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--minutes", type=float, default=5.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ChessNet(
        channels=cfg["model"]["channels"],
        num_blocks=cfg["model"]["residual_blocks"],
        use_sf_head=cfg["model"]["use_sf_head"],
    ).to(device)
    net.eval()

    for label, use_pool in (("with-engine", True), ("terminal-only", False)):
        pool = StockfishPool(cfg["engine"]) if use_pool else None
        plies = games = 0
        deadline = time.time() + args.minutes * 60
        try:
            while time.time() < deadline:
                t0 = time.time()
                trajs = play_games(
                    net, cfg["rl"], pool, n_games=cfg["selfplay"]["games_per_iteration"],
                    engine_depth=cfg["engine"]["depth"],
                    max_plies=cfg["selfplay"]["max_plies"],
                    temperature=1.0, device=device, seed=12345,
                    use_sf_head=False, threads=cfg["selfplay"]["threads"],
                )
                plies += sum(t.game_plies for t in trajs)
                games += len(trajs)
                _ = t0
        finally:
            if pool is not None:
                pool.close()
        hours = args.minutes / 60.0
        print(f"{label:>14}: {plies / hours:,.0f} plies/hour  "
              f"{games / hours:,.1f} games/hour")


if __name__ == "__main__":
    main()