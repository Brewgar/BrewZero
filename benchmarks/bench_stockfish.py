"""Benchmark Stockfish search throughput (spec Section 62).

Measures completed engine searches/sec for 1..N workers at the configured
depth.  Usage:
    python benchmarks/bench_stockfish.py [--workers 1 2 4 8] [--depth 16] [--n 64]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.stockfish_pool import StockfishPool
from train.config import DEFAULTS


def bench_workers(n_workers: int, depth: int, n_requests: int,
                  concurrency: int = 1) -> tuple[float, float]:
    """Measure pool throughput with ``concurrency`` concurrent submitters.

    concurrency=1 reproduces the old sequential-latency measurement; values
    >1 emulate the concurrent self-play load where multiple game threads
    submit analyses simultaneously and the pool's worker count matters.
    Returns (searches_per_sec, ms_per_search_sequential_avg).
    """
    cfg = dict(DEFAULTS["engine"])
    cfg["workers"] = n_workers
    cfg["depth"] = depth
    with StockfishPool(cfg) as pool:
        # Warm-up.
        pool.analyse(chess.STARTING_FEN, depth)
        t0 = time.time()
        if concurrency <= 1:
            for _ in range(n_requests):
                pool.analyse(chess.STARTING_FEN, depth)
        else:
            import concurrent.futures

            def _one(_: int) -> None:
                pool.analyse(chess.STARTING_FEN, depth)

            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                list(ex.map(_one, range(n_requests)))
        dt = time.time() - t0
    return n_requests / dt, 1000.0 * dt / n_requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--depth", type=int, default=DEFAULTS["engine"]["depth"])
    parser.add_argument("--n", type=int, default=64, help="requests per config")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="concurrent submitters; default = workers count")
    args = parser.parse_args()

    print(f"Stockfish search benchmark (depth={args.depth}, {args.n} searches/config)")
    print(f"{'workers':>8} {'searches/s':>12} {'ms/search':>12}")
    for w in args.workers:
        c = args.concurrency if args.concurrency is not None else w
        rate, ms = bench_workers(w, args.depth, args.n, concurrency=c)
        print(f"{w:>8} {rate:>12.1f} {ms:>12.1f}   (concurrency={c})")


if __name__ == "__main__":
    main()