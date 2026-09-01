"""Project entry point: training / evaluation / smoke test.

Usage:
    python main.py --config configs/smoke.yaml                      # Gate 7 smoke
    python main.py --config configs/combined.yaml --hours 4        # Experiment C
    python main.py --config configs/terminal_only.yaml --hours 4   # Experiment A
    python main.py --config configs/combined.yaml --resume         # continue run
    python main.py --config configs/combined.yaml --eval-only --checkpoint checkpoints/combined_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.config import config_overrides, load_config
from train.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="BrewZero chess RL")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hours", type=float, default=None,
                        help="training time budget in hours")
    parser.add_argument("--iterations", type=int, default=None,
                        help="training iteration budget")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest valid checkpoint")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny configuration for the Gate 7 smoke test")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    overrides = config_overrides(hours=args.hours, max_iterations=args.iterations)
    if args.smoke:
        overrides.update(
            {
                "project": {"name": "smoke", "seed": 7},
                "engine": {"depth": 8, "workers": 2, "hash_mb": 64},
                "model": {"channels": 64, "residual_blocks": 2, "norm_groups": 8},
                "selfplay": {"games_per_iteration": 4, "max_plies": 60,
                             "threads": 4},
                "ppo": {"epochs": 2, "batch_size": 128},
                "train": {"eval_every": 1, "eval_games": 4},
            }
        )
    cfg = load_config(args.config, overrides)

    trainer = Trainer(cfg)
    try:
        if args.eval_only:
            if not args.checkpoint:
                parser.error("--eval-only requires --checkpoint")
            from train.checkpoint import check_config_compatibility, load_checkpoint

            payload = load_checkpoint(args.checkpoint, map_location=trainer.device)
            check_config_compatibility(payload, cfg)
            trainer.load_net_state(payload["model"])
            trainer.iteration = int(payload["iteration"])
            trainer.evaluate()
            return

        if args.resume:
            trainer.resume()
        trainer.train(hours=args.hours, max_iterations=args.iterations)
    except KeyboardInterrupt:
        # Graceful Ctrl+C: persist state for later resume (Section 60).
        print("\nCtrl+C received -- saving checkpoint for resume...")
        trainer._checkpoint()
        print("Checkpoint saved. Resume with --resume.")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()