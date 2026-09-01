"""Training driver: the fixed iteration loop (spec Section 64).

    1. freeze theta_old            (net in eval mode; nothing recomputed)
    2. generate self-play
    3. collect Stockfish information
    4. calculate reward components
    5. calculate game-value targets
    6. calculate training-value targets
    7. calculate GAE
    8. PPO updates
    9. log metrics
   10. checkpoint
   11. periodic evaluation

Hard-fail discipline (Sections 55/81): a NaN/Inf loss or gradient raises
``NonFiniteLossError``; the failing batch, configuration, and model state
are dumped to ``logs/failure_*`` before the run stops.  Nothing is silently
repaired.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from engine.stockfish_pool import StockfishPool
from evaluation.matches import evaluate_suite
from evaluation.rating import RatingHistory, build_evaluation_result
from model.network import ChessNet
from selfplay.selfplay import play_games
from train.batch import build_ppo_batch
from train.checkpoint import (
    build_metadata,
    check_config_compatibility,
    find_latest_checkpoint,
    load_checkpoint,
    save_training_state,
)
from train.ppo import NonFiniteLossError, PPOConfig, ppo_update

_REPO = Path(__file__).resolve().parent.parent


def seed_everything(seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {"python": seed, "numpy": seed % (2**32), "torch": seed, "cuda": seed}


def get_rng_states() -> dict:
    """Snapshot every RNG that influences training (checkpoint payload)."""
    states: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict) -> None:
    """Restore a snapshot produced by :func:`get_rng_states` (resume)."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


class Trainer:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.name = cfg["project"]["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = seed_everything(cfg["project"]["seed"])

        raw_net = ChessNet(
            channels=cfg["model"]["channels"],
            num_blocks=cfg["model"]["residual_blocks"],
            use_sf_head=cfg["model"]["use_sf_head"],
            norm_groups=cfg["model"]["norm_groups"],
        ).to(self.device)
        # ``_net_module`` is the RAW module: always the state_dict/load target
        # so checkpoints stay compatible with torch.compile (keys of a
        # compiled wrapper get a "_orig_mod." prefix).
        self._net_module = raw_net
        self.net = raw_net
        perf = cfg.get("performance", {})
        if perf.get("torch_compile"):
            try:
                self.net = torch.compile(raw_net)
                print("[trainer] torch.compile enabled", flush=True)
            except Exception as exc:
                print(f"[trainer] torch.compile unavailable: {exc}", flush=True)
        self.optimizer = torch.optim.AdamW(
            raw_net.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
        self.ppo_cfg = PPOConfig(
            clip_eps=cfg["rl"]["ppo_clip"],
            epochs=cfg["ppo"]["epochs"],
            batch_size=cfg["ppo"]["batch_size"],
            entropy_coef=cfg["rl"]["entropy_coef"],
            training_value_coef=cfg["rl"]["training_value_coef"],
            game_value_coef=cfg["rl"]["game_value_coef"],
            stockfish_value_coef=cfg["rl"]["stockfish_value_coef"],
            max_grad_norm=1.0,
            amp=bool(perf.get("amp", False)),
            pinned_memory=bool(perf.get("pinned_memory", False)),
        )

        self.pool: StockfishPool | None = None
        if cfg["engine"]["enabled"]:
            self.pool = StockfishPool(cfg["engine"])

        self.iteration = 0
        self.total_games = 0
        self.total_plies = 0
        self.total_engine_evals = 0
        self.last_rating_summary: dict | None = None
        self.start_time = time.time()

        self.ckpt_dir = _REPO / "checkpoints"
        self.log_path = _REPO / "logs" / f"train_{self.name}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ lifecycle
    def load_net_state(self, state_dict: dict) -> None:
        """Load weights into the RAW module (torch.compile safe)."""
        self._net_module.load_state_dict(state_dict)

    def resume(self) -> bool:
        ckpt = find_latest_checkpoint(self.ckpt_dir, self.name)
        if ckpt is None:
            return False
        payload = load_checkpoint(ckpt)
        check_config_compatibility(payload, self.cfg)
        self.load_net_state(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.iteration = int(payload["iteration"])
        self.total_games = int(payload.get("total_games", 0))
        self.total_plies = int(payload.get("total_plies", 0))
        self.total_engine_evals = int(payload.get("total_engine_evals", 0))
        rng_states = payload.get("rng_states")
        if rng_states is not None:
            # Restore Python/NumPy/Torch(/CUDA) RNG state so a resumed run
            # continues the same random stream (audit M-3).  Checkpoints from
            # older versions without rng_states resume with fresh streams.
            set_rng_states(rng_states)
            self._log({"event": "rng_states_restored"})
        self._log({"event": "resume", "checkpoint": str(ckpt),
                   "iteration": self.iteration})
        return True

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()

    # ------------------------------------------------------------------ io
    def _log(self, record: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _checkpoint(self) -> None:
        payload = {
            "model": self._net_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "total_games": self.total_games,
            "total_plies": self.total_plies,
            "total_engine_evals": self.total_engine_evals,
            "config": self.cfg,
            "seeds": self.seeds,
            "rng_states": get_rng_states(),
            "meta": build_metadata(self.cfg),
            "elapsed_hours": (time.time() - self.start_time) / 3600.0,
        }
        save_training_state(
            self.ckpt_dir,
            self.name,
            payload,
            archive_every=self.cfg["train"]["eval_every"],
            iteration=self.iteration,
        )

    def _dump_failure(self, exc: Exception, batch: dict) -> None:
        """Save failing batch/config/model, then re-raise (Sections 55/81)."""
        out = _REPO / "logs" / f"failure_{self.name}_iter{self.iteration}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "failing_batch.npz",
            **{k: v for k, v in batch.items() if v is not None},
        )
        with open(out / "config.json", "w", encoding="utf-8") as fh:
            json.dump(self.cfg, fh, default=str, indent=2)
        torch.save({"model": self.net.state_dict(), "error": str(exc)}, out / "state.pt")
        self._log({"event": "failure", "iteration": self.iteration,
                   "error": str(exc), "dump": str(out)})

    # ---------------------------------------------------------------- loop
    def train(self, hours: float | None, max_iterations: int | None) -> None:
        deadline = None if hours is None else time.time() + hours * 3600.0
        while True:
            if max_iterations is not None and self.iteration >= max_iterations:
                break
            if deadline is not None and time.time() >= deadline:
                break
            self.run_iteration()
            if self.cfg["train"]["eval_every"] > 0 and (
                self.iteration % self.cfg["train"]["eval_every"] == 0
            ):
                self.evaluate()

    def run_iteration(self) -> dict:
        self.iteration += 1
        t0 = time.time()
        sp = self.cfg["selfplay"]

        # 1-3: frozen behavior policy generates self-play + engine signals.
        perf = self.cfg.get("performance", {})
        trajs = play_games(
            net=self.net,
            rl=self.cfg["rl"],
            pool=self.pool,
            n_games=sp["games_per_iteration"],
            engine_depth=self.cfg["engine"]["depth"] if self.pool else 0,
            max_plies=sp["max_plies"],
            temperature=sp["temperature"],
            device=self.device,
            seed=self.cfg["project"]["seed"] + 7919 * self.iteration,
            use_sf_head=self.cfg["model"]["use_sf_head"],
            threads=sp["threads"],
            infer_batch_size=int(perf.get("inference_batch_size", 1)),
            infer_max_wait_ms=float(perf.get("max_inference_wait_ms", 0.0)),
        )

        # 4-7: rewards, targets, GAE.
        batch, stats = build_ppo_batch(
            trajs, self.cfg["rl"], use_sf_head=self.cfg["model"]["use_sf_head"]
        )

        # 8: PPO update (hard-fail on non-finite values).
        try:
            ppo_metrics = ppo_update(self.net, batch, self.ppo_cfg, self.optimizer)
        except (NonFiniteLossError, ValueError) as exc:
            self._dump_failure(exc, batch)
            self._checkpoint()  # preserve last good state before stopping
            raise

        # 9: log metrics (Section 82).
        wall = time.time() - t0
        self.total_games += stats["games"]
        self.total_plies += stats["plies"]
        self.total_engine_evals += stats["engine_evals"]
        record = {
            "event": "iteration",
            "iteration": self.iteration,
            "wall_time_s": round(wall, 2),
            **stats,
            **{k: round(float(v), 6) for k, v in ppo_metrics.items()},
            "total_games": self.total_games,
            "total_plies": self.total_plies,
            "total_engine_evals": self.total_engine_evals,
            "elapsed_hours": round((time.time() - self.start_time) / 3600.0, 4),
        }
        self._log(record)
        print(
            f"[iter {self.iteration:>4}] {wall:6.1f}s  games={stats['games']} "
            f"plies={stats['plies']}  dS={stats['mean_delta_s']:+.3f} "
            f"regret={stats['mean_regret']:+.3f}  "
            f"H={ppo_metrics['entropy']:.3f}  "
            f"vtrain={ppo_metrics['value_train_loss']:.3f}  "
            f"kl={ppo_metrics['approx_kl']:+.4f}  "
            f"clip={ppo_metrics['clip_fraction']:.2f}",
            flush=True,
        )

        # 10: checkpoint.
        if self.cfg["train"]["checkpoint_every"] > 0 and (
            self.iteration % self.cfg["train"]["checkpoint_every"] == 0
        ):
            self._checkpoint()
        return record

    # ---------------------------------------------------------- evaluation
    def evaluate(self) -> list[dict]:
        opponents: list = ["random", "material"]
        engine_path: str | None = None
        if self.cfg["engine"]["enabled"] and os.path.exists(self.cfg["engine"]["path"]):
            # Fixed-depth Stockfish anchors evaluation to an engine baseline
            # so "playing strength" is not measured only against trivial
            # opponents.  Settings are frozen inside StockfishDepthOpponent.
            opponents += ["stockfish_d1", "stockfish_d4"]
            engine_path = self.cfg["engine"]["path"]
        n = self.cfg["train"]["eval_games"]
        # Evaluation has its own ply budget; None -> training self-play budget.
        max_plies = (
            self.cfg["train"].get("eval_max_plies")
            or self.cfg["selfplay"]["max_plies"]
        )
        results = evaluate_suite(
            self.net, opponents, n, max_plies=max_plies,
            temperature=0.25, device=self.device,
            seed=self.cfg["project"]["seed"] + 104729 * self.iteration,
            engine_path=engine_path,
        )
        # Structured rating (Relative Elo, inverse-variance weighted pool
        # estimate) -- persisted to reports/ratings_<name>.jsonl so the GUI
        # and later sessions can display the latest known evaluation.
        eval_result = build_evaluation_result(results)
        RatingHistory(self.name).append(eval_result, self.iteration)
        self.last_rating_summary = eval_result.summary()
        self._log({"event": "evaluation", "iteration": self.iteration,
                   "results": results,
                   "rating": self.last_rating_summary})
        for r in results:
            elo_txt = (
                f"  elo={r['elo_diff']:+.0f}" if r["elo_diff"] is not None else ""
            )
            trunc_txt = f"  trunc={r['truncated']}" if r.get("truncated") else ""
            print(
                f"   eval vs {r['opponent']:<12}: W-D-L {r['wins']}-{r['draws']}-"
                f"{r['losses']}  score={r['score']:.3f}{elo_txt}{trunc_txt}",
                flush=True,
            )
        rating_txt = ""
        if eval_result.rating is not None:
            rating_txt = (
                f"\n   {eval_result.rating_label}: {eval_result.rating:.0f}"
                f" +/- {eval_result.rating_uncertainty:.0f}"
                f"  (games={eval_result.games}, score={eval_result.score:.3f})"
            )
        else:
            rating_txt = "\n   Relative Elo: N/A (no opponent matchup qualified)"
        print(rating_txt, flush=True)
        return results
