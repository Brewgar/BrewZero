"""Worker thread wrapping the existing Trainer/Evaluation API.

The worker runs in a background thread.  It never touches Tkinter directly;
it pushes events onto a ``queue.Queue`` that the GUI thread polls.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
from pathlib import Path

import torch

from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import config_overrides, load_config
from train.train import Trainer

# Event types pushed onto the GUI queue.
EV_STATUS = "STATUS"
EV_METRIC = "METRIC"
EV_ITERATION = "ITERATION_COMPLETE"
EV_EVALUATION = "EVALUATION_COMPLETE"
EV_CHECKPOINT = "CHECKPOINT_SAVED"
EV_MESSAGE = "LOG_MESSAGE"
EV_ERROR = "ERROR"
EV_FINISHED = "TRAINING_FINISHED"
EV_PLAY_READY = "PLAY_READY"

_REPO = Path(__file__).resolve().parent.parent


def _make_event(kind: str, **payload) -> dict:
    return {"kind": kind, "payload": payload}


class TrainingWorker:
    """Background thread that drives the existing Trainer / evaluation APIs.

    The GUI thread polls ``self.q`` (a ``queue.Queue``).  All GUI interaction
    is routed through these events; the worker never touches Tkinter.
    """

    def __init__(self, config_path: str, gui_config, stop_event: threading.Event):
        """config_path  : path to the YAML experiment config (str).
        gui_config     : gui.state.GUIConfig dataclass (or None).
        stop_event     : threading.Event the GUI sets to request graceful stop.
        """
        self.config_path = config_path
        self.gui_config = gui_config
        self.stop_event = stop_event
        self.q: queue.Queue = queue.Queue()
        self.trainer: Trainer | None = None
        self.thread: threading.Thread | None = None
        self.max_iterations_override: int | None = None
        self._log = logging.getLogger("gui.worker")

    # ------------------------------------------------------------------ start
    def start(self, resume: bool = False, max_iterations: int | None = None):
        self.max_iterations_override = max_iterations

        def _run():
            try:
                self._bootstrap()
                if resume:
                    self._do_resume()
                else:
                    self._do_train()
            except Exception as exc:
                self._push(EV_ERROR, message=repr(exc), traceback=traceback.format_exc())
                self._push(EV_FINISHED, reason="error")
            finally:
                try:
                    if self.trainer is not None:
                        self.trainer.close()
                except Exception:
                    pass

        self.thread = threading.Thread(target=_run, daemon=True, name="gui-worker")
        self.thread.start()

    def start_evaluation(self, checkpoint_path: str):
        def _run():
            try:
                self._bootstrap()
                self._do_evaluate(checkpoint_path)
            except Exception as exc:
                self._push(EV_ERROR, message=repr(exc), traceback=traceback.format_exc())
            finally:
                try:
                    if self.trainer is not None:
                        self.trainer.close()
                except Exception:
                    pass

        self.thread = threading.Thread(target=_run, daemon=True, name="gui-eval")
        self.thread.start()

    def start_play(self, checkpoint_path: str, human_white: bool = True,
                   temperature: float = 0.0):
        def _run():
            try:
                self._bootstrap()
                self._do_play(checkpoint_path, human_white, temperature)
            except Exception as exc:
                self._push(EV_ERROR, message=repr(exc), traceback=traceback.format_exc())
            finally:
                try:
                    if self.trainer is not None:
                        self.trainer.close()
                except Exception:
                    pass

        self.thread = threading.Thread(target=_run, daemon=True, name="gui-play")
        self.thread.start()

    def request_stop(self):
        self.stop_event.set()
        self._push(EV_STATUS, state="Stopping")

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # ------------------------------------------------------------------ internals
    def _push(self, kind: str, **payload):
        self.q.put(_make_event(kind, **payload))

    def _bootstrap(self):
        hours = getattr(self.gui_config, "train_hours", None)
        overrides = config_overrides(hours=hours)
        cfg = load_config(self.config_path, overrides)
        self.cfg = cfg
        self.trainer = Trainer(cfg)
        self._push(EV_STATUS, state="Training")
        self._push(EV_MESSAGE, message=f"Loaded config: {self.config_path}")

    def _do_resume(self):
        if self.trainer.resume():
            self._push(EV_MESSAGE, message="Resumed from latest checkpoint")
            self._train_loop()
        else:
            self._push(EV_MESSAGE, message="No valid checkpoint found -- starting fresh")
            self._train_loop()

    def _do_train(self):
        self._push(EV_MESSAGE, message="Training started")
        self._train_loop()

    def _train_loop(self):
        tr = self.trainer
        deadline = None
        max_iter = self.max_iterations_override
        hours = getattr(self.gui_config, "train_hours", None)
        if hours is not None:
            import time
            deadline = time.time() + hours * 3600.0
        if max_iter is None:
            max_iter = self.cfg.get("train", {}).get("max_iterations")

        while not self.stop_event.is_set():
            if deadline is not None:
                import time
                if time.time() >= deadline:
                    break
            if max_iter is not None and tr.iteration >= max_iter:
                break
            try:
                rec = tr.run_iteration()
            except Exception as exc:
                self._push(EV_ERROR, message=repr(exc), traceback=traceback.format_exc())
                return
            self._push(EV_ITERATION, record=rec)
            self._push(EV_METRIC, record=rec)
            # checkpoint
            if rec.get("event") == "iteration":
                self._push(EV_CHECKPOINT, iteration=rec["iteration"])
            # periodic evaluation
            if (
                self.cfg.get("train", {}).get("eval_every", 0) > 0
                and rec["iteration"] % self.cfg["train"]["eval_every"] == 0
            ):
                try:
                    results = tr.evaluate()
                    self._push(EV_EVALUATION, results=results,
                               rating=getattr(tr, "last_rating_summary", None),
                               iteration=rec["iteration"])
                except Exception as exc:
                    self._push(EV_ERROR, message=repr(exc), traceback=traceback.format_exc())
                    return
        self._push(EV_FINISHED, reason="completed")

    def _do_evaluate(self, checkpoint_path: str):
        self._push(EV_STATUS, state="Evaluating")
        self._push(EV_MESSAGE, message=f"Evaluating checkpoint: {checkpoint_path}")
        payload = load_checkpoint(checkpoint_path, map_location=self.trainer.device)
        from train.checkpoint import check_config_compatibility
        check_config_compatibility(payload, self.cfg)
        self.trainer.net.load_state_dict(payload["model"])
        self.trainer.iteration = int(payload.get("iteration", 0))
        results = self.trainer.evaluate()
        self._push(EV_EVALUATION, results=results,
                   rating=getattr(self.trainer, "last_rating_summary", None),
                   iteration=self.trainer.iteration)
        self._push(EV_STATUS, state="Idle")
        self._push(EV_FINISHED, reason="evaluated")

    def _do_play(self, checkpoint_path, human_white=True, temperature=0.0):
        self._push(EV_STATUS, state="Playing")
        self._push(EV_MESSAGE, message=f"Play mode: {checkpoint_path}")
        payload = load_checkpoint(checkpoint_path, map_location=self.trainer.device)
        from train.checkpoint import check_config_compatibility
        check_config_compatibility(payload, self.cfg)
        self.trainer.net.load_state_dict(payload["model"])
        self.trainer.net.eval()
        # Run an interactive game using the existing play pipeline.
        # Output goes to the GUI event log so no console is required.
        from gui.play_game import PlayGameWindow
        self._push(EV_PLAY_READY,
                   net=self.trainer.net, device=self.trainer.device,
                   cfg=self.cfg, human_white=human_white, temperature=temperature)


def count_params(net) -> str:
    n = sum(p.numel() for p in net.parameters())
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n:,}"


def gpu_info() -> tuple[bool, str]:
    avail = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if avail else "none"
    return avail, name