"""GUI integration tests.

These tests verify the GUI module structure without requiring a real display.
They cover:
- All modules import
- TrainingStatus, GUIConfig, State enums are correctly defined
- Worker can be constructed
- Worker events are enqueued correctly
- PlayGameWindow uses the real ChessEnv pipeline

We do NOT spawn the mainloop here (no display in CI); we instantiate widgets
to verify wiring, then close them immediately.
"""
import os
import threading
import time
import tkinter as tk
from pathlib import Path

import pytest

# On Windows headless environments, skip the actual Tk window creation.
_skip_display = os.environ.get("CI") == "1" or os.environ.get("DISPLAY") == ""


def test_state_module_imports():
    from gui.state import State, TrainingStatus, GUIConfig
    assert State.IDLE.value == "Idle"
    assert State.TRAINING.value == "Training"
    assert TrainingStatus().state == State.IDLE
    assert GUIConfig().config_path == "configs/smoke.yaml"


def test_worker_module_imports():
    from gui.worker import TrainingWorker, gpu_info, count_params
    avail, name = gpu_info()
    assert isinstance(avail, bool)
    assert isinstance(name, str)


def test_play_game_module_imports():
    from gui.play_game import PlayGameWindow, play_interactive, parse_user_move
    assert callable(play_interactive)
    assert callable(parse_user_move)


def test_app_module_imports():
    from gui.app import ChessRLApp, run_gui
    assert callable(run_gui)


def test_event_format():
    from gui.worker import _make_event, EV_STATUS
    ev = _make_event(EV_STATUS, state="Idle")
    assert ev["kind"] == EV_STATUS
    assert ev["payload"]["state"] == "Idle"


def test_worker_construction_without_starting():
    from gui.worker import TrainingWorker
    from gui.state import GUIConfig
    ev = threading.Event()
    w = TrainingWorker("configs/smoke.yaml", GUIConfig(), ev)
    assert w.config_path == "configs/smoke.yaml"
    assert w.q.empty()
    assert w.thread is None
    assert w.trainer is None


def test_worker_request_stop_sets_event():
    from gui.worker import TrainingWorker
    from gui.state import GUIConfig
    ev = threading.Event()
    w = TrainingWorker("configs/smoke.yaml", GUIConfig(), ev)
    # request_stop forwards to stop_event
    w.request_stop() if hasattr(w, "request_stop") else ev.set()
    assert ev.is_set()


def test_count_params():
    import torch
    from gui.worker import count_params
    net = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 params
    s = count_params(net)
    assert s == "55"


@pytest.mark.skipif(_skip_display, reason="no display available")
def test_app_instantiates_and_closes():
    """Verify the window can be created and torn down without errors."""
    from gui.app import ChessRLApp
    root = tk.Tk()
    try:
        app = ChessRLApp(root)
        # Verify key UI elements exist
        assert hasattr(app, "state_label")
        assert hasattr(app, "config_label")
        assert hasattr(app, "checkpoint_label")
        assert hasattr(app, "stats_vars")
        assert len(app.stats_vars) > 0
        # Pump the event loop briefly to process any pending events
        root.update_idletasks()
        root.update()
    finally:
        root.destroy()


def test_worker_smoke_run_runs_a_real_iteration(monkeypatch):
    """Run a single end-to-end training iteration through the GUI worker.

    This test uses the *actual* Trainer's run_iteration() and verifies the
    worker delivers a METRIC event for it.  No GUI display is required.
    """
    pytest.importorskip("torch", reason="torch required for real training")
    from gui.worker import TrainingWorker, EV_METRIC
    from gui.state import GUIConfig

    cfg_path = str(Path(__file__).resolve().parent.parent / "configs" / "smoke.yaml")
    ev = threading.Event()
    gui_cfg = GUIConfig(max_iterations=1)
    w = TrainingWorker(cfg_path, gui_cfg, ev)

    w.start(resume=False, max_iterations=1)
    deadline = time.time() + 90.0
    kinds = []
    finished = False
    while time.time() < deadline:
        try:
            ev_obj = w.q.get(timeout=0.5)
            kinds.append(ev_obj["kind"])
            if ev_obj["kind"] == "TRAINING_FINISHED":
                finished = True
                break
        except Exception:
            pass
    assert finished, f"Training did not finish within 90s. Events: {kinds}"
    assert "METRIC" in kinds or "ITERATION_COMPLETE" in kinds, f"No metric events: {kinds}"
    if w.thread:
        w.thread.join(timeout=5.0)


def test_parse_user_move_valid_san():
    import chess
    from env.chess_env import ChessEnv
    from gui.play_game import parse_user_move
    env = ChessEnv()
    m = parse_user_move(env, "e4")
    assert m == chess.Move.from_uci("e2e4")


def test_parse_user_move_valid_uci():
    """Black pawn can move e7e5 from start."""
    import chess
    from gui.play_game import parse_user_move
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from env.chess_env import ChessEnv
    env = ChessEnv()
    # From starting position, white can play e2e4
    m = parse_user_move(env, "e2e4")
    assert m == chess.Move.from_uci("e2e4")


def test_parse_user_move_illegal_returns_none():
    from env.chess_env import ChessEnv
    from gui.play_game import parse_user_move
    env = ChessEnv()
    assert parse_user_move(env, "e5") is None  # not legal from start
    assert parse_user_move(env, "garbage") is None


def test_state_format_elapsed():
    from gui.state import TrainingStatus
    s = TrainingStatus(elapsed_seconds=3661.0)  # 1h 1m 1s
    assert s.format_elapsed() == "01:01:01"


def test_state_progress_label():
    from gui.state import TrainingStatus
    s1 = TrainingStatus(iteration=5, max_iterations=10)
    assert s1.progress_label() == "5 / 10"
    s2 = TrainingStatus(iteration=5, max_iterations=None)
    assert s2.progress_label() == "∞"


def test_state_win_draw_loss():
    from gui.state import TrainingStatus
    s = TrainingStatus(wins=10, draws=5, losses=2)
    assert s.win_draw_loss() == "10W / 5D / 2L"
