"""Chess RL training GUI - main application window.

A thin Tkinter/ttk control panel over the existing training, evaluation,
checkpoint, and play APIs.  The GUI never re-implements RL logic; it only
observes the worker thread and forwards user actions to the existing APIs.
"""

from __future__ import annotations

import json
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

from gui.state import State, TrainingStatus
from gui.worker import TrainingWorker, gpu_info

_REPO = Path(__file__).resolve().parent.parent


def _load_gui_config() -> dict:
    path = _REPO / "gui" / "config.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_gui_config(cfg: dict) -> None:
    path = _REPO / "gui" / "config.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


class ChessRLApp:
    """Single-window control panel for the chess-RL training system."""

    REFRESH_MS = 1000  # worker-queue poll interval

    def __init__(self, root: tk.Tk):
        self.root = root
        saved = _load_gui_config()
        self.last_config = saved.get("config", "smoke.yaml")
        self.last_checkpoint = saved.get("checkpoint", "")
        self.hours_default = saved.get("hours", 0.0)

        self.status = TrainingStatus()
        self.worker = None
        self.stop_event = threading.Event()
        self.train_hours = None
        self.max_iterations = None

        self._build_ui()
        self._refresh_runtime_info()
        self._populate_checkpoints()
        self._populate_configs()
        self._refresh_assistance_status()
        self._load_latest_rating_for_config()
        self._start_hw_sampler()
        self._append_log("Chess RL GUI ready.")
        self._append_log("CUDA available: " + ("yes" if self.status.cuda_available else "NO"))
        if self.status.cuda_available:
            self._append_log(f"GPU: {self.status.gpu}")

        self.root.after(self.REFRESH_MS, self._poll_worker_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.root.title("Chess RL - control panel")
        self.root.geometry("900x660")
        self.root.minsize(760, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        pad = {"padx": 8, "pady": 4}
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # ---- status row
        status = ttk.LabelFrame(outer, text="Status", padding=6)
        status.pack(fill="x", **pad)
        self.state_label = ttk.Label(status, text="Idle", font=("", 10, "bold"))
        self.state_label.pack(side="left", padx=(0, 24))
        ttk.Label(status, text="Config:").pack(side="left")
        self.config_label = ttk.Label(status, text="smoke.yaml", width=22, anchor="w")
        self.config_label.pack(side="left", padx=(4, 24))
        ttk.Label(status, text="Checkpoint:").pack(side="left")
        self.checkpoint_label = ttk.Label(status, text="none", width=30, anchor="w")
        self.checkpoint_label.pack(side="left", padx=(4, 0))

        # ---- statistics grid
        stats = ttk.LabelFrame(outer, text="Statistics", padding=6)
        stats.pack(fill="x", **pad)
        self.stats_vars: dict[str, tk.StringVar] = {}

        def stat_row(parent, name, default="—", width=16):
            f = ttk.Frame(parent)
            ttk.Label(f, text=f"{name}:", width=18, anchor="e").pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Label(f, textvariable=var, width=width, anchor="w").pack(side="left", padx=(4, 0))
            f.pack(fill="x", **pad)
            key = name.lower().replace(" ", "_").replace("/", "_")
            self.stats_vars[key] = var

        stat_row(stats, "Experiment", default="—")
        stat_row(stats, "Iteration", "0")
        stat_row(stats, "Games", "0")
        stat_row(stats, "Plies", "0")
        stat_row(stats, "Elapsed Time", "00:00:00")
        stat_row(stats, "Relative Elo", "N/A")
        stat_row(stats, "Eval Games", "—")
        stat_row(stats, "Eval Score", "N/A")
        stat_row(stats, "Win/Draw/Loss", "—")
        stat_row(stats, "Mean Reward", "0.000")
        stat_row(stats, "Mean Dense Reward", "0.000")
        stat_row(stats, "Mean Delta S", "0.000")
        stat_row(stats, "Mean Regret", "0.000")
        stat_row(stats, "Policy Entropy", "0.000")
        stat_row(stats, "PPO Loss", "0.000")
        stat_row(stats, "Value Loss", "0.000")
        stat_row(stats, "Self-play Plies/s", "—")
        stat_row(stats, "Engine Evals/s", "—")
        stat_row(stats, "GPU Util", "—")
        stat_row(stats, "VRAM", "—")
        stat_row(stats, "RAM", "—")
        stat_row(stats, "Stockfish Assistance", "—")
        stat_row(stats, "SF Reward Coef", "—")
        stat_row(stats, "Model Params", "—")
        stat_row(stats, "Device", "—")

        # ---- controls
        ctrl = ttk.LabelFrame(outer, text="Controls", padding=6)
        ctrl.pack(fill="x", **pad)

        btn_row = ttk.Frame(ctrl)
        self.btn_train = ttk.Button(btn_row, text="Start Training", command=self._on_train)
        self.btn_train.pack(side="left", **pad)
        self.btn_stop = ttk.Button(btn_row, text="Stop", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", **pad)
        self.btn_resume = ttk.Button(btn_row, text="Resume", command=self._on_resume, state="disabled")
        self.btn_resume.pack(side="left", **pad)
        self.btn_eval = ttk.Button(btn_row, text="Evaluate", command=self._on_evaluate, state="disabled")
        self.btn_eval.pack(side="left", **pad)
        self.btn_load = ttk.Button(btn_row, text="Load Checkpoint", command=self._on_load, state="normal")
        self.btn_load.pack(side="left", **pad)
        self.btn_play = ttk.Button(btn_row, text="Play Against AI", command=self._on_play, state="normal")
        self.btn_play.pack(side="left", **pad)
        btn_row.pack(fill="x", **pad)

        # config / hours controls
        cfg_row = ttk.Frame(ctrl)
        ttk.Label(cfg_row, text="Config:").pack(side="left")
        self.config_combo = ttk.Combobox(cfg_row, state="readonly", width=20)
        self.config_combo.pack(side="left", padx=(4, 16))
        self.config_combo.bind("<<ComboboxSelected>>", lambda e: self._on_config_change())
        ttk.Label(cfg_row, text="Hours:").pack(side="left")
        self.hours_entry = ttk.Entry(cfg_row, width=8)
        self.hours_entry.insert(0, str(self.hours_default) if self.hours_default else "")
        self.hours_entry.pack(side="left", padx=(4, 0))
        cfg_row.pack(fill="x", **pad)

        # checkpoint selector
        ck_row = ttk.Frame(ctrl)
        ttk.Label(ck_row, text="Checkpoint:").pack(side="left")
        self.checkpoint_combo = ttk.Combobox(ck_row, state="readonly", width=30)
        self.checkpoint_combo.pack(side="left", padx=(4, 0))
        ck_row.pack(fill="x", **pad)

        # play options
        play_row = ttk.Frame(ctrl)
        self.play_color = tk.StringVar(value="White")
        ttk.Label(play_row, text="Play as:").pack(side="left")
        ttk.Radiobutton(play_row, text="White", variable=self.play_color, value="White").pack(side="left", padx=(4, 8))
        ttk.Radiobutton(play_row, text="Black", variable=self.play_color, value="Black").pack(side="left")
        ttk.Label(play_row, text="Temperature:").pack(side="left", padx=(12, 4))
        self.play_temp = tk.StringVar(value="0.0")
        ttk.Entry(play_row, textvariable=self.play_temp, width=6).pack(side="left")
        play_row.pack(fill="x", **pad)

        self.eval_label = ttk.Label(ctrl, text="Evaluation: —")
        self.eval_label.pack(anchor="w", **pad)

        # ---- event log
        log_frame = ttk.LabelFrame(outer, text="Event Log", padding=6)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(log_frame, height=10, wrap="word", font=("", 9))
        self.log_text.pack(side="left", fill="both", expand=True)
        self.log_text.config(state="disabled")
        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        vsb.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=vsb.set)

    # ------------------------------------------------------------------ handlers
    def _selected_config_path(self):
        cfg_name = self.config_combo.get() or self.last_config
        return str((_REPO / "configs" / cfg_name).resolve())

    def _on_config_change(self):
        self.last_config = self.config_combo.get()
        self.config_label.config(text=self.last_config)
        self._refresh_assistance_status()
        self._load_latest_rating_for_config()

    # ------------------------------------------------- assistance + rating UI
    def _refresh_assistance_status(self):
        """Show the permanent Stockfish-assistance policy of the selected config.

        Reads the config through the SAME loader the trainer uses, so the
        displayed values are the actual runtime values (spec Section 53).
        """
        try:
            from train.config import load_config
            cfg = load_config(self._selected_config_path())
        except Exception as exc:
            self.stats_vars["stockfish_assistance"].set("config error")
            self._append_log(f"Config load failed: {exc}")
            return
        engine_on = bool(cfg["engine"]["enabled"])
        if engine_on:
            self.stats_vars["stockfish_assistance"].set("ON (permanent)")
        else:
            self.stats_vars["stockfish_assistance"].set("OFF (explicit ablation)")
        self.stats_vars["sf_reward_coef"].set(
            f"{float(cfg['rl']['lambda_stockfish']):.2f}"
        )

    def _load_latest_rating_for_config(self):
        """Restore the latest persisted rating for the selected experiment.

        The rating comes from the evaluator's own history file
        (reports/ratings_<experiment>.jsonl) -- never recomputed here.
        """
        try:
            from train.config import load_config
            cfg = load_config(self._selected_config_path())
        except Exception:
            return
        from evaluation.rating import load_latest_rating
        record = load_latest_rating(str(cfg["project"]["name"]))
        if record is None:
            return
        self._apply_eval_results([], rating_summary=record)
        self._append_log(
            f"Loaded latest rating: iteration {record.get('iteration')} "
            f"({record.get('timestamp', '?')})"
        )

    # ------------------------------------------------------ hardware metrics
    def _start_hw_sampler(self):
        """Background sampler for real hardware metrics (spec Sections 48/52).

        GPU utilization and VRAM come from nvidia-smi; RAM from the Windows
        GlobalMemoryStatusEx API.  Sampled every 2 s into a plain dict that
        the Tk main thread reads once per refresh -- no Tk calls from the
        sampler thread, no fabricated numbers.
        """
        self._hw_metrics: dict[str, str] = {}

        def _ram_total_and_used_gb():
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            total = st.ullTotalPhys / (1024 ** 3)
            used = (st.ullTotalPhys - st.ullAvailPhys) / (1024 ** 3)
            return total, used

        def _loop():
            import subprocess
            while True:
                try:
                    out = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5,
                    ).stdout.strip()
                    util, vram = out.split(",")
                    self._hw_metrics["gpu_util"] = f"{int(util)}%"
                    self._hw_metrics["vram"] = f"{int(vram)} MiB"
                except Exception:
                    self._hw_metrics["gpu_util"] = "N/A"
                    self._hw_metrics["vram"] = "N/A"
                try:
                    total, used = _ram_total_and_used_gb()
                    self._hw_metrics["ram"] = f"{used:.1f} / {total:.1f} GB"
                except Exception:
                    self._hw_metrics["ram"] = "N/A"
                time.sleep(2.0)

        threading.Thread(target=_loop, daemon=True, name="hw-sampler").start()

    def _apply_hw_metrics(self):
        m = getattr(self, "_hw_metrics", None)
        if not m:
            return
        for var_key, metric_key in (
            ("gpu_util", "gpu_util"), ("vram", "vram"), ("ram", "ram"),
        ):
            if metric_key in m and var_key in self.stats_vars:
                self.stats_vars[var_key].set(m[metric_key])
    def _on_train(self):
        cfg_path = self._selected_config_path()
        try:
            hours = float(self.hours_entry.get()) if self.hours_entry.get() else None
        except ValueError:
            hours = None
        if hours is not None and hours <= 0:
            hours = None
        self.train_hours = hours
        self._set_state(State.TRAINING)
        self.stop_event = threading.Event()
        self.worker = TrainingWorker(cfg_path, self, self.stop_event)
        self.worker.start(resume=False)
        self._update_button_states()
        self._append_log(f"Training started: config={Path(cfg_path).name}"
                         + (f" hours={hours}" if hours else " indefinite"))

    def _on_resume(self):
        cfg_path = self._selected_config_path()
        try:
            hours = float(self.hours_entry.get()) if self.hours_entry.get() else None
        except ValueError:
            hours = None
        self.train_hours = hours
        self._set_state(State.TRAINING)
        self.stop_event = threading.Event()
        self.worker = TrainingWorker(cfg_path, self, self.stop_event)
        self.worker.start(resume=True, max_iterations=self.max_iterations)
        self._update_button_states()
        self._append_log(f"Resume requested: config={Path(cfg_path).name}")

    def _on_stop(self):
        if self.worker and self.worker.is_alive():
            self._append_log("Stop requested...")
            self.worker.request_stop()

    def _on_evaluate(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Busy", "Training is running; stop it first.")
            return
        ck = self.checkpoint_combo.get()
        if not ck:
            messagebox.showwarning("No checkpoint", "Select a checkpoint first.")
            return
        cfg_path = self._selected_config_path()
        self._set_state(State.EVALUATING)
        self.stop_event = threading.Event()
        self.worker = TrainingWorker(cfg_path, self, self.stop_event)
        self.worker.start_evaluation(ck)
        self._append_log(f"Evaluating checkpoint: {ck}")

    def _on_load(self):
        ck = self.checkpoint_combo.get()
        if not ck:
            messagebox.showwarning("No checkpoint", "Select a checkpoint first.")
            return
        self._append_log(f"Checkpoint selected: {ck}")

    def _on_play(self):
        ck = self.checkpoint_combo.get()
        if not ck:
            messagebox.showwarning("No checkpoint", "Select a checkpoint first.")
            return
        cfg_path = self._selected_config_path()
        try:
            temp = float(self.play_temp.get())
        except ValueError:
            temp = 0.0
        human_white = self.play_color.get() == "White"
        self._set_state(State.PLAYING)
        self.stop_event = threading.Event()
        self.worker = TrainingWorker(cfg_path, self, self.stop_event)
        self.worker.start_play(ck, human_white=human_white, temperature=temp)
        self._append_log(f"Play started (you are {'White' if human_white else 'Black'}, temp={temp}).")

    def _update_button_states(self):
        running = self.status.state in (State.TRAINING, State.STOPPING, State.EVALUATING, State.PLAYING)
        idle = self.status.state == State.IDLE
        self.btn_train.config(state="disabled" if running else "normal")
        self.btn_resume.config(state="disabled" if running else "normal")
        self.btn_stop.config(state="normal" if running else "disabled")
        self.btn_eval.config(state="normal" if idle else "disabled")

    def _set_state(self, st):
        self.status.state = st
        self.state_label.config(text=st.value,
                                foreground="red" if st == State.ERROR else "black")
        self._update_button_states()

    def _populate_checkpoints(self):
        ck_dir = _REPO / "checkpoints"
        if ck_dir.exists():
            files = sorted(p.name for p in ck_dir.glob("*.pt"))
        else:
            files = []
        self.checkpoint_combo["values"] = files
        if files:
            if self.last_checkpoint in files:
                self.checkpoint_combo.set(self.last_checkpoint)
            else:
                self.checkpoint_combo.set(files[-1])
                self.last_checkpoint = files[-1]
        self.checkpoint_label.config(text=self.checkpoint_combo.get() or "none")

    def _populate_configs(self):
        cfg_dir = _REPO / "configs"
        files = sorted(p.name for p in cfg_dir.glob("*.yaml")) if cfg_dir.exists() else []
        self.config_combo["values"] = files
        if files:
            if self.last_config in files:
                self.config_combo.set(self.last_config)
            else:
                self.config_combo.set(files[0])
        self.config_label.config(text=self.config_combo.get() or "?")

    def _refresh_runtime_info(self):
        self.status.cuda_available, self.status.gpu = gpu_info()
        self.status.device = "cuda" if self.status.cuda_available else "cpu"

    def _append_log(self, msg):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.config(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _apply_stats(self, rec):
        def set_var(key, val, fmt="{:.4f}"):
            if val is not None and key in self.stats_vars:
                self.stats_vars[key].set(fmt.format(float(val)))
        if "iteration" in rec:
            self.stats_vars["iteration"].set(str(rec["iteration"]))
        if "total_games" in rec:
            self.stats_vars["games"].set(str(rec["total_games"]))
        if "total_plies" in rec:
            self.stats_vars["plies"].set(str(rec["total_plies"]))
        el = rec.get("elapsed_hours", 0)
        if el is not None:
            secs = float(el) * 3600
            hh, rem = divmod(int(secs), 3600)
            mm, ss = divmod(rem, 60)
            self.stats_vars["elapsed_time"].set(f"{hh:02d}:{mm:02d}:{ss:02d}")
        set_var("mean_reward", rec.get("mean_reward"))
        set_var("mean_dense_reward", rec.get("mean_stockfish_reward"))
        set_var("mean_delta_s", rec.get("mean_delta_s"))
        set_var("mean_regret", rec.get("mean_regret"))
        set_var("policy_entropy", rec.get("entropy"))
        set_var("ppo_loss", rec.get("ppo_loss"))
        set_var("value_loss", rec.get("value_train_loss"))
        # Throughput (from the evaluator's own iteration record; no guessing).
        wall = rec.get("wall_time_s")
        if wall and float(wall) > 0:
            if "plies" in rec:
                self.stats_vars["self-play_plies/s"].set(
                    f"{float(rec['plies']) / float(wall):.1f}")
            if "engine_evals" in rec:
                self.stats_vars["engine_evals/s"].set(
                    f"{float(rec['engine_evals']) / float(wall):.1f}")

    def _apply_eval_results(self, results, rating_summary=None):
        """Update evaluation statistics from the EVALUATOR's rating summary.

        The GUI performs NO rating math of its own: the Relative Elo, its
        uncertainty, games, and score come from the evaluation subsystem
        (evaluation.rating), either live from the Trainer or from the
        persisted rating history.
        """
        if not results:
            return
        total_w = sum(r["wins"] for r in results)
        total_d = sum(r["draws"] for r in results)
        total_l = sum(r["losses"] for r in results)
        n = total_w + total_d + total_l
        score = (total_w + 0.5 * total_d) / n if n else 0.0
        self.eval_label.config(
            text=f"Evaluation: {total_w}W / {total_d}D / {total_l}L  score={score:.3f}  games={n}"
        )
        self.stats_vars["win_draw_loss"].set(f"{total_w}W / {total_d}D / {total_l}L")
        self.stats_vars["eval_score"].set(f"{score:.3f}")
        self.stats_vars["eval_games"].set(str(n))
        if rating_summary is None:
            # No evaluator summary available -- do NOT fabricate a rating.
            self.stats_vars["relative_elo"].set("N/A")
        elif rating_summary.get("rating") is None:
            self.stats_vars["relative_elo"].set("N/A (no matchup qualified)")
        else:
            txt = (f"{rating_summary['rating']:.0f}"
                   f" ± {rating_summary['rating_uncertainty']:.0f}")
            self.stats_vars["relative_elo"].set(txt)
        self._append_log(
            f"Evaluation complete: {total_w}W / {total_d}D / {total_l}L "
            f"(score {score:.3f}, games {n})"
        )

    def _poll_worker_queue(self):
        self._apply_hw_metrics()
        if self.worker is None:
            self.root.after(self.REFRESH_MS, self._poll_worker_queue)
            return
        try:
            while True:
                event = self.worker.q.get_nowait()
                self._handle_event(event)
        except Exception:
            pass
        self.root.after(self.REFRESH_MS, self._poll_worker_queue)

    def _handle_event(self, event):
        kind = event["kind"]
        payload = event["payload"]
        if kind == "STATUS":
            st_name = payload.get("state", "Idle")
            try:
                st = State(st_name)
            except ValueError:
                st = State.IDLE
            self._set_state(st)
            self._append_log(f"Status: {st_name}")
        elif kind in ("METRIC", "ITERATION_COMPLETE"):
            rec = payload.get("record", {})
            self._apply_stats(rec)
        elif kind == "EVALUATION_COMPLETE":
            self._apply_eval_results(payload.get("results", []),
                                     payload.get("rating"))
        elif kind == "PLAY_READY":
            self._set_state(State.IDLE)
            from gui.play_game import PlayGameWindow
            PlayGameWindow(self.root, payload)
        elif kind == "CHECKPOINT_SAVED":
            it = payload.get("iteration", "?")
            self._append_log(f"Checkpoint saved at iteration {it}")
            self._populate_checkpoints()
        elif kind == "LOG_MESSAGE":
            self._append_log(payload.get("message", ""))
        elif kind == "ERROR":
            msg = payload.get("message", "")
            self._set_state(State.ERROR)
            self._append_log(f"ERROR: {msg}")
            self._append_log(payload.get("traceback", ""))
        elif kind == "TRAINING_FINISHED":
            reason = payload.get("reason", "")
            self._append_log(f"Training finished: {reason}")
            self._set_state(State.IDLE)
            self._update_button_states()

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if messagebox.askyesno("Quit", "Training is running. Stop and exit?"):
                self.worker.request_stop()
                self.root.after(1000, self._safe_destroy)
            else:
                return
        else:
            self._safe_destroy()

    def _safe_destroy(self):
        _save_gui_config({
            "config": self.last_config,
            "checkpoint": self.last_checkpoint,
            "hours": self.train_hours or 0.0,
        })
        self.root.destroy()


def run_gui():
    root = tk.Tk()
    ChessRLApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
