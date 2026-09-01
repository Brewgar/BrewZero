"""Play a game against the loaded model using the existing policy pipeline.

Provides:
- PlayGameWindow: a small Tk Toplevel with a board display and move entry.
- play_interactive: a terminal-based variant (retained for CLI use).
"""

from __future__ import annotations

import threading

import chess
import numpy as np
import tkinter as tk
from tkinter import ttk

from env.chess_env import ChessEnv
from env.action_space import ActionCodec
from env.encoding import encode_state
from selfplay.selfplay import _infer
from selfplay.sampling import sample_action


def get_net_move(env, net, device, temperature, rng):
    canon = env.canonical_board()
    state = encode_state(canon)
    legal = ActionCodec.encode_legal_moves(canon)
    logits, _, _, _ = _infer(net, state, device)
    return sample_action(logits, legal, temperature, rng)


def parse_user_move(env, text):
    text = text.strip()
    if not text:
        return None
    try:
        return env.board.parse_san(text)
    except ValueError:
        try:
            m = chess.Move.from_uci(text)
            if m in env.board.legal_moves:
                return m
        except ValueError:
            pass
    return None


def board_display(board):
    return board.unicode(borders=True, empty_square=".")


def play_interactive(net, cfg, device, human_white=True, temperature=0.0,
                     log_fn=print, stop_event=None, max_plies=300):
    env = ChessEnv()
    log_fn("Game started. You are " + ("White" if human_white else "Black") + ".")
    while True:
        if stop_event and stop_event.is_set():
            log_fn("Play aborted.")
            return None
        if env.is_terminal():
            break
        if env.ply >= max_plies:
            log_fn("Truncated at max plies (draw for scoring).")
            return 0.0
        net_turn = (env.turn == chess.WHITE) != human_white
        if net_turn:
            action = get_net_move(env, net, device, temperature, np.random.default_rng(0))
            env.step(action)
            log_fn("AI: " + env.board.peek().uci())
        else:
            log_fn(board_display(env.board))
            log_fn("Your move (" + ("White" if env.turn == chess.WHITE else "Black") + "): ")
            move = None
            while move is None:
                if stop_event and stop_event.is_set():
                    return None
                text = input("> ")
                move = parse_user_move(env, text)
                if move is None:
                    log_fn("Illegal/unparseable move, try again.")
            env.step_direct(move)
    if env.board.is_checkmate():
        loser = "White" if env.board.turn == chess.WHITE else "Black"
        log_fn(f"Checkmate! {loser} loses.")
        z = -1.0 if env.board.turn == chess.WHITE else 1.0
    else:
        log_fn("Draw.")
        z = 0.0
    return z if human_white else -z


class PlayGameWindow:
    """Minimal graphical board for playing against the loaded model.

    Uses a Text widget for board display (unicode) and an Entry for move
    input.  Runs entirely in the GUI thread (no Tk threading concerns).
    """

    def __init__(self, root, payload):
        self.net = payload["net"]
        self.device = payload["device"]
        self.cfg = payload.get("cfg", {})
        self.human_white = payload.get("human_white", True)
        self.temperature = payload.get("temperature", 0.0)
        self.stop_event = threading.Event()

        self.env = ChessEnv()
        self.root = tk.Toplevel(root)
        self.root.title("Play vs AI - " + ("You (White)" if self.human_white else "You (Black)"))
        self.root.geometry("480x580")

        self.board_text = tk.Text(self.root, width=24, height=12,
                                  font=("Consolas", 14), state="disabled",
                                  bg="#2b2b2b", fg="#f0f0f0")
        self.board_text.pack(padx=8, pady=8)

        self.move_entry = ttk.Entry(self.root, width=20)
        self.move_entry.pack(pady=4)
        self.move_entry.bind("<Return>", self._on_enter_move)
        self.move_entry.focus()

        btn_row = ttk.Frame(self.root)
        ttk.Button(btn_row, text="Resign", command=self._on_resign).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", command=self._on_close).pack(side="left", padx=4)
        btn_row.pack(pady=4)

        self._update_board()
        self._append_msg("You are " + ("White" if self.human_white else "Black") + ".")

        if not self.human_white:
            self.root.after(100, self._ai_move)

    def _update_board(self):
        display = board_display(self.env.board)
        self.board_text.config(state="normal")
        self.board_text.delete("1.0", "end")
        self.board_text.insert("end", display)
        self.board_text.config(state="disabled")

    def _append_msg(self, msg):
        self.board_text.config(state="normal")
        self.board_text.insert("end", f"\n{msg}\n")
        self.board_text.config(state="disabled")
        self.board_text.see("end")

    def _on_enter_move(self, event):
        text = self.move_entry.get().strip()
        self.move_entry.delete(0, "end")
        if not text:
            return
        move = parse_user_move(self.env, text)
        if move is None:
            self._append_msg("Illegal/unparseable, try again.")
            return
        self.env.step_direct(move)
        self._update_board()
        max_plies = self.cfg.get("selfplay", {}).get("max_plies", 300)
        if not self.env.is_terminal() and self.env.ply < max_plies:
            self.root.after(50, self._ai_move)
        else:
            self._game_over()

    def _ai_move(self):
        if self.env.is_terminal() or self.stop_event.is_set():
            self._game_over()
            return
        rng = np.random.default_rng(0)
        action = get_net_move(self.env, self.net, self.device, self.temperature, rng)
        self.env.step(action)
        self._update_board()
        max_plies = self.cfg.get("selfplay", {}).get("max_plies", 300)
        if self.env.is_terminal() or self.env.ply >= max_plies:
            self._game_over()

    def _game_over(self):
        if self.env.board.is_checkmate():
            loser = "White" if self.env.board.turn == chess.WHITE else "Black"
            self._append_msg(f"Checkmate! {loser} loses.")
        elif self.env.board.is_stalemate():
            self._append_msg("Stalemate. Draw.")
        elif self.env.board.is_insufficient_material():
            self._append_msg("Insufficient material. Draw.")
        else:
            self._append_msg("Draw (truncated or ended).")
        self.move_entry.config(state="disabled")

    def _on_resign(self):
        self._append_msg("You resigned. AI wins.")
        self.move_entry.config(state="disabled")

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()
