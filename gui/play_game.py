"""Play-against-AI: shared game controller + two view modes.

``PlayGame`` owns the game state (a ``ChessEnv``) and all game logic;
the views are thin:

* ``PlayBoardWindow``  -- graphical mode: a real click-to-move board
  (see ``gui.chessboard``), move list, status bar;
* ``PlayTextWindow``   -- text mode: type ``e4``, the AI replies ``e5``
  in plain text; SAN or UCI input, under-promotions supported (``e8=N``);
* ``play_interactive`` -- plain terminal variant (CLI use).

The main GUI offers a mode selector (graphical board vs text console) and
both windows consume the same controller, so game behavior is identical.
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


class PlayGame:
    """Game-state controller shared by every play view."""

    def __init__(self, human_white: bool = True, max_plies: int = 300,
                 net=None, device=None, temperature: float = 0.0,
                 rng: np.random.Generator | None = None,
                 ai_step=None) -> None:
        """``ai_step`` (optional) replaces the network: a callable that
        receives this controller's ``env``, applies one legal AI move to it,
        and returns ``(move, san)``.  Used by tests and for future
        engine-backed opponents."""
        self.env = ChessEnv()
        self.human_white = bool(human_white)
        self.max_plies = int(max_plies)
        self.net = net
        self.device = device
        self.temperature = float(temperature)
        self.rng = rng or np.random.default_rng()
        self._ai_step = ai_step
        self.moves_san: list[str] = []

    # ------------------------------------------------------------- queries
    def human_turn(self) -> bool:
        return (self.env.turn == chess.WHITE) == self.human_white

    def is_over(self) -> bool:
        return self.env.is_terminal() or self.env.ply >= self.max_plies

    def status(self) -> str:
        b = self.env.board
        side = "White" if b.turn == chess.WHITE else "Black"
        if b.is_checkmate():
            winner = "Black" if b.turn == chess.WHITE else "White"
            return f"Checkmate -- {winner} wins"
        if b.is_stalemate():
            return "Draw -- stalemate"
        if b.is_insufficient_material():
            return "Draw -- insufficient material"
        if b.is_seventyfive_moves() or b.is_fivefold_repetition():
            return "Draw (automatic rule)"
        if b.can_claim_fifty_moves() or b.can_claim_threefold_repetition():
            return "Draw -- claimable repetition/50-move"
        if b.is_check():
            return f"{side} to move -- CHECK"
        if self.env.ply >= self.max_plies:
            return "Draw -- ply limit reached"
        return f"{side} to move"

    def result_for_human(self):
        """+1 human win / 0 draw / -1 loss / None unfinished."""
        if not self.env.is_terminal():
            return None
        from env.chess_env import result_for_player
        return float(result_for_player(self.env.board, self.human_white))

    # -------------------------------------------------------------- moves
    def user_move(self, move: chess.Move) -> tuple[bool, str]:
        """Apply the human's move.  Returns ``(ok, message)``."""
        if self.is_over():
            return False, "The game is already over."
        if not self.human_turn():
            return False, "It is the AI's turn."
        if move not in self.env.board.legal_moves:
            return False, "Illegal move."
        san = self.env.board.san(move)
        self.env.step_direct(move)
        self.moves_san.append(san)
        return True, san

    def ai_reply(self):
        """Apply the AI's move; returns ``(move, san)`` or None if it is not
        the AI's turn / the game is over."""
        if self.is_over() or self.human_turn():
            return None
        if self._ai_step is not None:
            move, san = self._ai_step(self.env)
            self.moves_san.append(san)
            return move, san
        action = get_net_move(self.env, self.net, self.device,
                              self.temperature, self.rng)
        move = self.env.canonical_action_to_move(action)
        san = self.env.board.san(move)
        self.env.step(action)
        self.moves_san.append(san)
        return move, san


class _PlayWindowBase(tk.Toplevel):
    """Shared plumbing for both play windows."""

    def _make_game(self, payload) -> PlayGame:
        cfg = payload.get("cfg", {}) or {}
        max_plies = cfg.get("selfplay", {}).get("max_plies", 300)
        return PlayGame(
            human_white=payload.get("human_white", True),
            max_plies=max_plies,
            net=payload["net"],
            device=payload["device"],
            temperature=payload.get("temperature", 0.0),
        )

    def _finish_log(self, log_fn) -> None:
        result = self.game.result_for_human()
        text = self.game.status()
        if result is not None:
            verdict = ("You win" if result == 1
                       else "You lose" if result == -1 else "Draw")
            text += f"  ({verdict})"
        log_fn(text)


class PlayBoardWindow(_PlayWindowBase):
    """Graphical play: real click-to-move board + move list + status."""

    def __init__(self, root, payload):
        from gui.chessboard import ChessBoardWidget

        self.net = payload["net"]
        self.game = self._make_game(payload)
        self.stop_event = threading.Event()

        self.root = tk.Toplevel(root)
        self.root.title(
            "Play vs AI -- " + ("You (White)" if self.game.human_white
                                else "You (Black)")
        )
        self.root.geometry("960x680")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        self.board_widget = ChessBoardWidget(
            main, cell=64, margin=26,
            white_bottom=self.game.human_white,
            on_move=self._on_board_move,
        )
        self.board_widget.pack(side="left", fill="both", expand=True)

        side = ttk.Frame(main, padding=(10, 0, 0, 0))
        side.pack(side="left", fill="y")
        self.status_var = tk.StringVar(value=self.game.status())
        ttk.Label(side, textvariable=self.status_var,
                  font=("", 11, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Moves").pack(anchor="w")
        self.move_text = tk.Text(side, width=18, height=26, state="disabled",
                                 font=("Consolas", 10))
        self.move_text.pack(fill="y", expand=True)
        btns = ttk.Frame(side)
        btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="New game",
                   command=self._new_game).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Flip board",
                   command=lambda: self.board_widget.flip()).pack(side="left")
        ttk.Button(btns, text="Close",
                   command=self._on_close).pack(side="left", padx=(4, 0))

        self._refresh()
        if not self.game.human_turn():
            self.root.after(120, self._ai_move)

    def _refresh(self):
        last = self.game.env.board.peek() if self.game.env.ply else None
        self.board_widget.set_position(self.game.env.board, last)
        self.status_var.set(self.game.status())
        self.move_text.config(state="normal")
        self.move_text.delete("1.0", "end")
        sans = self.game.moves_san
        for i in range(0, len(sans), 2):
            line = f"{i // 2 + 1}. {sans[i]}"
            if i + 1 < len(sans):
                line += f"  {sans[i + 1]}"
            self.move_text.insert("end", line + "\n")
        self.move_text.see("end")
        self.move_text.config(state="disabled")

    def _on_board_move(self, move: chess.Move) -> None:
        ok, msg = self.game.user_move(move)
        if not ok:
            self.status_var.set(msg)
            return
        self._refresh()
        if not self.game.is_over():
            self.root.after(80, self._ai_move)

    def _ai_move(self):
        if self.game.is_over() or self.stop_event.is_set():
            self._refresh()
            return
        self.status_var.set("AI is thinking...")
        self.root.update_idletasks()
        self.game.ai_reply()
        self._refresh()

    def _new_game(self):
        payload = {
            "net": self.net, "device": self.game.device,
            "cfg": {"selfplay": {"max_plies": self.game.max_plies}},
            "human_white": self.game.human_white,
            "temperature": self.game.temperature,
        }
        root = self.root.master
        self.root.destroy()
        PlayBoardWindow(root, payload)

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


class PlayTextWindow(_PlayWindowBase):
    """Text-only play: type ``e4``, the AI replies ``e5`` in plain text."""

    def __init__(self, root, payload):
        self.game = self._make_game(payload)
        self.stop_event = threading.Event()

        self.root = tk.Toplevel(root)
        self.root.title("Play vs AI (text) -- "
                        + ("You (White)" if self.game.human_white
                           else "You (Black)"))
        self.root.geometry("560x500")

        frame = ttk.Frame(self.root, padding=8)
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, wrap="word", state="disabled",
                           font=("Consolas", 11))
        self.log.pack(fill="both", expand=True)
        entry_row = ttk.Frame(frame)
        entry_row.pack(fill="x", pady=(6, 0))
        ttk.Label(entry_row, text="Move:").pack(side="left")
        self.entry = ttk.Entry(entry_row)
        self.entry.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.entry.bind("<Return>", self._on_enter)
        ttk.Button(entry_row, text="Close",
                   command=self._on_close).pack(side="left")
        self.entry.focus()

        self._log_line("Text mode: enter moves as SAN (e4, Nf3, O-O, e8=N) "
                       "or UCI (e2e4).")
        self._log_line(self.game.status())
        if not self.game.human_turn():
            self.root.after(120, self._ai_move)

    def _log_line(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _on_enter(self, event=None):
        text = self.entry.get().strip()
        self.entry.delete(0, "end")
        if not text:
            return
        if text.lower() in ("quit", "exit", "resign"):
            self._log_line("Game abandoned.")
            self._on_close()
            return
        if self.game.is_over():
            self._log_line(self.game.status())
            return
        move = parse_user_move(self.game.env, text)
        if move is None:
            self._log_line(f"? cannot parse / illegal: {text}")
            return
        ok, msg = self.game.user_move(move)
        if not ok:
            self._log_line(f"? {msg}")
            return
        self._log_line(f"You: {msg}")
        if self.game.is_over():
            self._finish_log(self._log_line)
            return
        self._log_line("AI is thinking...")
        self.root.update_idletasks()
        reply = self.game.ai_reply()
        if reply is None:
            self._log_line(self.game.status())
            return
        self._log_line(f"AI: {reply[1]}")
        if self.game.is_over():
            self._finish_log(self._log_line)
        else:
            self._log_line(self.game.status())

    def _ai_move(self):
        if self.game.is_over() or self.stop_event.is_set():
            return
        reply = self.game.ai_reply()
        if reply is None:
            self._log_line(self.game.status())
            return
        self._log_line(f"AI: {reply[1]}")
        if self.game.is_over():
            self._finish_log(self._log_line)
        else:
            self._log_line(self.game.status())

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


# Backward-compatible name used by earlier callers/tests.
PlayGameWindow = PlayBoardWindow


def play_interactive(net, cfg, device, human_white=True, temperature=0.0,
                     log_fn=print, stop_event=None, max_plies=300):
    """Plain-terminal play (CLI).  Uses the same controller as the GUI."""
    game = PlayGame(human_white=human_white, max_plies=max_plies, net=net,
                    device=device, temperature=temperature)
    log_fn("Game started. You are " + ("White" if human_white else "Black") + ".")
    while not game.is_over():
        if stop_event and stop_event.is_set():
            log_fn("Play aborted.")
            return None
        if game.human_turn():
            log_fn(board_display(game.env.board))
            move = None
            while move is None:
                if stop_event and stop_event.is_set():
                    return None
                text = input("> ")
                move = parse_user_move(game.env, text)
                if move is None:
                    log_fn("Illegal/unparseable move, try again.")
            ok, msg = game.user_move(move)
            if ok:
                log_fn("You: " + msg)
        else:
            _, san = game.ai_reply()
            log_fn("AI: " + san)
    result = game.result_for_human()
    log_fn("Result: " + game.status())
    return result

