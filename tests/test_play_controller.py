"""PlayGame controller tests (torch-free, stubbed AI).

The controller owns all game logic shared by the board and text views:
turn order, legality, AI replies, and terminal detection.
"""

from __future__ import annotations

import chess
import pytest

from gui.play_game import PlayGame


def _stub_ai(moves_uci: list[str]):
    """AI plays scripted original-board UCI moves in order."""
    queue = list(moves_uci)

    def ai_step(env) -> tuple[chess.Move, str]:
        uci = queue.pop(0)
        move = chess.Move.from_uci(uci)
        assert move in env.board.legal_moves, f"{uci} not legal"
        san = env.board.san(move)
        env.step_direct(move)
        return move, san

    return ai_step


def test_human_move_and_ai_reply_alternate():
    game = PlayGame(human_white=True, ai_step=_stub_ai(["e7e5"]))
    ok, msg = game.user_move(chess.Move.from_uci("e2e4"))
    assert ok and msg == "e4"
    reply = game.ai_reply()
    assert reply is not None
    move, san = reply
    assert san == "e5" and move == chess.Move.from_uci("e7e5")
    assert game.env.ply == 2


def test_human_move_rejected_out_of_turn():
    game = PlayGame(human_white=True, ai_step=_stub_ai(["e7e5"]))
    game.user_move(chess.Move.from_uci("e2e4"))
    game.ai_reply()
    # Now it is White's (human's) turn again; the AI cannot move via user path.
    ok, msg = game.user_move(chess.Move.from_uci("e7e5"))
    assert not ok  # e7e5 is not a White move anyway
    assert "Illegal" in msg


def test_illegal_user_move_rejected():
    game = PlayGame(human_white=True, ai_step=_stub_ai([]))
    ok, msg = game.user_move(chess.Move.from_uci("e2e5"))
    assert not ok and "Illegal" in msg
    assert game.env.ply == 0


def test_ai_reply_none_on_human_turn():
    game = PlayGame(human_white=True, ai_step=_stub_ai([]))
    assert game.ai_reply() is None  # human to move: AI must not act
    assert game.env.ply == 0


def test_checkmate_detected_and_result():
    # Fool's mate: human plays White and loses.
    game = PlayGame(human_white=True, ai_step=_stub_ai(["e7e5", "d8h4"]))
    game.user_move(chess.Move.from_uci("f2f3"))
    game.ai_reply()
    game.user_move(chess.Move.from_uci("g2g4"))
    game.ai_reply()
    assert game.is_over()
    assert game.env.board.is_checkmate()
    assert game.result_for_human() == -1.0
    assert "Checkmate" in game.status() and "Black" in game.status()


def test_human_as_black_plays_second():
    game = PlayGame(human_white=False,
                    ai_step=_stub_ai(["e2e4"]))  # AI (White) opens with e4
    assert not game.human_turn()  # AI (White) moves first
    # Simulate the AI's opening move through the same path the GUI uses.
    move, san = game.ai_reply()
    assert san == "e4"
    assert move == chess.Move.from_uci("e2e4")
    assert game.human_turn()