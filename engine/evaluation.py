"""Structured parsing of Stockfish's UCI evaluation output.

``EngineInfo`` is a transfer object that stores every quantity needed by the
reward module, split into semantically different fields (centipawn, mate,
WDL) - never merged into one opaque number.

The conversion to a centered score ``S`` in [-1, +1] prefers the WDL-derived
expected-score signal and treats forced mates as +/-1 deterministically (see
:func:`EngineInfo.centered_score_white`).  No universal centipawn -> win
probability formula is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.engine


class EngineWdlMissingError(RuntimeError):
    """Raised when a non-terminal position's analysis lacks a WDL signal."""


@dataclass
class EngineConfig:
    path: str
    depth: int = 16
    threads: int = 1
    hash_mb: int = 256
    wdl_enabled: bool = True
    extra_options: dict = field(default_factory=dict)


@dataclass
class EngineInfo:
    """Structured evaluation of a single position (root search)."""

    best_move: Optional[chess.Move]
    depth: int
    seldepth: int
    nodes: int
    nps: int
    time_ms: int
    hashfull: int
    # White-perspective raw values:
    cp: Optional[int] = None          # centipawn score (None when mate)
    mate_n: Optional[int] = None      # signed mate value (None when cp)
    wdl: Optional[tuple[int, int, int]] = None  # (wins, draws, losses) White POV
    string: Optional[str] = None

    # ------------------------------------------------------------- predicates
    @property
    def is_mate(self) -> bool:
        return self.mate_n is not None

    @property
    def has_wdl(self) -> bool:
        return self.wdl is not None

    def expected_score_white(self) -> float:
        """Expected score E in [0, 1] from White's perspective."""
        return (self.centered_score_white() + 1.0) / 2.0

    # ------------------------------------------------------------- conversion
    def centered_score_white(self) -> float:
        """Centered expected-score from White's perspective, in [-1, +1].

        Ordered preference:
          1. forced mate  -> sign of the mate value (+1 / -1);
          2. WDL statistics of a non-terminal position -> 2*(p_w + p_d/2) - 1;
          3. otherwise raise (never invent a universal cp->win calibration).
        """
        if self.is_mate:
            if self.mate_n == 0:
                raise ValueError(
                    "invalid engine response: mate 0 is not a legal UCI score"
                )
            return 1.0 if self.mate_n > 0 else -1.0
        if self.has_wdl:
            from engine.wdl import centered_from_wdl

            return centered_from_wdl(*self.wdl)
        raise EngineWdlMissingError(
            f"no WDL and no mate signal in engine output at depth {self.depth} "
            f"(cp={self.cp}, mate={self.mate_n})"
        )


def engine_info_from_python_chess(info: dict) -> EngineInfo:
    """Convert a python-chess analysis ``info`` dict into ``EngineInfo``."""
    score = info.get("score")
    cp: Optional[int] = None
    mate_n: Optional[int] = None
    if score is not None:
        white_score = score.white()
        if white_score.is_mate():
            # python-chess Mate stores the signed value in `.moves`
            # (`Mate.mate()` is an accessor method; MateGiven has moves == 0,
            # which we treat as an invalid engine response below).
            mate_n = int(white_score.moves)
        else:
            cp = white_score.cp

    wdl = None
    if "wdl" in info:
        w = info["wdl"].white()
        wdl = (int(w.wins), int(w.draws), int(w.losses))

    pv = info.get("pv")
    best = pv[0] if pv else None

    return EngineInfo(
        best_move=best,
        depth=int(info.get("depth", 0)),
        seldepth=int(info.get("seldepth", 0)),
        nodes=int(info.get("nodes", 0)),
        nps=int(info.get("nps", 0)),
        time_ms=int(info.get("time", 0)),
        hashfull=int(info.get("hashfull", 0)),
        cp=cp,
        mate_n=mate_n,
        wdl=wdl,
        string=info.get("string"),
    )


class StockfishEngine:
    """Persistent, single-process Stockfish engine (UCI) via python-chess."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._engine: Optional[chess.engine.SimpleEngine] = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Launch, handshake (uci), configure options, and await readiness."""
        if self.config.wdl_enabled and "UCI_ShowWDL" not in self.config.extra_options:
            self.config.extra_options["UCI_ShowWDL"] = "true"
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self.config.path, debug=False
        )
        self._configure()

    def _configure(self) -> None:
        assert self._engine is not None
        options = {"Threads": self.config.threads, "Hash": self.config.hash_mb}
        options.update(self.config.extra_options)
        for key, value in options.items():
            self._engine.configure({key: value})
        # Force protocol sync; nothing can be analyzed before readiness.
        self._engine.ping()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "StockfishEngine":
        if self._engine is None:
            self.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: D105
        self.close()

    # ------------------------------------------------------------- analysis
    def analyse(self, board: chess.Board, depth: Optional[int] = None) -> EngineInfo:
        """Root analysis (search A).  ``board`` is non-terminal."""
        assert self._engine is not None, "engine not started"
        d = self.config.depth if depth is None else depth
        info = self._engine.analyse(board, chess.engine.Limit(depth=d))
        return engine_info_from_python_chess(info)

    def analyse_child(
        self, board: chess.Board, move: chess.Move, depth: Optional[int] = None
    ) -> EngineInfo:
        """Analysis of the position resulting from ``move`` (search B)."""
        child = board.copy()
        child.push(move)
        return self.analyse(child, depth=depth)


def centered_score_stm(info: EngineInfo, board: chess.Board) -> float:
    """Centered score from the side-to-move perspective of ``board``."""
    s_white = info.centered_score_white()
    return s_white if board.turn == chess.WHITE else -s_white


def mover_centered_score(child_info: EngineInfo, child_board: chess.Board) -> float:
    """Score of a position from the perspective of the *previous* mover.

    The position ``child_board`` is the result of the previous mover's move;
    the side to move there is the opponent.  Per the project convention the
    previous mover's relative value is ``-S_stm(child)``.
    """
    return -centered_score_stm(child_info, child_board)