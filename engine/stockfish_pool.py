"""Persistent Stockfish worker pool (spec Sections 46-48).

Architecture::

    request queue -> worker processes -> Stockfish instances -> structured results

Each worker process owns exactly one long-lived Stockfish process (never
one-per-move).  Workers perform the full startup protocol once (uci ->
identity -> options -> isready) and then serve analysis requests until
shutdown.  A dead worker restarts transparently on its next job; a request
that fails twice raises (Rule 4: invalid engine responses are hard errors).

The public API is thread-safe (used from concurrent self-play games).
Results cross process boundaries as plain dicts (picklable); use
:func:`engine_info_from_worker` to rebuild an ``EngineInfo``.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import threading
import time
import uuid

import chess

from engine.evaluation import EngineConfig, EngineInfo

# --------------------------------------------------------------------- worker


def _worker_main(worker_id: int, cfg: dict, request_q, response_q) -> None:
    """Worker loop: owns one Stockfish process, serves analysis requests."""
    import chess.engine

    from engine.evaluation import engine_info_from_python_chess

    engine = None
    while True:
        try:
            job = request_q.get()
        except (KeyboardInterrupt, EOFError):
            break
        if job is None:
            break
        job_id, kind, payload = job
        try:
            if kind == "restart":
                if engine is not None:
                    engine.quit()
                    engine = None
                response_q.put((job_id, "ok", None))
                continue
            if engine is None:
                eng = chess.engine.SimpleEngine.popen_uci(cfg["path"], debug=False)
                eng.configure({"Threads": cfg["threads"], "Hash": cfg["hash_mb"]})
                if cfg.get("wdl", True):
                    eng.configure({"UCI_ShowWDL": "true"})
                eng.ping()
                engine = eng
            if kind == "analyse":
                board = chess.Board(payload["fen"])
                info = engine.analyse(
                    board, chess.engine.Limit(depth=payload["depth"])
                )
                ei = engine_info_from_python_chess(info)
                response_q.put(
                    (
                        job_id,
                        "ok",
                        {
                            "best_move_uci": ei.best_move.uci() if ei.best_move else None,
                            "depth": ei.depth,
                            "seldepth": ei.seldepth,
                            "nodes": ei.nodes,
                            "nps": ei.nps,
                            "time_ms": ei.time_ms,
                            "hashfull": ei.hashfull,
                            "cp": ei.cp,
                            "mate_n": ei.mate_n,
                            "wdl": ei.wdl,
                        },
                    )
                )
            elif kind == "ping":
                response_q.put((job_id, "ok", None))
            else:
                response_q.put((job_id, "error", f"unknown job kind {kind}"))
        except Exception as exc:  # engine protocol failure -> report, reset
            response_q.put((job_id, "error", f"{type(exc).__name__}: {exc}"))
            try:
                if engine is not None:
                    engine.close()
            except Exception:
                pass
            engine = None
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass


def engine_info_from_worker(payload: dict) -> EngineInfo:
    """Rebuild an :class:`EngineInfo` from a worker result dict."""
    return EngineInfo(
        best_move=(
            chess.Move.from_uci(payload["best_move_uci"])
            if payload["best_move_uci"]
            else None
        ),
        depth=int(payload["depth"]),
        seldepth=int(payload["seldepth"]),
        nodes=int(payload["nodes"]),
        nps=int(payload["nps"]),
        time_ms=int(payload["time_ms"]),
        hashfull=int(payload["hashfull"]),
        cp=payload["cp"],
        mate_n=payload["mate_n"],
        wdl=payload["wdl"],
    )


# ----------------------------------------------------------------------- pool


class StockfishPool:
    """Thread-safe pool of persistent Stockfish worker processes."""

    def __init__(self, engine_cfg: dict) -> None:
        self.cfg = engine_cfg
        self.workers = max(1, int(engine_cfg.get("workers", 1)))
        self._request_q: mp.Queue = mp.get_context("spawn").Queue()
        self._response_q: mp.Queue = mp.get_context("spawn").Queue()
        self._processes: list[mp.Process] = []
        self._futures: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self._closed = False
        for wid in range(self.workers):
            self._spawn(wid)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._wait_ready()

    # ------------------------------------------------------------ lifecycle
    def _spawn(self, wid: int) -> None:
        p = mp.Process(
            target=_worker_main,
            args=(wid, self.cfg, self._request_q, self._response_q),
            daemon=True,
        )
        p.start()
        self._processes.append(p)

    def _wait_ready(self, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        for _ in range(self.workers):
            fut = self._submit("ping", None)
            remaining = max(1.0, deadline - time.time())
            if not fut["event"].wait(remaining) or fut["status"] != "ok":
                raise RuntimeError("StockfishPool workers failed to start")

    def identity(self) -> str:
        """Cheap pool descriptor (workers/depth); full version in metadata."""
        return f"workers={self.workers} depth={self.cfg.get('depth')}"

    # ------------------------------------------------------------ requests
    def _submit(self, kind: str, payload: dict | None) -> dict:
        job_id = uuid.uuid4().hex
        fut = {"event": threading.Event(), "status": None, "result": None}
        with self._lock:
            self._futures[job_id] = fut
        self._request_q.put((job_id, kind, payload))
        return fut

    def _read_loop(self) -> None:
        while True:
            try:
                job_id, status, result = self._response_q.get(timeout=0.5)
            except queue.Empty:
                if self._closed:
                    return
                continue
            with self._lock:
                fut = self._futures.pop(job_id, None)
            if fut is not None:
                fut["status"] = status
                fut["result"] = result
                fut["event"].set()

    def analyse(self, fen: str, depth: int | None = None) -> EngineInfo:
        """Blocking root analysis on any pool worker (search A / search B)."""
        if self._closed:
            raise RuntimeError("StockfishPool is closed")
        depth = int(depth if depth is not None else self.cfg["depth"])
        payload = {"fen": fen, "depth": depth}
        fut = self._submit("analyse", payload)
        if not fut["event"].wait(timeout=180.0):
            raise RuntimeError("engine request timed out")
        if fut["status"] == "error":
            # One retry (the worker resets its engine on failure); a second
            # failure is a protocol failure without recovery -> hard error.
            fut = self._submit("analyse", payload)
            if not fut["event"].wait(timeout=180.0):
                raise RuntimeError("engine request timed out after retry")
            if fut["status"] == "error":
                raise RuntimeError(f"engine failure without recovery: {fut['result']}")
        return engine_info_from_worker(fut["result"])

    # ------------------------------------------------------------ shutdown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for _ in self._processes:
            try:
                self._request_q.put(None, timeout=2)
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self._processes.clear()

    def __enter__(self) -> "StockfishPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def size(self) -> int:
        return self.workers


# ------------------------------------------------------------------ helpers


def default_engine_config(engine_cfg: dict) -> EngineConfig:
    """Map the YAML engine section onto :class:`EngineConfig`."""
    return EngineConfig(
        path=engine_cfg["path"],
        depth=int(engine_cfg["depth"]),
        threads=int(engine_cfg["threads"]),
        hash_mb=int(engine_cfg["hash_mb"]),
    )


def binary_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_version_line(path: str) -> str:
    """Best-effort engine identity via a throwaway UCI handshake."""
    import subprocess

    proc = subprocess.Popen(
        [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
    )
    try:
        proc.stdin.write("uci\nquit\n")
        proc.stdin.flush()
        for line in proc.stdout:
            if line.startswith("id name"):
                return line.strip()
    finally:
        try:
            proc.kill()
        except Exception:
            pass
    return "unknown"

