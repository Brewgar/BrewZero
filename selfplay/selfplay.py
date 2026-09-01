"""Self-play game driver (spec Section 44).

Per move:
  1. canonicalize -> encode -> policy logits + value heads (frozen net, eval)
  2. mask illegal actions, temperature-sample, store old log-prob
  3. engine search A on the PRE-move position -> S_t (stm perspective)
  4. execute the move; terminal / truncation checks

Dense-reward deferral: the mover-relative child score ``S_{t+1}^{mover} =
-S_stm(s_{t+1})`` is available from the NEXT step's root analysis, so each
position is analyzed exactly once (spec Section 21 forbids redundant
searches).  Step ``t``'s reward is finalized when step ``t+1``'s score
arrives, or immediately at a terminal (z in {-1,0,+1}) or truncated (one
extra child search) boundary.

    Canonicalization caveat (CRITICAL, regression-tested): the engine analyses the
    *original* board (``env.board.fen()``), so its White-perspective score refers to
    the ORIGINAL White -- NOT to the canonical relabeled side to move.  The score
    must therefore be converted with the ANALYZED board (``env.board``), never with
    the canonical board (whose ``turn`` is always WHITE and would silently defeat
    the side-to-move conversion at every Black-to-move ply).

Perspective (Sections 18-20):
    S_t            -- engine centered score, stm perspective at s_t
    S_{t+1}^{mover} = -S_stm(s_{t+1})
    Delta S_t       = S_{t+1}^{mover} - S_t
    G_t (regret)    = S_t - S_{t+1}^{mover} = -Delta S_t

Regret semantics (documented design decision): G_t is *prospective* -- it
measures how much the mover's chosen move changed the engine's assessment of
their own position relative to the pre-move assessment.  The engine's root
best move is NOT analyzed, so this is NOT the counterfactual regret
``S_best - S_actual`` (that would require one extra child search per move).
Consequently ``regret_reward = +coef * tanh(Delta S / tau)`` is monotonically
increasing in Delta S and partially collinear with ``delta_reward``; it acts
as a saturating (tanh) duplicate of the position-change signal, not as a
deviation-from-best-play penalty.  This is the intended reward for
Experiments B/C/D.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time

import numpy as np
import torch

from env.action_space import ActionCodec
from env.chess_env import ChessEnv, result_for_player
from env.encoding import encode_state
from engine.evaluation import centered_score_stm
from engine.reward import compute_reward_components
from selfplay.sampling import (
    log_prob_of,
    sample_action_from_probs,
    temperature_probs,
)
from selfplay.trajectory import Step, Trajectory


class _BatchedInferer:
    """Microbatching inference server shared by concurrent game threads.

    Game threads submit encoded states; a single server thread collects
    requests until ``batch_size`` are pending or ``max_wait_ms`` has elapsed
    since collection started, then runs ONE forward pass and delivers
    per-request results.  Each game receives exactly its own logits and
    value outputs -- the policy distribution per state is unchanged (same
    frozen net in eval mode); only the GEMM batch size differs, which can
    change results in the last float bits (no semantic change).

    This removes the per-state GPU round trip from the game threads' critical
    path and raises GPU utilization by amortizing kernel launches across
    concurrent games (spec Sections 29-30).
    """

    def __init__(self, net, device: str, batch_size: int, max_wait_ms: float) -> None:
        self.net = net
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_wait_s = max(0.0, float(max_wait_ms)) / 1000.0
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="batched-inferer")
        self._thread.start()

    def infer(self, state: np.ndarray):
        entry = {"state": state, "event": threading.Event(), "out": None}
        with self._lock:
            self._pending.append(entry)
        entry["event"].wait()
        out = entry["out"]
        if isinstance(out, BaseException):
            raise out
        return out

    def _serve(self) -> None:
        while True:
            with self._lock:
                batch = self._pending[: self.batch_size]
                overflow = self._pending[self.batch_size :]
                self._pending = overflow
            if not batch:
                if self._closed:
                    return
                time.sleep(0.0005)
                continue
            if self.max_wait_s > 0.0 and len(batch) < self.batch_size:
                # Brief collection window so concurrent games can fill the batch.
                time.sleep(self.max_wait_s)
                with self._lock:
                    more = self._pending[: self.batch_size - len(batch)]
                    self._pending = self._pending[len(more):]
                    batch = batch + more
            try:
                x = torch.from_numpy(
                    np.stack([e["state"] for e in batch])
                ).to(self.device)
                with torch.inference_mode():
                    out = self.net(x)
                logits = out["policy_logits"].float().cpu().numpy()
                v_game = out["v_game"].float().cpu().numpy()
                v_train = out["v_train"].float().cpu().numpy()
                v_sf = (
                    out["v_sf"].float().cpu().numpy() if "v_sf" in out else None
                )
                for i, e in enumerate(batch):
                    e["out"] = (
                        logits[i],
                        float(v_game[i]),
                        float(v_train[i]),
                        float(v_sf[i]) if v_sf is not None else None,
                    )
                    e["event"].set()
            except BaseException as exc:  # deliver the failure to every waiter
                for e in batch:
                    e["out"] = exc
                    e["event"].set()

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=5.0)


def _infer(net, state: np.ndarray, device: str):
    """Single-state frozen-policy inference.

    Returns (logits, v_game, v_train, v_sf or None)."""
    x = torch.from_numpy(state).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = net(x)
    logits = out["policy_logits"][0].float().cpu().numpy()
    v_game = float(out["v_game"][0])
    v_train = float(out["v_train"][0])
    v_sf = float(out["v_sf"][0]) if "v_sf" in out else None
    return logits, v_game, v_train, v_sf


def _finalize(step: Step, s_t: float | None, s_next_mover: float, rl: dict) -> None:
    """Complete step ``t``'s dense reward once the child score is known."""
    if s_t is None:  # engine disabled (Experiment A)
        step.reward = {
            "delta_score": 0.0, "engine_regret": 0.0, "delta_reward": 0.0,
            "regret_reward": 0.0, "stockfish_dense_reward": 0.0,
            "terminal_game_reward": 0.0,
            # lambda_game * z with z = 0 mid-game (engine disabled).
            "total_training_reward": 0.0,
        }
        return
    comp = compute_reward_components(
        delta_score=s_next_mover - s_t,
        engine_regret=s_t - s_next_mover,
        terminal_z_mover=None,   # game component is added by the caller
        delta_coef=rl["delta_coef"],
        regret_coef=rl["regret_coef"],
        regret_tau=rl["regret_tau"],
        rmax=rl["r_max"],
        lambda_stockfish=rl["lambda_stockfish"],
        lambda_game=rl["lambda_game"],
    )
    step.reward = {
        "delta_score": comp.delta_score,
        "engine_regret": comp.engine_regret,
        "delta_reward": comp.delta_reward,
        "regret_reward": comp.regret_reward,
        "stockfish_dense_reward": comp.stockfish_dense_reward,
        "terminal_game_reward": comp.terminal_game_reward,
        "total_training_reward": comp.total_training_reward,
    }


def _total_reward(step: Step, z_mover: float, rl: dict) -> float:
    sf_component = step.reward.get("stockfish_dense_reward", 0.0)
    return rl["lambda_stockfish"] * sf_component + rl["lambda_game"] * float(z_mover)

def play_single_game(
    net,
    rl: dict,
    pool,
    engine_depth: int,
    max_plies: int,
    temperature: float,
    device: str,
    rng: np.random.Generator,
    use_sf_head: bool,
    inferer: "_BatchedInferer | None" = None,
) -> Trajectory:
    """Play one full self-play game under the frozen behavior policy."""
    use_sf = pool is not None
    if use_sf_head and not use_sf:
        raise ValueError("use_sf_head requires the Stockfish engine (aux targets)")
    if float(temperature) != 1.0:
        # BEHAVIOR-POLICY CONTRACT: the stored old log-prob must come from the
        # same distribution PPO recomputes (masked log-softmax at T=1).  A
        # tempered rollout policy would silently bias every PPO ratio.
        raise ValueError(
            "training self-play requires temperature == 1.0 (PPO ratio identity, "
            f"spec Section 80); got temperature={temperature}"
        )
    env = ChessEnv()
    steps: list[Step] = []
    s_prev: float | None = None          # S_stm of the pre-move position
    pending_idx: int | None = None       # step awaiting child-score finalization
    engine_evals = 0
    truncated = False
    z_white: float | None = None
    bootstrap_v_train: float | None = None
    bootstrap_v_game: float | None = None

    while True:
        canon = env.canonical_board()
        state = encode_state(canon)
        legal = ActionCodec.encode_legal_moves(canon)
        if inferer is not None:
            logits, v_game, v_train, _ = inferer.infer(state)
        else:
            logits, v_game, v_train, _ = _infer(net, state, device)

        # One temperature distribution serves both sampling and the stored
        # behavior-policy log-probability (they are the same quantity).
        probs = temperature_probs(logits, legal, temperature)
        action = sample_action_from_probs(legal, probs, rng)
        log_prob = log_prob_of(probs, action)

        # Search A on the pre-move position (mover-relative S_t).
        s_stm: float | None = None
        if use_sf:
            info = pool.analyse(env.board.fen(), engine_depth)
            engine_evals += 1
            # Convert with the ANALYZED board: env.board.turn is the true side
            # to move.  Passing `canon` here (turn always WHITE) returned the
            # original-White-perspective score at every Black-to-move ply --
            # a silent sign inversion of the entire dense reward (fixed).
            s_stm = centered_score_stm(info, env.board)

        # Finalize the previous step now that this position's score exists.
        if pending_idx is not None:
            if use_sf:
                _finalize(steps[pending_idx], s_prev, -s_stm, rl)
            else:
                # Engine disabled (Experiment A): no dense signal mid-game.
                _finalize(steps[pending_idx], None, 0.0, rl)
            pending_idx = None

        mover = env.turn
        step = Step(
            state=state,
            action=action,
            legal_indices=legal,
            log_prob=log_prob,
            v_game=v_game,
            v_train=v_train,
            v_sf_target=s_stm if use_sf_head else None,
        )
        steps.append(step)
        s_prev = s_stm

        env.step(action)

        if env.is_terminal():
            z_mover = float(result_for_player(env.board, mover))
            # The terminal child's mover-perspective score is exactly z
            # (mate -> +/-1, draw -> 0; Sec. 22) -- no engine call needed.
            _finalize(step, s_prev, z_mover, rl)
            step.reward["terminal_game_reward"] = z_mover
            step.reward["total_training_reward"] = _total_reward(step, z_mover, rl)
            step.terminal = True
            z_white = z_mover if mover else -z_mover
            break

        if env.ply >= max_plies:
            truncated = True
            if use_sf:
                # One extra child search finalizes the last dense signal.
                info = pool.analyse(env.board.fen(), engine_depth)
                engine_evals += 1
                # Same perspective rule as the pre-move search: convert with
                # the analyzed (original) board, not the canonical relabeling.
                s_child = centered_score_stm(info, env.board)
                _finalize(step, s_prev, -s_child, rl)
            else:
                _finalize(step, None, 0.0, rl)
            step.reward["terminal_game_reward"] = 0.0
            step.reward["total_training_reward"] = _total_reward(step, 0.0, rl)
            step.truncated = True
            # Bootstrap boundary: stm at s_T is the opponent of the last mover.
            bc = env.canonical_board()
            _, boot_vg, boot_vt, _ = _infer(net, encode_state(bc), device)
            bootstrap_v_train = boot_vt
            bootstrap_v_game = boot_vg
            break

        pending_idx = len(steps) - 1

    return Trajectory(
        steps=steps,
        z_white=z_white,
        truncated=truncated,
        bootstrap_v_train=bootstrap_v_train,
        bootstrap_v_game=bootstrap_v_game,
        engine_evals=engine_evals,
        game_plies=len(steps),
    )


def play_games(
    net,
    rl: dict,
    pool,
    n_games: int,
    engine_depth: int,
    max_plies: int,
    temperature: float,
    device: str,
    seed: int,
    use_sf_head: bool,
    threads: int,
    progress_cb=None,
    infer_batch_size: int = 1,
    infer_max_wait_ms: float = 0.0,
) -> list[Trajectory]:
    """Play ``n_games`` self-play games concurrently.

    Trajectory data stays on the CPU (spec Section 54); only minibatches
    move to the GPU at training time.  With ``infer_batch_size > 1`` a
    microbatching inference server amortizes GPU work across concurrent
    games (spec Sections 29-30); with 1 the per-state inference path is
    used unchanged.
    """
    net.eval()
    inferer = None
    if int(infer_batch_size) > 1:
        inferer = _BatchedInferer(net, device, infer_batch_size, infer_max_wait_ms)
    results: list[Trajectory | None] = [None] * n_games
    print_lock = threading.Lock()

    def _one(gi: int) -> None:
        rng = np.random.default_rng(seed + 1000003 * gi)
        traj = play_single_game(
            net, rl, pool, engine_depth, max_plies, temperature,
            device, rng, use_sf_head, inferer=inferer,
        )
        results[gi] = traj
        if progress_cb is not None:
            with print_lock:
                progress_cb(gi, traj)

    workers = max(1, min(threads, n_games))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, range(n_games)))
    finally:
        if inferer is not None:
            inferer.close()
    out = [r for r in results if r is not None]
    if len(out) != n_games:
        raise RuntimeError("self-play produced fewer games than requested")
    return out

