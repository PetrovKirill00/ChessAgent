# main.py
# -*- coding: utf-8 -*-

import os

# Set before importing numpy/torch to reduce Intel runtime noise and Ctrl+C crashes.
os.environ.setdefault("FOR_DISABLE_STACK_TRACE", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_SETTINGS", "0")
# Important for Windows: prevents some fortran/OMP stacks from aborting on Ctrl+C.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import time
import math
import shutil
import threading
import queue as _py_queue
import multiprocessing as mp
from collections import deque, Counter

import numpy as np
import torch
import chess

from constants import *
from nw import AlphaZeroNet
from replay_buffer import ReplayBuffer
from agent import InferenceClient, self_play_game, train_one_step, MCTS, board_to_obs, policy_to_pi_vector, move_to_index


# -----------------------------
# Windows Ctrl handler
# -----------------------------
def _install_windows_ctrl_handler(callback):
    """Register SetConsoleCtrlHandler on Windows. No-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        HANDLER_ROUTINE = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def _handler(ctrl_type):
            try:
                callback(int(ctrl_type))
            except Exception:
                pass
            # True => handled (prevents default abort / forrtl crash)
            return True

        handler = HANDLER_ROUTINE(_handler)
        kernel32.SetConsoleCtrlHandler(handler, True)
        # Keep a reference to avoid GC.
        _install_windows_ctrl_handler._handler_ref = handler  # type: ignore[attr-defined]
    except Exception:
        # If we fail, we just rely on Python's KeyboardInterrupt.
        pass


# -----------------------------
# Helpers for logging
# -----------------------------
def _ts() -> str:
    if not DEBUG_PREFIX_TIME:
        return ""
    return time.strftime("%H:%M:%S")


def _gpm_100(last_finish_ts: deque) -> float:
    if len(last_finish_ts) < 2:
        return 0.0
    dt = last_finish_ts[-1] - last_finish_ts[0]
    if dt <= 1e-9:
        return 0.0
    games = len(last_finish_ts) - 1
    return (games / dt) * 60.0


def _qsize_safe(q):
    """Best-effort qsize() for multiprocessing queues (may be unsupported on Windows)."""
    try:
        return q.qsize()
    except Exception:
        return None




def _async_save_replay_buffer(rb: ReplayBuffer, rb_lock: threading.Lock, path: str, games_tag: int, compressed: bool):
    """Background replay-buffer saver.

    Holds rb_lock while snapshot+save runs to avoid concurrent mutation.
    Note: while this runs, main loop should avoid blocking on rb_lock (skip drain/train).
    """
    try:
        with rb_lock:
            rb.save(path, compressed=compressed)
        print(f"[{_ts()}][save] replay buffer DONE (games={games_tag}) -> {path}", flush=True)
    except Exception as e:
        print(f"[{_ts()}][save] replay buffer ERROR: {e}", flush=True)


def _score_ci_lower_bound(score: float, n: int, z: float) -> float:
    """Normal approx. CI lower bound for score in [0,1]."""
    if n <= 0:
        return 0.0
    # clamp score into (0,1) for variance
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    var = s * (1.0 - s) / n
    return max(0.0, score - z * math.sqrt(var))


def _elo_from_score(score: float) -> float:
    """Approx Elo from expected score (draw treated as 0.5). Informational only."""
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(s / (1.0 - s))


# -----------------------------
# Worker process
# -----------------------------
def _selfplay_worker(worker_id: int, request_q, response_q, game_q, stop_event: mp.Event, seed: int, pause_event=None):
    try:
        import numpy as _np
        import random as _random
        import queue as _queue
        import torch as _torch

        _torch.set_num_threads(1)
        _random.seed(seed)
        _np.random.seed(seed)

        def _progress_put(msg):
            """Non-blocking progress update: keep only the latest message in the queue."""
            try:
                result_q.put(msg, block=False)
            except _queue.Full:
                try:
                    _ = result_q.get(block=False)
                except _queue.Empty:
                    pass
                try:
                    result_q.put(msg, block=False)
                except _queue.Full:
                    pass


        client = InferenceClient(request_q=request_q, response_q=response_q, worker_id=worker_id, stop_event=stop_event, pause_event=pause_event)

        # Dummy model object only to satisfy MCTS constructor inside self_play_game.
        # In worker we never call model.forward directly when infer_client is provided.
        dummy_model = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES)
        dummy_model.eval()

        first_game_printed = False
        finished_games = 0

        while not stop_event.is_set():
            samples, outcome, moves_cnt, draw_reason, prof = self_play_game(
                dummy_model,
                client,
                worker_id,
                max_moves=MAX_GAME_LENGTH,
            )

            is_mate = (
                outcome is not None
                and outcome.winner is not None
                and outcome.termination == chess.Termination.CHECKMATE
            )

            # Winner for quick stats
            if outcome is not None and outcome.winner is not None:
                winner = 1 if outcome.winner == chess.WHITE else -1
            else:
                winner = 0

            termination = str(outcome.termination) if outcome is not None else "UNKNOWN"

            finished_games += 1
            if (not first_game_printed) and DEBUG_PRINT_WORKER_FIRST_GAME:
                first_game_printed = True
                print(f"[{_ts()}][worker {worker_id}] first finished game: moves={moves_cnt} term={termination} winner={winner} draw_reason={draw_reason} | prof: mcts_s={prof.get('mcts_s',0.0):.3f} infer_wait_s={prof.get('infer_wait_s',0.0):.3f} infer_calls={prof.get('infer_calls',0)}", flush=True)

            if DEBUG_SELFPLAY_PROFILE and (finished_games % SELFPLAY_PROFILE_EVERY_N_GAMES == 0):
                print(f"[{_ts()}][worker {worker_id}] game prof: moves={moves_cnt} term={termination} winner={winner} draw_reason={draw_reason} | mcts_s={prof.get('mcts_s',0.0):.3f} infer_wait_s={prof.get('infer_wait_s',0.0):.3f} infer_calls={prof.get('infer_calls',0)}", flush=True)
            # Send to main
            # Send to main (with timeout so Ctrl+C / stop_event can't deadlock here)
            payload = (samples, bool(is_mate), int(winner), termination, draw_reason, moves_cnt)
            _t0_put = time.perf_counter()
            _last_warn = 0.0
            while not stop_event.is_set():
                try:
                    game_q.put(payload, block=True, timeout=WORKER_GAME_PUT_TIMEOUT_S)
                    break
                except _py_queue.Full:
                    now = time.perf_counter()
                    if DEBUG_WORKER_INFER_LATENCY and (now - _t0_put >= WORKER_INFER_WARN_AFTER_S) and (now - _last_warn >= WORKER_GAME_PUT_WARN_EVERY_S):
                        _last_warn = now
                        print(f"[worker {worker_id}] waiting to PUT finished game... {now - _t0_put:.1f}s", flush=True)

    except KeyboardInterrupt:
        return
    except BaseException as e:
        # Silent exit during shutdown (Ctrl+C / stop_event) or when IPC is already closing.
        if stop_event.is_set():
            return
        if isinstance(e, (EOFError, BrokenPipeError, ConnectionResetError, OSError)):
            return
        raise


# -----------------------------
# Eval worker (separate process)
# -----------------------------
def _eval_worker(best_path: str, cand_path: str, num_games: int, result_q, stop_event: mp.Event, seed: int):
    """CPU-only evaluation: plays cand vs best for num_games and reports live progress.

    Messages to result_q:
      - ("PROGRESS", done, total, w, d, l)  (best-effort, keeps only latest)
      - (w, d, l, score, lb, elo, term_counter_dict)  final
      - ("ERROR", str(e)) on failure
    """
    try:
        import numpy as _np
        import random as _random
        import queue as _queue
        import torch as _torch

        _torch.set_num_threads(1)
        _random.seed(seed)
        _np.random.seed(seed)

        def _progress_put(msg):
            """Best-effort: keep only latest progress message."""
            try:
                result_q.put(msg, block=False)
                return
            except Exception:
                pass
            try:
                while True:
                    _ = result_q.get(block=False)
            except Exception:
                pass
            try:
                result_q.put(msg, block=False)
            except Exception:
                pass

        device = _torch.device("cpu")

        best = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device).eval()
        cand = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device).eval()

        best.load_state_dict(_torch.load(best_path, map_location=device))
        cand.load_state_dict(_torch.load(cand_path, map_location=device))

        def play_one(model_white, model_black) -> chess.Outcome:
            board = chess.Board()
            hist = deque(maxlen=LAST_POSITIONS)

            mcts_white = MCTS(model_white, infer_client=None, simulations=EVAL_MCTS_SIMULATIONS)
            mcts_black = MCTS(model_black, infer_client=None, simulations=EVAL_MCTS_SIMULATIONS)

            moves_cnt = 0
            while not board.is_game_over(claim_draw=True) and moves_cnt < EVAL_MAX_GAME_LENGTH:
                hist.append(board.copy(stack=False))
                mcts = mcts_white if board.turn == chess.WHITE else mcts_black

                policy, _vc = mcts.run(
                    board,
                    hist,
                    add_dirichlet_noise=False,
                    reuse_tree=False,
                )

                if policy:
                    move = max(policy.items(), key=lambda kv: kv[1])[0]
                else:
                    move = _random.choice(list(board.legal_moves))

                board.push(move)
                moves_cnt += 1

            out = board.outcome(claim_draw=True)
            if out is None:
                out = chess.Outcome(termination=chess.Termination.VARIANT_DRAW, winner=None)
            return out

        w = d = l = 0
        term_counter = Counter()

        for i in range(num_games):
            if stop_event.is_set():
                return

            # Alternate colors for candidate: even -> cand=White, odd -> cand=Black
            if i % 2 == 0:
                out = play_one(cand, best)
                cand_side = chess.WHITE
            else:
                out = play_one(best, cand)
                cand_side = chess.BLACK

            term_counter[str(out.termination)] += 1

            if out.winner is None:
                d += 1
            else:
                cand_win = (out.winner == cand_side)
                if cand_win:
                    w += 1
                else:
                    l += 1

            _progress_put(("PROGRESS", i + 1, num_games, w, d, l))
            if DEBUG_EVALUATION:
                print(f"[{_ts()}][eval] progress {i+1}/{num_games} W/D/L={w}/{d}/{l}", flush=True)

        n = w + d + l
        score = (w + 0.5 * d) / max(1, n)
        lb = _score_ci_lower_bound(score, n, EVAL_SCORE_Z)
        elo = _elo_from_score(score)

        # Ensure final result isn't blocked behind a progress message.
        try:
            while True:
                _ = result_q.get(block=False)
        except _queue.Empty:
            pass

        result_q.put((w, d, l, score, lb, elo, dict(term_counter)), block=True)

    except BaseException as e:
        try:
            import queue as _queue
            try:
                while True:
                    _ = result_q.get(block=False)
            except _queue.Empty:
                pass
        except Exception:
            pass
        try:
            result_q.put(("ERROR", str(e)), block=True)
        except Exception:
            pass
        return



def _eval_worker_ipc(
    request_q,
    cand_response_q,
    best_response_q,
    cand_wid: int,
    best_wid: int,
    num_games: int,
    result_q,
    stop_event: mp.Event,
    seed: int,
):
    """Evaluation using the main-process IPC inference server (GPU + batching).

    Plays candidate vs best for num_games and reports live progress.

    Messages to result_q:
      - ("PROGRESS", done, total, w, d, l)  (best-effort, keeps only latest)
      - (w, d, l, score, lb, elo, term_counter_dict)  final
      - ("ERROR", str(e)) on failure
    """
    try:
        import random as _random
        import queue as _queue
        import time as _time
        from collections import Counter, deque

        _random.seed(int(seed) & 0xFFFFFFFF)

        # Eval uses IPC inference -> dummy models (MCTS will call infer_client).
        dummy = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES)
        dummy.eval()

        cand_client = InferenceClient(
            request_q=request_q,
            response_q=cand_response_q,
            worker_id=int(cand_wid),
            stop_event=stop_event,
            model_id=0,
            suppress_first_print=True,
        )
        best_client = InferenceClient(
            request_q=request_q,
            response_q=best_response_q,
            worker_id=int(best_wid),
            stop_event=stop_event,
            model_id=1,
            suppress_first_print=True,
        )

        def _progress_put(msg):
            # Keep only latest progress update
            try:
                while True:
                    _ = result_q.get(block=False)
            except _queue.Empty:
                pass
            except Exception:
                pass
            try:
                result_q.put(msg, block=False)
            except Exception:
                pass

        def play_one(cand_is_white: bool):
            board = chess.Board()
            hist = deque(maxlen=LAST_POSITIONS)

            # Two separate MCTS objects (one per side-to-move model).
            mcts_cand = MCTS(dummy, infer_client=cand_client, simulations=EVAL_MCTS_SIMULATIONS)
            mcts_best = MCTS(dummy, infer_client=best_client, simulations=EVAL_MCTS_SIMULATIONS)

            mcts_white = mcts_cand if cand_is_white else mcts_best
            mcts_black = mcts_best if cand_is_white else mcts_cand

            moves_cnt = 0
            while not board.is_game_over(claim_draw=True) and moves_cnt < EVAL_MAX_GAME_LENGTH:
                if stop_event.is_set():
                    return None
                hist.append(board.copy(stack=False))
                mcts = mcts_white if board.turn == chess.WHITE else mcts_black

                policy, _vc = mcts.run(
                    board,
                    hist,
                    add_dirichlet_noise=False,
                    reuse_tree=False,
                )

                if policy:
                    move = max(policy.items(), key=lambda kv: kv[1])[0]
                else:
                    move = _random.choice(list(board.legal_moves))

                board.push(move)
                moves_cnt += 1

            out = board.outcome(claim_draw=True)
            if out is None:
                out = chess.Outcome(termination=chess.Termination.VARIANT_DRAW, winner=None)
            return out

        w = d = l = 0
        term_counter = Counter()

        for i in range(int(num_games)):
            if stop_event.is_set():
                return

            cand_is_white = (i % 2 == 0)
            out = play_one(cand_is_white)
            if out is None:
                return

            term_counter[str(out.termination)] += 1

            if out.winner is None:
                d += 1
            else:
                cand_side = chess.WHITE if cand_is_white else chess.BLACK
                if out.winner == cand_side:
                    w += 1
                else:
                    l += 1

            _progress_put(("PROGRESS", i + 1, int(num_games), int(w), int(d), int(l)))

        n = w + d + l
        score = (w + 0.5 * d) / max(1, n)
        lb = _score_ci_lower_bound(score, n, EVAL_SCORE_Z)
        elo = _elo_from_score(score)

        result_q.put((int(w), int(d), int(l), float(score), float(lb), float(elo), dict(term_counter)), block=True)
        return

    except Exception as e:
        try:
            import queue as _queue
            try:
                while True:
                    _ = result_q.get(block=False)
            except _queue.Empty:
                pass
        except Exception:
            pass
        try:
            result_q.put(("ERROR", str(e)), block=True)
        except Exception:
            pass
        return


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REPLAY_PATH), exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context("spawn")
    print(f"[cfg] workers={NUM_SELFPLAY_WORKERS} leaf_parallel={MCTS_LEAF_PARALLELISM} max_batch={SERVER_MAX_INFERENCE_POSITIONS} batch_timeout={SERVER_BATCH_TIMEOUT_S} obs_uint8={IPC_OBS_UINT8} legal_only={IPC_SEND_ONLY_LEGAL} req_q={REQUEST_Q_MAXSIZE} resp_q={RESPONSE_Q_MAXSIZE} game_q={GAME_Q_MAXSIZE}", flush=True)
    stop_event = ctx.Event()

    def _on_ctrl(_ctrl_type: int):
        try:
            stop_event.set()
        except Exception:
            pass

    _install_windows_ctrl_handler(_on_ctrl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Candidate model (trained)
    model = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    else:
        print("No candidate checkpoint found; starting from random init.")

    # Ensure baseline exists
    if not os.path.exists(BEST_CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(BEST_CHECKPOINT_PATH), exist_ok=True)
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
        print(f"Created baseline checkpoint: {BEST_CHECKPOINT_PATH}")


    # Baseline model (used for eval and promotion gating)
    best_model = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device)
    best_model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=device))
    best_model.eval()

    # Replay buffer
    rb = ReplayBuffer(capacity=REPLAY_CAPACITY)
    if rb.load(REPLAY_PATH):
        print(f"Loaded replay buffer: {REPLAY_PATH}")
    else:
        print("Replay buffer not found or legacy format (starting empty).")
        print("If you have an old z-buffer: delete replay_buffer/replay_buffer.npz to rebuild WDL buffer.")

    print(f"Replay sizes: {rb.sizes()}")

    # Replay-buffer save coordination (avoid multi-second stalls in the main loop)
    rb_lock = threading.Lock()
    rb_save_thread = None


    # IPC
    request_q = ctx.Queue(maxsize=REQUEST_Q_MAXSIZE)

    # Exclusive eval: pause self-play (and optionally training) while evaluation runs.
    pause_selfplay_event = ctx.Event() if EVAL_EXCLUSIVE_MODE else None

    # Response queues: self-play workers + (optional) eval IPC clients (candidate+best).
    extra_eval_resp = 2 if EVAL_USE_IPC_INFERENCE else 0
    response_qs = [ctx.Queue(maxsize=RESPONSE_Q_MAXSIZE) for _ in range(NUM_SELFPLAY_WORKERS + extra_eval_resp)]
    game_q = ctx.Queue(maxsize=GAME_Q_MAXSIZE)

    # Reserve worker ids for eval IPC clients (candidate/best). Only valid if EVAL_USE_IPC_INFERENCE.
    eval_wid_cand = NUM_SELFPLAY_WORKERS
    eval_wid_best = NUM_SELFPLAY_WORKERS + 1

    # Start workers
    workers = []
    base_seed = int(time.time()) & 0x7FFFFFFF
    for wid in range(NUM_SELFPLAY_WORKERS):
        p = ctx.Process(
            target=_selfplay_worker,
            args=(wid, request_q, response_qs[wid], game_q, stop_event, base_seed + wid * 17, pause_selfplay_event),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {NUM_SELFPLAY_WORKERS} self-play workers.")

    # Eval process state
    eval_in_flight = False
    eval_proc = None
    eval_result_q = ctx.Queue(maxsize=1)
    last_eval_at_games = 0
    last_eval_wdl = (0, 0, 0)
    last_eval_score = 0.0
    last_eval_lb = 0.0
    last_eval_elo = 0.0
    last_eval_promoted = False
    last_eval_terms = {}

    # Live eval progress (updates every finished eval game)
    eval_progress_done = 0
    eval_progress_wdl = (0, 0, 0)

    # Stats
    games_played = 0
    mates = 0
    draws = 0
    last_finish_ts = deque(maxlen=101)

    # Per-tick metrics
    served_positions = 0
    served_positions_since = 0
    served_t0 = time.time()

    # Loss EMA for logging
    ema_loss = None
    ema_pl = None
    ema_vl = None
    ema_beta = 0.95

    draw_reason_counter = Counter()

    # -----------------------------
    # Profiling / bottleneck hunting
    # -----------------------------
    prof_req_s = 0.0      # time spent receiving requests (total)
    prof_infer_s = 0.0    # time spent doing NN inference + postproc (total)
    prof_route_s = 0.0    # time spent routing responses to workers (total)
    prof_drain_s = 0.0    # time spent draining finished games (total)
    prof_train_s = 0.0    # time spent training steps (subset of drain)

    # Inference breakdown
    prof_stack_s = 0.0    # np.stack(obs)
    prof_cast_s = 0.0     # uint8->float32 cast (if any)
    prof_h2d_s = 0.0      # host->device transfer / tensor prep
    prof_fw_s = 0.0       # model forward pass (CPU wall time)
    prof_post_s = 0.0     # postproc on GPU/CPU (e.g. softmax/value)
    prof_gather_s = 0.0   # gather/select legal logits on device (best-effort wall time)
    prof_d2h_s = 0.0      # device->host transfers + dtype conversion

    # Route breakdown
    prof_slice_s = 0.0    # slicing logits to legal moves
    prof_put_s = 0.0      # time spent in response_q.put (includes waiting)
    prof_put_msgs = 0   # number of response_q.put calls
    prof_put_items = 0  # number of (rid,logits,value) triplets sent


    # Drain/train breakdown
    prof_rb_add_s = 0.0   # rb.add_game time
    prof_sample_s = 0.0   # rb.sample time

    # Idle time (main loop sleep)
    prof_idle_s = 0.0

    prof_batches = 0
    prof_reqs = 0
    prof_drained_games = 0
    prof_train_steps = 0
    prof_drop_resp = Counter()

    # Optional IPC payload diagnostics
    prof_legal_req = 0
    prof_legal_idx_sum = 0
    prof_obs_uint8 = 0
    prof_obs_f32 = 0

    last_infer_gpu_ms = 0.0
    last_infer_batch = 0
    last_legal_maxlen = 0
    last_used_batched_legal = False


    # Main loop
    last_tick_t = time.time()
    last_save_model_games = 0
    last_save_buf_games = 0

    # --- inference batching state ---
    pending = []  # list of (worker_id, rid, obs, idxs_or_None)
    pending_deadline = None

    try:
        while not stop_event.is_set():
            did_work = False

            # 1) collect inference requests into a batch
            t_req0 = time.perf_counter()
            got_reqs = 0

            now = time.time()
            if pending_deadline is None and len(pending) == 0:
                pending_deadline = now + SERVER_BATCH_TIMEOUT_S

            # Drain requests quickly
            while len(pending) < SERVER_MAX_INFERENCE_POSITIONS:
                try:
                    item = request_q.get_nowait()
                except _py_queue.Empty:
                    break

                # Supported formats:
                #  - (wid, rid, obs)
                #  - (wid, rid, obs, legal_move_indices)                  legacy legal-only
                #  - (wid, rid, obs, model_id)                           routed model
                #  - (wid, rid, obs, legal_move_indices, model_id)        legal-only + routed model
                try:
                    if len(item) == 3:
                        worker_id, rid, obs = item
                        idxs = None
                        model_id = 0
                    elif len(item) == 4:
                        worker_id, rid, obs, x4 = item
                        if isinstance(x4, (int, np.integer)):
                            idxs = None
                            model_id = int(x4)
                        else:
                            idxs = x4
                            model_id = 0
                    else:
                        worker_id, rid, obs, idxs, model_id = item
                        model_id = int(model_id)

                    # Normalize model_id: 0=candidate, 1=best; treat anything else as candidate.
                    if model_id != 1:
                        model_id = 0

                    pending.append((int(worker_id), int(rid), obs, idxs, int(model_id)))
                except Exception:
                    # Malformed request; ignore to keep server alive.
                    continue

                got_reqs += 1
                did_work = True

            if got_reqs:

                prof_reqs += got_reqs
            prof_req_s += (time.perf_counter() - t_req0)

            # 2) serve batch if deadline reached or batch full
            now = time.time()
            if pending and (len(pending) >= SERVER_MAX_INFERENCE_POSITIONS or (pending_deadline is not None and now >= pending_deadline)):
                t_infer0 = time.perf_counter()
                batch_sz = len(pending)
                prof_batches += 1
                last_infer_batch = batch_sz
                # Build batch tensor (with timing)
                _t0 = time.perf_counter()
                obs_batch = np.stack([x[2] for x in pending], axis=0)
                prof_stack_s += (time.perf_counter() - _t0)

                # Track incoming obs dtype (IPC_OBS_UINT8 uses uint8)
                if obs_batch.dtype == np.uint8:
                    prof_obs_uint8 += batch_sz
                elif obs_batch.dtype == np.float32:
                    prof_obs_f32 += batch_sz

                if obs_batch.dtype != np.float32:
                    _t0 = time.perf_counter()
                    obs_batch = obs_batch.astype(np.float32, copy=False)
                    prof_cast_s += (time.perf_counter() - _t0)

                _t0 = time.perf_counter()
                x = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32, non_blocking=True)
                if device.type == "cuda" and USE_CHANNELS_LAST:
                    x = x.contiguous(memory_format=torch.channels_last)
                prof_h2d_s += (time.perf_counter() - _t0)
                # Model routing: model_id 0 -> candidate model, model_id 1 -> best baseline.
                mids = [int(it[4]) for it in pending]
                any_best = any(mid == 1 for mid in mids)
                any_cand = any(mid == 0 for mid in mids)

                # Candidate model may be in train() mode due to training; restore it after inference.
                was_training = bool(model.training)
                model.eval()
                best_model.eval()

                gpu_ms = None
                _ev0 = None
                _ev1 = None
                if DEBUG_SERVER_TIMINGS and device.type == "cuda":
                    _ev0 = torch.cuda.Event(enable_timing=True)
                    _ev1 = torch.cuda.Event(enable_timing=True)

                def _forward(_m, _x):
                    if device.type == "cuda" and USE_FP16_INFERENCE:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            return _m(_x)
                    return _m(_x)

                with torch.no_grad():
                    if _ev0 is not None:
                        _ev0.record()

                    _t0 = time.perf_counter()
                    if any_best and (not any_cand):
                        out = _forward(best_model, x)
                        pol_logits = out[0]  # (B, A)
                        wdl_logits = out[2]  # (B, 3)
                    elif any_cand and (not any_best):
                        out = _forward(model, x)
                        pol_logits = out[0]  # (B, A)
                        wdl_logits = out[2]  # (B, 3)
                    else:
                        # Mixed batch: run two forwards and scatter back to original order.
                        idx_cand = [i for i, mid in enumerate(mids) if mid == 0]
                        idx_best = [i for i, mid in enumerate(mids) if mid == 1]

                        idx_cand_t = torch.tensor(idx_cand, device=device, dtype=torch.long) if idx_cand else None
                        idx_best_t = torch.tensor(idx_best, device=device, dtype=torch.long) if idx_best else None

                        out_cand = _forward(model, x.index_select(0, idx_cand_t)) if idx_cand_t is not None else None
                        out_best = _forward(best_model, x.index_select(0, idx_best_t)) if idx_best_t is not None else None

                        ref_out = out_cand if out_cand is not None else out_best
                        pol_logits = torch.empty((batch_sz, TOTAL_MOVES), device=device, dtype=ref_out[0].dtype)
                        wdl_logits = torch.empty((batch_sz, 3), device=device, dtype=ref_out[2].dtype)

                        if out_cand is not None:
                            pol_logits.index_copy_(0, idx_cand_t, out_cand[0])
                            wdl_logits.index_copy_(0, idx_cand_t, out_cand[2])
                        if out_best is not None:
                            pol_logits.index_copy_(0, idx_best_t, out_best[0])
                            wdl_logits.index_copy_(0, idx_best_t, out_best[2])

                    prof_fw_s += (time.perf_counter() - _t0)

                    if _ev1 is not None:
                        _ev1.record()

                    # pol_logits: (B, A), wdl_logits: (B, 3)
                    # Contempt-aware scalar value used by MCTS in workers
                    _t0 = time.perf_counter()
                    probs = torch.softmax(wdl_logits.float(), dim=1)  # (B,3)
                    values_t = (probs[:, 0] - probs[:, 2] + CONTEMPT_DRAW * probs[:, 1])
                    prof_post_s += (time.perf_counter() - _t0)

                    # Move value to host
                    _t0 = time.perf_counter()
                    values = values_t.detach().cpu().numpy().astype(np.float32, copy=False)
                    prof_d2h_s += (time.perf_counter() - _t0)

                    # Policy: if every request includes legal indices, gather only legal logits on-device first
                    use_batched_legal = False
                    legal_lens = None
                    pol_np = None
                    if IPC_SEND_ONLY_LEGAL and pending:
                        all_have_idxs = True
                        max_len = 0
                        for (_wid, _rid, _obs, _idxs, _mid) in pending:
                            if _idxs is None:
                                all_have_idxs = False
                                break
                            try:
                                n = int(len(_idxs))
                            except Exception:
                                n = 0
                            if n > max_len:
                                max_len = n
                        if all_have_idxs:
                            legal_lens = np.zeros((batch_sz,), dtype=np.int32)
                            if max_len <= 0:
                                # Rare but possible (e.g., game-over positions): empty policy payload
                                pol_np = np.zeros((batch_sz, 0), dtype=np.float32)
                                use_batched_legal = True
                            else:
                                idx_mat = np.zeros((batch_sz, max_len), dtype=np.int64)
                                for i_req, (_wid, _rid, _obs, _idxs, _mid) in enumerate(pending):
                                    n = int(len(_idxs))
                                    legal_lens[i_req] = n
                                    if n:
                                        idx_mat[i_req, :n] = np.asarray(_idxs, dtype=np.int64)
                                _t0 = time.perf_counter()
                                idx_t = torch.from_numpy(idx_mat).to(device=device, dtype=torch.long)
                                gathered = torch.gather(pol_logits, dim=1, index=idx_t)
                                prof_gather_s += (time.perf_counter() - _t0)

                                _t0 = time.perf_counter()
                                pol_np = gathered.detach().cpu().numpy()
                                prof_d2h_s += (time.perf_counter() - _t0)
                                use_batched_legal = True

                    if pol_np is None:
                        _t0 = time.perf_counter()
                        pol_np = pol_logits.detach().cpu().numpy()
                        prof_d2h_s += (time.perf_counter() - _t0)
                    # Record whether we used batched-legal gather (for logs)
                    last_used_batched_legal = bool(use_batched_legal)
                    try:
                        last_legal_maxlen = int(pol_np.shape[1]) if use_batched_legal and pol_np is not None else 0
                    except Exception:
                        last_legal_maxlen = 0


                if DEBUG_SERVER_TIMINGS and device.type == "cuda":
                    # Avoid per-batch global sync; D2H copies above are synchronous by default.
                    try:
                        gpu_ms = float(_ev0.elapsed_time(_ev1))
                    except Exception:
                        gpu_ms = None

                if gpu_ms is not None:
                    last_infer_gpu_ms = float(gpu_ms)
                if was_training:
                    model.train()

                prof_infer_s += (time.perf_counter() - t_infer0)
                # Route responses back to workers (batched per worker to reduce IPC overhead)
                t_route0 = time.perf_counter()
                out_by_wid = {}
                for i_req, ((worker_id, rid, _obs, idxs, _mid), logits_full, v) in enumerate(zip(pending, pol_np, values)):
                    # If request sent only legal move indices, return only that subset to reduce IPC.
                    if idxs is not None:
                        prof_legal_req += 1
                        try:
                            prof_legal_idx_sum += int(len(idxs))
                        except Exception:
                            pass
                        _t0 = time.perf_counter()
                        if use_batched_legal and (legal_lens is not None):
                            n = int(legal_lens[i_req])
                            logits_row = logits_full[:n]
                        else:
                            logits_row = logits_full[idxs]
                        prof_slice_s += (time.perf_counter() - _t0)
                    else:
                        logits_row = logits_full
                
                    wid = int(worker_id)
                    out_by_wid.setdefault(wid, []).append((int(rid), logits_row, float(v)))
                
                for wid, items in out_by_wid.items():
                    q = response_qs[wid]
                    ok = False
                    while not stop_event.is_set():
                        _t0 = time.perf_counter()
                        try:
                            q.put(items, block=True, timeout=SERVER_RESPONSE_PUT_TIMEOUT_S)
                            prof_put_s += (time.perf_counter() - _t0)
                            prof_put_msgs += 1
                            prof_put_items += len(items)
                            ok = True
                            break
                        except _py_queue.Full:
                            prof_put_s += (time.perf_counter() - _t0)
                            # If worker died, don't block the whole server.
                            alive = True
                            if wid < NUM_SELFPLAY_WORKERS:
                                alive = workers[wid].is_alive()
                            elif eval_proc is not None:
                                alive = eval_proc.is_alive()
                            if not alive:
                                prof_drop_resp[wid] += len(items)
                                break
                    if not ok and stop_event.is_set():
                        break


                prof_route_s += (time.perf_counter() - t_route0)

                served_positions += batch_sz
                served_positions_since += batch_sz
                did_work = True

                pending.clear()
                pending_deadline = None
                did_work = True

            # 3) drain finished games and train
            rb_lock_acquired = False
            if ASYNC_REPLAY_SAVE:
                rb_lock_acquired = rb_lock.acquire(blocking=False)
            if (not ASYNC_REPLAY_SAVE) or rb_lock_acquired:
                try:
                    t_drain0 = time.perf_counter()
                    deadline = (t_drain0 + TRAIN_TIME_BUDGET_S) if (TRAIN_TIME_BUDGET_S and TRAIN_TIME_BUDGET_S > 0.0) else None
                    out_of_time = False
                    drained_games = 0
                    while drained_games < SERVER_DRAIN_GAMES_PER_TICK:
                        if deadline is not None and time.perf_counter() >= deadline:
                            out_of_time = True
                            break
                        try:
                            samples, is_mate, winner, termination, draw_reason, moves_cnt = game_q.get_nowait()
                        except _py_queue.Empty:
                            break

                        drained_games += 1
                        did_work = True

                        games_played += 1
                        last_finish_ts.append(time.time())

                        if is_mate:
                            mates += 1
                        if winner == 0:
                            draws += 1
                            if draw_reason:
                                draw_reason_counter[draw_reason] += 1
                            else:
                                draw_reason_counter[termination] += 1

                        _t0 = time.perf_counter()
                        rb.add_game(samples, is_mate=is_mate)
                        prof_rb_add_s += (time.perf_counter() - _t0)

                        # Train a few steps
                        if not (EVAL_EXCLUSIVE_MODE and EVAL_PAUSE_TRAINING and eval_in_flight):
                            for _ in range(TRAIN_STEPS_PER_GAME):
                                if deadline is not None and time.perf_counter() >= deadline:
                                    out_of_time = True
                                    break
                                _t0 = time.perf_counter()
                                batch = rb.sample(BATCH_SIZE, p_mate=P_MATE_IN_BATCH, device=device)
                                prof_sample_s += (time.perf_counter() - _t0)
                                if batch is None:
                                    break
                                _t_step0 = time.perf_counter()
                                losses = train_one_step(model, optimizer, batch, device=device)
                                prof_train_s += (time.perf_counter() - _t_step0)
                                prof_train_steps += 1

                                # EMA smoothing for logs
                                if ema_loss is None:
                                    ema_loss = losses["loss"]
                                    ema_pl = losses["policy_loss"]
                                    ema_vl = losses["value_loss"]
                                else:
                                    ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * losses["loss"]
                                    ema_pl = ema_beta * ema_pl + (1.0 - ema_beta) * losses["policy_loss"]
                                    ema_vl = ema_beta * ema_vl + (1.0 - ema_beta) * losses["value_loss"]

                        if out_of_time:
                            break

                    if drained_games:
                        prof_drained_games += drained_games
                    prof_drain_s += (time.perf_counter() - t_drain0)

                finally:
                    if ASYNC_REPLAY_SAVE:
                        rb_lock.release()

            # 4) eval scheduling + result polling
            if (not eval_in_flight) and (games_played - last_eval_at_games >= EVAL_EVERY_GAMES) and games_played > 0:
                # Clear stale messages (queue is size=1; we keep only latest progress/result).
                try:
                    while True:
                        _ = eval_result_q.get_nowait()
                except _py_queue.Empty:
                    pass
                except Exception:
                    pass

                # Exclusive eval: pause self-play (and optionally training).
                if EVAL_EXCLUSIVE_MODE and pause_selfplay_event is not None:
                    pause_selfplay_event.set()

                eval_in_flight = True
                eval_progress_done = 0
                eval_progress_wdl = (0, 0, 0)

                # Write snapshots for CPU eval (and for reproducibility/logging).
                os.makedirs("tmp_eval", exist_ok=True)
                cand_tmp = "tmp_eval/candidate_eval.pth"
                best_tmp = "tmp_eval/best_eval.pth"
                torch.save(model.state_dict(), cand_tmp)
                shutil.copy2(BEST_CHECKPOINT_PATH, best_tmp)

                print(
                    f"[{_ts()}][eval] schedule: games={EVAL_NUM_GAMES} every={EVAL_EVERY_GAMES} "
                    f"promote_if_score_lb>={EVAL_PROMOTE_SCORE} (z={EVAL_SCORE_Z})",
                    flush=True,
                )

                seed_eval = base_seed ^ 0x12345
                if EVAL_USE_IPC_INFERENCE:
                    # GPU-batched eval through main-process inference server.
                    eval_proc = ctx.Process(
                        target=_eval_worker_ipc,
                        args=(
                            request_q,
                            response_qs[eval_wid_cand],
                            response_qs[eval_wid_best],
                            int(eval_wid_cand),
                            int(eval_wid_best),
                            int(EVAL_NUM_GAMES),
                            eval_result_q,
                            stop_event,
                            int(seed_eval),
                        ),
                        daemon=True,
                    )
                else:
                    # CPU-only eval in a separate process.
                    eval_proc = ctx.Process(
                        target=_eval_worker,
                        args=(best_tmp, cand_tmp, int(EVAL_NUM_GAMES), eval_result_q, stop_event, int(seed_eval)),
                        daemon=True,
                    )

                eval_proc.start()
                did_work = True


            if eval_in_flight:
                try:
                    res = eval_result_q.get_nowait()
                    did_work = True
                except _py_queue.Empty:
                    res = None

                if res is not None:
                    # Live progress messages: ("PROGRESS", done, total, w, d, l)
                    if isinstance(res, tuple) and len(res) >= 1 and res[0] == "PROGRESS":
                        _, done, total, w, d, l = res
                        eval_progress_done = int(done)
                        eval_progress_wdl = (int(w), int(d), int(l))
                        if DEBUG_EVALUATION:
                            pass
                    else:
                        eval_in_flight = False

                        if EVAL_EXCLUSIVE_MODE and pause_selfplay_event is not None:

                            pause_selfplay_event.clear()
                        if isinstance(res, tuple) and len(res) == 2 and res[0] == "ERROR":
                            print(f"[{_ts()}][eval] ERROR: {res[1]}", flush=True)
                        else:
                            w, d, l, score, lb, elo, term_counts = res
                            last_eval_at_games = games_played
                            last_eval_wdl = (int(w), int(d), int(l))
                            last_eval_score = float(score)
                            last_eval_lb = float(lb)
                            last_eval_elo = float(elo)
                            last_eval_terms = term_counts

                            # Finalize live-progress view
                            eval_progress_done = int(w) + int(d) + int(l)
                            eval_progress_wdl = (int(w), int(d), int(l))

                            promote = (lb >= EVAL_PROMOTE_SCORE)
                            last_eval_promoted = bool(promote)

                            print(f"[{_ts()}][eval] done: W/D/L={w}/{d}/{l} score={score:.4f} lb={lb:.4f} elo~{elo:.1f} promoted={promote}", flush=True)
                            if DEBUG_PRINT_EVAL_TERMINATIONS and term_counts:
                                top = sorted(term_counts.items(), key=lambda kv: -kv[1])[:6]
                                top_str = ", ".join([f"{k}:{v}" for k,v in top])
                                print(f"[{_ts()}][eval] terminations: {top_str}", flush=True)

                            if promote:
                                # Promote: baseline <- candidate snapshot
                                torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)
                                print(f"[{_ts()}][eval] PROMOTED -> wrote '{BEST_CHECKPOINT_PATH}'", flush=True)

                                # Refresh in-memory baseline model used by the inference server
                                best_model.load_state_dict(torch.load(BEST_CHECKPOINT_PATH, map_location=device))
                                best_model.eval()


# 5) periodic save
            if games_played - last_save_model_games >= SAVE_MODEL_PER_GAMES and games_played > 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                last_save_model_games = games_played
                did_work = True
                print(f"[{_ts()}][save] candidate checkpoint -> {CHECKPOINT_PATH}", flush=True)

            if games_played - last_save_buf_games >= SAVE_BUFFER_PER_GAMES and games_played > 0:
                if ASYNC_REPLAY_SAVE:
                    if (rb_save_thread is None) or (not rb_save_thread.is_alive()):
                        last_save_buf_games = games_played
                        rb_save_thread = threading.Thread(
                            target=_async_save_replay_buffer,
                            args=(rb, rb_lock, REPLAY_PATH, games_played, REPLAY_SAVE_COMPRESSED),
                            daemon=True,
                        )
                        rb_save_thread.start()
                        did_work = True
                        print(f"[{_ts()}][save] replay buffer SCHEDULED (games={games_played}) -> {REPLAY_PATH}", flush=True)
                    else:
                        # Still saving; don't start another one.
                        pass
                else:
                    rb.save(REPLAY_PATH, compressed=REPLAY_SAVE_COMPRESSED)
                    last_save_buf_games = games_played
                    did_work = True
                    print(f"[{_ts()}][save] replay buffer -> {REPLAY_PATH}", flush=True)

            # 6) periodic tick log
            if time.time() - last_tick_t >= SERVER_TICK_EVERY_S:
                last_tick_t = time.time()

                lines = []
                tss = _ts()

                if DEBUG_GAMES:
                    lines.append(f"[{tss}][games] played={games_played} mates={mates} draws={draws}")

                if DEBUG_INFER:
                    dt = time.time() - served_t0
                    if dt <= 1e-9:
                        infer_rate = 0.0
                    else:
                        infer_rate = served_positions_since / dt
                    served_positions_since = 0
                    served_t0 = time.time()

                    lines.append(f"[{tss}][infer] infer_pos/5s={infer_rate * SERVER_TICK_EVERY_S:.0f} gpm(100)={_gpm_100(last_finish_ts):.2f}")

                if DEBUG_IPC_QUEUES:
                    rq = _qsize_safe(request_q)
                    gq = _qsize_safe(game_q)
                    resp_sizes = [s for s in (_qsize_safe(q) for q in response_qs) if s is not None]
                    if resp_sizes:
                        rmin, rmax = min(resp_sizes), max(resp_sizes)
                        ravg = sum(resp_sizes) / max(1, len(resp_sizes))
                        resp_str = f"{rmin}/{ravg:.1f}/{rmax}"
                    else:
                        resp_str = "NA"
                    lines.append(f"[{tss}][ipc] pending={len(pending)} req_q={rq} game_q={gq} resp_q(min/avg/max)={resp_str}")

                if DEBUG_SERVER_TIMINGS:
                    drops = dict(prof_drop_resp) if prof_drop_resp else {}
                    avg_batch = (prof_reqs / prof_batches) if prof_batches else 0.0
                    avg_legal = (prof_legal_idx_sum / prof_legal_req) if prof_legal_req else 0.0

                    lines.append(
                        f"[{tss}][prof] req={prof_req_s:.3f}s infer={prof_infer_s:.3f}s route={prof_route_s:.3f}s "
                        f"drain+train={prof_drain_s:.3f}s train={prof_train_s:.3f}s "
                        f"batches={prof_batches} reqs={prof_reqs} avg_batch={avg_batch:.1f} "
                        f"drained={prof_drained_games} train_steps={prof_train_steps} "
                        f"last_infer(batch={last_infer_batch} gpu_ms={last_infer_gpu_ms:.2f}) "
                        f"avg_legal={avg_legal:.1f} noidx={int(max(0, prof_reqs - prof_legal_req))} last_legal_maxlen={last_legal_maxlen} batched_legal={int(last_used_batched_legal)} drop_resp={drops}"
                    )
                    lines.append(
                        f"[{tss}][prof2] stack={prof_stack_s:.3f}s cast={prof_cast_s:.3f}s h2d={prof_h2d_s:.3f}s "
                        f"fw={prof_fw_s:.3f}s post={prof_post_s:.3f}s gather={prof_gather_s:.3f}s d2h={prof_d2h_s:.3f}s "
                        f"slice={prof_slice_s:.3f}s put={prof_put_s:.3f}s msgs={prof_put_msgs} items={prof_put_items} "
                        f"rb_add={prof_rb_add_s:.3f}s sample={prof_sample_s:.3f}s idle={prof_idle_s:.3f}s "
                        f"obs(u8={prof_obs_uint8},f32={prof_obs_f32})"
                    )

                    # Reset accumulators each tick (so you can see the bottleneck immediately)
                    prof_req_s = prof_infer_s = prof_route_s = prof_drain_s = prof_train_s = 0.0
                    prof_stack_s = prof_cast_s = prof_h2d_s = prof_fw_s = prof_post_s = prof_gather_s = prof_d2h_s = 0.0
                    prof_slice_s = prof_put_s = 0.0
                    prof_put_msgs = 0
                    prof_put_items = 0
                    prof_rb_add_s = prof_sample_s = 0.0
                    prof_idle_s = 0.0

                    prof_batches = prof_reqs = prof_drained_games = prof_train_steps = 0
                    prof_legal_req = prof_legal_idx_sum = 0
                    prof_obs_uint8 = prof_obs_f32 = 0

                    prof_drop_resp.clear()
                    last_infer_gpu_ms = 0.0
                    last_infer_batch = 0



                if DEBUG_EVALUATION:
                    lines.append(
                        f"[{tss}][eval] in_flight={eval_in_flight} last_eval_at_games={last_eval_at_games} "
                        f"last=(W/D/L={last_eval_wdl} score={last_eval_score:.3f} lb={last_eval_lb:.3f} elo={last_eval_elo:.1f} promoted={last_eval_promoted})"
                    )

                    if eval_in_flight:
                        lines.append(
                            f"[{tss}][eval] progress={eval_progress_done}/{EVAL_NUM_GAMES} cur=(W/D/L={eval_progress_wdl})"
                        )
                if DEBUG_REPLAY:
                    lines.append(f"[{tss}][replay] sizes={rb.sizes()}")

                if DEBUG_LOSSES:
                    if ema_loss is None:
                        lines.append(f"[{tss}][loss] -")
                    else:
                        lines.append(f"[{tss}][loss] total={ema_loss:.4f} policy={ema_pl:.4f} value={ema_vl:.4f}")

                if DEBUG_DRAW_REASON:
                    if draws == 0:
                        lines.append(f"[{tss}][draw] total: -")
                    else:
                        top = draw_reason_counter.most_common(5)
                        top_str = ", ".join([f"{k}:{v}" for k, v in top]) if top else "-"
                        lines.append(f"[{tss}][draw] total={draws} top={top_str}")

                print("\n".join(lines), flush=True)

            if not did_work:
                _t0 = time.perf_counter()
                time.sleep(SERVER_IDLE_SLEEP_S)
                prof_idle_s += (time.perf_counter() - _t0)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()

        # -----------------------------
        # Robust cleanup (Ctrl+C safe)
        # -----------------------------
        def _terminate_process(p, name: str):
            if p is None:
                return
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
            try:
                p.join(timeout=0.5)
            except Exception:
                pass
            # On Windows sometimes terminate() isn't enough (or join() hangs); kill() is harsher.
            try:
                if hasattr(p, "kill") and p.is_alive():
                    p.kill()
            except Exception:
                pass
            try:
                p.join(timeout=0.5)
            except Exception:
                pass

        for p in workers:
            _terminate_process(p, "selfplay")

        _terminate_process(eval_proc, "eval")

        # Close queues to avoid "Done." but process never exits (Windows feeder threads)
        def _close_queue(q):
            if q is None:
                return
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

        _close_queue(request_q)
        _close_queue(game_q)
        _close_queue(eval_result_q)
        for q in response_qs:
            _close_queue(q)

        # Save final snapshots
        try:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        except Exception:
            pass
        try:
            rb.save(REPLAY_PATH)
        except Exception:
            pass

        try:
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        print("Done.", flush=True)



if __name__ == "__main__":
    main()
