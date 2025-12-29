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
import queue as _py_queue
import multiprocessing as mp
from collections import deque, Counter

import numpy as np
import torch
import random
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
def _selfplay_worker(worker_id: int, request_q, response_q, game_q, stop_event: mp.Event, seed: int):
    try:
        import numpy as _np
        import random as _random
        import torch as _torch

        try:
            _torch.set_num_threads(max(1, int(EVAL_TORCH_THREADS)))
        except Exception:
            pass
        _random.seed(seed)
        _np.random.seed(seed)

        client = InferenceClient(request_q=request_q, response_q=response_q, worker_id=worker_id, stop_event=stop_event)

        # Dummy model object only to satisfy MCTS constructor inside self_play_game.
        # In worker we never call model.forward directly when infer_client is provided.
        dummy_model = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES)
        dummy_model.eval()

        first_game_printed = False

        while not stop_event.is_set():
            samples, outcome, moves_cnt, draw_reason = self_play_game(
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

            if (not first_game_printed) and DEBUG_PRINT_WORKER_FIRST_GAME:
                first_game_printed = True
                print(f"[{_ts()}][worker {worker_id}] first finished game: moves={moves_cnt} term={termination} winner={winner} draw_reason={draw_reason}", flush=True)

            # Send to main
            game_q.put((samples, bool(is_mate), int(winner), termination, draw_reason, moves_cnt), block=True)

    except BaseException:
        # Silent exit when stopping; otherwise re-raise for visibility.
        if stop_event.is_set():
            return
        raise


# -----------------------------
# Eval worker (separate process)
# -----------------------------
def _eval_worker(best_path: str, cand_path: str, num_games: int, result_q, stop_event: mp.Event, seed: int):
    try:
        import numpy as _np
        import random as _random
        import torch as _torch

        try:
            _torch.set_num_threads(max(1, int(EVAL_TORCH_THREADS)))
        except Exception:
            pass
        _random.seed(seed)
        _np.random.seed(seed)

        device = torch.device("cpu")

        best = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device).eval()
        cand = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device).eval()

        best.load_state_dict(torch.load(best_path, map_location=device))
        cand.load_state_dict(torch.load(cand_path, map_location=device))

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
                    move = random.choice(list(board.legal_moves))

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
            # Alternate colors for candidate
            if i % 2 == 0:
                out = play_one(cand, best)   # cand as White
                cand_winner = out.winner
            else:
                out = play_one(best, cand)   # cand as Black
                # if out.winner==BLACK => cand win
                cand_winner = None if out.winner is None else (chess.BLACK if out.winner == chess.BLACK else chess.WHITE)
                # Above mapping isn't needed; we'll handle by explicit check:
                # For i odd, candidate is Black:
                #   cand win if out.winner == BLACK
                #   cand loss if out.winner == WHITE

            term_counter[str(out.termination)] += 1

            if out.winner is None:
                d += 1
            else:
                if i % 2 == 0:
                    # candidate is white
                    if out.winner == chess.WHITE:
                        w += 1
                    else:
                        l += 1
                else:
                    # candidate is black
                    if out.winner == chess.BLACK:
                        w += 1
                    else:
                        l += 1

            # Send progress after each completed game (non-blocking).
            # Message format: ("PROGRESS", games_done, w, d, l)
            try:
                result_q.put_nowait(("PROGRESS", i + 1, w, d, l))
            except Exception:
                pass


            if DEBUG_PRINT_EVAL_TERMINATIONS and (i + 1) % 10 == 0:
                print(f"[{_ts()}][eval] progress {i+1}/{num_games} W/D/L={w}/{d}/{l}", flush=True)

        n = w + d + l
        score = (w + 0.5 * d) / max(1, n)
        lb = _score_ci_lower_bound(score, n, EVAL_SCORE_Z)
        elo = _elo_from_score(score)

        result_q.put(("DONE", w, d, l, score, lb, elo, dict(term_counter)), block=True)

    except BaseException as e:
        result_q.put(("ERROR", str(e)), block=True)
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
    stop_event = ctx.Event()

    def _on_ctrl(_ctrl_type: int):
        try:
            stop_event.set()
        except Exception:
            pass

    _install_windows_ctrl_handler(_on_ctrl)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        # Performance knobs
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


    # Candidate model (trained)
    model = AlphaZeroNet(in_channels=TOTAL_LAYERS, n_actions=TOTAL_MOVES).to(device)
    if device.type == "cuda" and USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
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

    # Replay buffer
    rb = ReplayBuffer(capacity=REPLAY_CAPACITY)
    if rb.load(REPLAY_PATH):
        print(f"Loaded replay buffer: {REPLAY_PATH}")
    else:
        print("Replay buffer not found or legacy format (starting empty).")
        print("If you have an old z-buffer: delete replay_buffer/replay_buffer.npz to rebuild WDL buffer.")

    print(f"Replay sizes: {rb.sizes()}")

    # IPC
    request_q = ctx.Queue(maxsize=NUM_SELFPLAY_WORKERS * 4)
    response_qs = [ctx.Queue(maxsize=16) for _ in range(NUM_SELFPLAY_WORKERS)]
    game_q = ctx.Queue(maxsize=NUM_SELFPLAY_WORKERS * 2)

    # Start workers
    workers = []
    base_seed = int(time.time()) & 0x7FFFFFFF
    for wid in range(NUM_SELFPLAY_WORKERS):
        p = ctx.Process(
            target=_selfplay_worker,
            args=(wid, request_q, response_qs[wid], game_q, stop_event, base_seed + wid * 17),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {NUM_SELFPLAY_WORKERS} self-play workers.")

    # Eval process state
    eval_in_flight = False
    eval_proc = None
    eval_result_q = ctx.Queue(maxsize=256)
    eval_progress_games = 0
    eval_progress_wdl = (0, 0, 0)
    last_eval_at_games = 0
    last_eval_wdl = (0, 0, 0)
    last_eval_score = 0.0
    last_eval_lb = 0.0
    last_eval_elo = 0.0
    last_eval_promoted = False
    last_eval_terms = {}

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

    # Perf counters (reset every tick)
    perf_batches = 0
    perf_bs_sum = 0
    perf_prep_ms = 0.0
    perf_h2d_ms = 0.0
    perf_fwd_ms = 0.0
    perf_d2h_ms = 0.0
    perf_send_ms = 0.0
    perf_train_ms = 0.0

    # Main loop
    last_tick_t = time.time()
    last_save_model_games = 0
    last_save_buf_games = 0

    # --- inference batching state ---
    pending = []  # list of (worker_id, rid, obs, idxs_u16|None)
    pending_deadline = None

    try:
        while not stop_event.is_set():
            did_work = False

            # 1) collect inference requests into a batch
            now = time.time()
            if pending_deadline is None and len(pending) == 0:
                pending_deadline = now + SERVER_BATCH_TIMEOUT_S

            # Drain requests quickly
            while len(pending) < SERVER_MAX_INFERENCE_POSITIONS:
                try:
                    item = request_q.get_nowait()
                    if len(item) == 3:
                        worker_id, rid, obs = item
                        idxs = None
                    else:
                        worker_id, rid, obs, idxs = item
                    pending.append((worker_id, rid, obs, idxs))
                    did_work = True
                except _py_queue.Empty:
                    break

            # If nothing pending, block a bit to reduce busy-wait
            if len(pending) == 0:
                try:
                    item = request_q.get(timeout=SERVER_BATCH_TIMEOUT_S)
                    if len(item) == 3:
                        worker_id, rid, obs = item
                        idxs = None
                    else:
                        worker_id, rid, obs, idxs = item
                    pending.append((worker_id, rid, obs, idxs))
                    did_work = True
                except _py_queue.Empty:
                    pass

            # 2) serve batch if deadline reached or batch full
            now = time.time()
            target_bs = min(SERVER_MAX_INFERENCE_POSITIONS, NUM_SELFPLAY_WORKERS)
            if pending and (len(pending) >= target_bs or (pending_deadline is not None and now >= pending_deadline)):
                t_batch0 = time.perf_counter()
                # Build batch tensor
                obs_batch = np.stack([x[2] for x in pending], axis=0)  # uint8 or float32
                t_prep1 = time.perf_counter()
                x = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32, non_blocking=True)
                if device.type == "cuda" and USE_CHANNELS_LAST:
                    x = x.contiguous(memory_format=torch.channels_last)

                t_h2d1 = time.perf_counter()
                was_training = model.training
                model.eval()

                t_fwd0 = time.perf_counter()
                with torch.no_grad():
                    if device.type == "cuda" and USE_FP16_INFERENCE:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(x)
                    else:
                        out = model(x)

                    pol_logits = out[0]  # (B, A)
                    wdl_logits = out[2]  # (B, 3)

                    # Contempt-aware scalar value used by MCTS in workers
                    probs = torch.softmax(wdl_logits.float(), dim=1)  # (B,3)
                    values_t = (probs[:, 0] - probs[:, 2] + CONTEMPT_DRAW * probs[:, 1])

                t_fwd1 = time.perf_counter()

                if was_training:
                    model.train()

                # Send back per worker (sparse logits for legal moves if idxs provided)
                # NOTE: a single D2H copy for the whole batch avoids many tiny CUDA ops / syncs.
                t_d2h0 = time.perf_counter()
                if device.type == "cuda":
                    pol_cpu = pol_logits.detach().to(dtype=torch.float16).cpu().numpy()
                else:
                    pol_cpu = pol_logits.detach().to(dtype=torch.float32).cpu().numpy()
                values_cpu = values_t.detach().cpu().numpy().astype(np.float32, copy=False)
                t_d2h1 = time.perf_counter()

                for i, (worker_id, rid, _obs, idxs_u16) in enumerate(pending):
                    row = pol_cpu[i]
                    if idxs_u16 is None:
                        logits_np = row
                    else:
                        logits_np = row[idxs_u16]
                    v = float(values_cpu[i])
                    response_qs[int(worker_id)].put((int(rid), logits_np, v), block=True)

                t_send1 = time.perf_counter()

                # Accumulate perf (ms per tick)
                perf_batches += 1
                perf_bs_sum += len(pending)
                perf_prep_ms += (t_prep1 - t_batch0) * 1000.0
                perf_h2d_ms += (t_h2d1 - t_prep1) * 1000.0
                perf_fwd_ms += (t_fwd1 - t_fwd0) * 1000.0
                perf_d2h_ms += (t_d2h1 - t_d2h0) * 1000.0
                perf_send_ms += (t_send1 - t_d2h1) * 1000.0

                served_positions += len(pending)
                served_positions_since += len(pending)
                pending.clear()
                pending_deadline = None
                did_work = True

            # 3) drain finished games and train
            drained_games = 0
            while drained_games < SERVER_DRAIN_GAMES_PER_TICK:
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

                rb.add_game(samples, is_mate=is_mate)

                # Train a few steps
                t_train0 = time.perf_counter()
                for _ in range(TRAIN_STEPS_PER_GAME):
                    batch = rb.sample(BATCH_SIZE, p_mate=P_MATE_IN_BATCH, device=device)
                    if batch is None:
                        break
                    losses = train_one_step(model, optimizer, batch, device=device)

                    # EMA smoothing for logs
                    if ema_loss is None:
                        ema_loss = losses["loss"]
                        ema_pl = losses["policy_loss"]
                        ema_vl = losses["value_loss"]
                    else:
                        ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * losses["loss"]
                        ema_pl = ema_beta * ema_pl + (1.0 - ema_beta) * losses["policy_loss"]
                        ema_vl = ema_beta * ema_vl + (1.0 - ema_beta) * losses["value_loss"]

                perf_train_ms += (time.perf_counter() - t_train0) * 1000.0

            # 4) eval scheduling + result polling
            if (not eval_in_flight) and (games_played - last_eval_at_games >= EVAL_EVERY_GAMES) and games_played > 0:
                # write candidate snapshot for eval process
                os.makedirs("tmp_eval", exist_ok=True)
                cand_tmp = "tmp_eval/candidate_eval.pth"
                best_tmp = "tmp_eval/best_eval.pth"

                torch.save(model.state_dict(), cand_tmp)
                shutil.copy2(BEST_CHECKPOINT_PATH, best_tmp)

                print(f"[{_ts()}][eval] scheduled: games={games_played} num={EVAL_NUM_GAMES} best='{BEST_CHECKPOINT_PATH}' cand='{CHECKPOINT_PATH}' promote_if_score_lb>={EVAL_PROMOTE_SCORE} (z={EVAL_SCORE_Z})", flush=True)

                eval_in_flight = True
                eval_progress_games = 0
                eval_progress_wdl = (0, 0, 0)
                last_eval_promoted = False

                eval_proc = ctx.Process(
                    target=_eval_worker,
                    args=(best_tmp, cand_tmp, EVAL_NUM_GAMES, eval_result_q, stop_event, base_seed ^ 0x12345),
                    daemon=True,
                )
                eval_proc.start()
                did_work = True

            if eval_in_flight:
                # Drain eval queue: may contain multiple progress updates + final result
                while True:
                    try:
                        res = eval_result_q.get_nowait()
                        did_work = True
                    except _py_queue.Empty:
                        break

                    # Progress message: ("PROGRESS", games_done, w, d, l)
                    if isinstance(res, tuple) and len(res) >= 1 and res[0] == "PROGRESS":
                        try:
                            _, done, pw, pd, pl = res
                            eval_progress_games = int(done)
                            eval_progress_wdl = (int(pw), int(pd), int(pl))
                        except Exception:
                            pass
                        continue

                    # Final / error
                    if isinstance(res, tuple) and len(res) >= 1 and res[0] == "ERROR":
                        eval_in_flight = False
                        print(f"[{_ts()}][eval] ERROR: {res[1]}", flush=True)
                        continue

                    if isinstance(res, tuple) and len(res) >= 1 and res[0] == "DONE":
                        eval_in_flight = False
                        _, w, d, l, score, lb, elo, term_counts = res
                        last_eval_at_games = games_played
                        last_eval_wdl = (int(w), int(d), int(l))
                        last_eval_score = float(score)
                        last_eval_lb = float(lb)
                        last_eval_elo = float(elo)
                        last_eval_terms = term_counts

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
                    break

            # 5) periodic save
            if games_played - last_save_model_games >= SAVE_MODEL_PER_GAMES and games_played > 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                last_save_model_games = games_played
                did_work = True
                print(f"[{_ts()}][save] candidate checkpoint -> {CHECKPOINT_PATH}", flush=True)

            if games_played - last_save_buf_games >= SAVE_BUFFER_PER_GAMES and games_played > 0:
                rb.save(REPLAY_PATH)
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
                    if perf_batches > 0:
                        avg_bs = perf_bs_sum / max(1, perf_batches)
                        lines.append(f"[{tss}][perf] batches={perf_batches} avg_bs={avg_bs:.1f} prep_ms={perf_prep_ms:.1f} h2d_ms={perf_h2d_ms:.1f} fwd_ms={perf_fwd_ms:.1f} d2h_ms={perf_d2h_ms:.1f} send_ms={perf_send_ms:.1f} train_ms={perf_train_ms:.1f}")
                        perf_batches = 0
                        perf_bs_sum = 0
                        perf_prep_ms = 0.0
                        perf_h2d_ms = 0.0
                        perf_fwd_ms = 0.0
                        perf_d2h_ms = 0.0
                        perf_send_ms = 0.0
                        perf_train_ms = 0.0

                if DEBUG_EVALUATION:
                    lines.append(
                        f"[{tss}][eval] in_flight={eval_in_flight} "
                        f"progress={eval_progress_games}/{EVAL_NUM_GAMES} curWDL={eval_progress_wdl} "
                        f"last_eval_at_games={last_eval_at_games} "
                        f"last=(W/D/L={last_eval_wdl} score={last_eval_score:.3f} lb={last_eval_lb:.3f} elo={last_eval_elo:.1f} promoted={last_eval_promoted})"
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
                time.sleep(SERVER_IDLE_SLEEP_S)

    except KeyboardInterrupt:
        stop_event.set()

    finally:
        stop_event.set()

        # Avoid Windows hang on exit: close queues & disable feeder-thread join
        def _close_q(q):
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

        # Terminate workers first, then join
        for p in workers:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        for p in workers:
            try:
                p.join(timeout=0.5)
            except Exception:
                pass

        if eval_proc is not None:
            try:
                if eval_proc.is_alive():
                    eval_proc.terminate()
            except Exception:
                pass
            try:
                eval_proc.join(timeout=0.5)
            except Exception:
                pass

        # Close IPC queues
        try:
            _close_q(request_q)
        except Exception:
            pass
        try:
            _close_q(game_q)
        except Exception:
            pass
        try:
            _close_q(eval_result_q)
        except Exception:
            pass
        try:
            for q in response_qs:
                _close_q(q)
        except Exception:
            pass

        # Save final snapshots
        try:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        except Exception:
            pass
        try:
            rb.save(REPLAY_PATH)
        except Exception:
            pass

        print("Done.")


if __name__ == "__main__":
    main()