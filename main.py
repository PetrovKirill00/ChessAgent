# main.py
# -*- coding: utf-8 -*-
"""Main training loop.

Architecture:
- Main process hosts:
    * the current training model (candidate)
    * a central inference server (batched)
    * replay buffer + training step loop
    * an evaluation process (arena) for gating
- N self-play worker processes generate games using MCTS,
  but they do NOT hold weights locally; they ask the inference server
  for (policy_logits, value) via multiprocessing queues.

Gating (baseline freezing):
- BEST_CHECKPOINT_PATH is the frozen baseline.
- Candidate trains continuously in the main process.
- Periodically we run arena: candidate vs best.
- Promote if score lower bound >= EVAL_PROMOTE_SCORE,
  where score = (W + 0.5*D) / N (standard chess score).

Important:
- Training uses draw contempt (see constants.py).
- Evaluation/gating uses the standard chess score (draw=0.5).
- On Windows, Ctrl+C noise from Intel runtimes (MKL/oneAPI) is silenced via
  FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 set BEFORE importing numpy/torch.
"""

from __future__ import annotations

import os as _os

# --- Windows: silence Intel Fortran (MKL/oneAPI) Ctrl+C handler ---
# Some native runtimes print:
#   forrtl: error (200): program aborting due to control-C event
# when Ctrl+C is broadcast to child processes attached to the same console.
# Setting this BEFORE importing NumPy/PyTorch prevents that noisy handler.
if _os.name == "nt":
    _os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import os
import time
import queue
import signal
import platform
import traceback
import multiprocessing as mp

import numpy as np
import torch
import chess

from constants import (
    CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    REPLAY_PATH,
    EVAL_CANDIDATE_PATH,
    EVAL_TMP_DIR,
    NUM_SELFPLAY_WORKERS,
    MAX_GAME_LENGTH,
    SERVER_MAX_INFERENCE_POSITIONS,
    SERVER_BATCH_TIMEOUT_S,
    SERVER_TICK_EVERY_S,
    SERVER_DRAIN_GAMES_PER_TICK,
    SAVE_MODEL_PER_GAMES,
    SAVE_BUFFER_PER_GAMES,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MIN_REPLAY_SIZE,
    TRAIN_STEPS_PER_GAME,
    EVAL_EVERY_GAMES,
    EVAL_NUM_GAMES,
    EVAL_PROMOTE_SCORE,
    EVAL_SCORE_Z,
    MCTS_SIMULATIONS, EVAL_MCTS_SIMULATIONS,
)

from nw import AlphaZeroNet
from agent import (
    InferenceClient,
    self_play_game,
    train_one_step,
    play_game_models,
    elo_diff_from_score,
    score_lower_bound_from_counts,
)
from replay_buffer import get_replay_buffer, load_replay_buffer, save_replay_buffer


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _atomic_torch_save(state_dict, path: str) -> None:
    """Write a checkpoint atomically to avoid partial files on crash."""
    _ensure_parent_dir(path)
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def _install_sigint_handler(stop_event: mp.Event) -> None:
    """Reliable Ctrl+C behavior.

    Why we do NOT swallow SIGINT:
    - If you install a handler that only sets a flag and returns,
      Python may *not* interrupt blocking calls (Queue.get/put, etc.).
      That makes Ctrl+C feel "stuck sometimes".
    - We set stop_event AND then re-raise KeyboardInterrupt via default handler,
      so main breaks out immediately and performs a graceful shutdown.

    Double Ctrl+C:
    - First Ctrl+C: graceful stop (stop_event set, then KeyboardInterrupt).
    - Second Ctrl+C: hard exit (os._exit).
    """
    if platform.system().lower() == "windows":
        # Windows can also emit Ctrl+Break (SIGBREAK).
        sigbreak = getattr(signal, "SIGBREAK", None)
    else:
        sigbreak = None

    def _handler(sig, frame):
        if not stop_event.is_set():
            print("\n[main] Ctrl+C -> stopping gracefully (press Ctrl+C again to force)")
            stop_event.set()
            # Raise KeyboardInterrupt to break out of any blocking calls immediately.
            signal.default_int_handler(sig, frame)
        else:
            print("\n[main] Ctrl+C again -> hard exit")
            os._exit(130)

    # Install for Ctrl+C
    signal.signal(signal.SIGINT, _handler)
    # Install for Ctrl+Break on Windows (if present)
    if sigbreak is not None:
        try:
            signal.signal(sigbreak, _handler)
        except Exception:
            pass


def _install_windows_console_ctrl_ignore() -> None:
    """Best-effort: ignore console Ctrl events in child processes on Windows.

    Important: We do NOT install this in the main process, to keep Ctrl+C responsive.
    In child processes it helps suppress native-runtime noise on Ctrl+C broadcast.
    """
    if _os.name != "nt":
        return
    try:
        import ctypes  # stdlib
        kernel32 = ctypes.windll.kernel32

        HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def _handler(_ctrl_type: int) -> bool:
            # Return TRUE => handled. The main process coordinates shutdown.
            return True

        global _WIN_CTRL_HANDLER_REF  # keep reference so GC doesn't remove it
        _WIN_CTRL_HANDLER_REF = HandlerRoutine(_handler)
        kernel32.SetConsoleCtrlHandler(_WIN_CTRL_HANDLER_REF, True)
    except Exception:
        pass


def _selfplay_worker(worker_id: int, request_q, response_q, data_q, stop_event: mp.Event):
    """Worker process: generate self-play games and push them to data_q."""
    # Workers must NOT react to Ctrl+C; the main process coordinates shutdown.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    _install_windows_console_ctrl_ignore()

    try:
        infer_client = InferenceClient(request_q, response_q, worker_id=worker_id)
        dummy = AlphaZeroNet()  # only to satisfy signature; weights not used in worker

        while not stop_event.is_set():
            try:
                data, outcome, plies = self_play_game(
                    model=dummy,
                    infer_client=infer_client,
                    worker_id=worker_id,
                    max_moves=MAX_GAME_LENGTH,
                )
                data_q.put((data, outcome, plies), block=True)
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)
    except KeyboardInterrupt:
        return


def _eval_worker(task_q, result_q):
    """Separate process: arena eval candidate vs best, with progress reports."""
    _install_windows_console_ctrl_ignore()
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    while True:
        task = task_q.get()
        if task is None:
            return

        eval_id = task.get("eval_id", 0)
        best_path = task["best_path"]
        cand_path = task["candidate_path"]
        num_games = int(task["num_games"])
        device = task.get("device", "cpu")
        mcts_sims = int(task.get("mcts_sims", MCTS_SIMULATIONS))
        progress_every = int(task.get("progress_every", 5))

        t0 = time.time()
        try:
            device_t = torch.device(device)

            best = AlphaZeroNet().to(device_t)
            cand = AlphaZeroNet().to(device_t)

            best.load_state_dict(torch.load(best_path, map_location=device_t))
            cand.load_state_dict(torch.load(cand_path, map_location=device_t))

            best.eval()
            cand.eval()

            wins = draws = losses = 0
            total_plies = 0

            for g in range(num_games):
                # Alternate colors to reduce first-move bias.
                if g % 2 == 0:
                    outcome, plies, _term = play_game_models(
                        model_white=cand,
                        model_black=best,
                        max_moves=MAX_GAME_LENGTH,
                        mcts_sims=mcts_sims,
                    )
                    cand_is_white = True
                else:
                    outcome, plies, _term = play_game_models(
                        model_white=best,
                        model_black=cand,
                        max_moves=MAX_GAME_LENGTH,
                        mcts_sims=mcts_sims,
                    )
                    cand_is_white = False

                total_plies += int(plies)

                # Candidate perspective
                if outcome is None or outcome.winner is None:
                    draws += 1
                else:
                    cand_won = (outcome.winner == chess.WHITE and cand_is_white) or (
                        outcome.winner == chess.BLACK and (not cand_is_white)
                    )
                    if cand_won:
                        wins += 1
                    else:
                        losses += 1

                done = g + 1
                if (done % progress_every == 0) or (done == num_games):
                    score = (wins + 0.5 * draws) / max(1, wins + draws + losses)
                    score_lb = score_lower_bound_from_counts(wins, draws, losses, z=EVAL_SCORE_Z)
                    elo = elo_diff_from_score(score)
                    result_q.put(
                        {
                            "type": "eval_progress",
                            "eval_id": eval_id,
                            "done": done,
                            "num_games": num_games,
                            "wins": wins,
                            "draws": draws,
                            "losses": losses,
                            "score": float(score),
                            "score_lb": float(score_lb),
                            "elo_diff": float(elo),
                            "elapsed_s": float(time.time() - t0),
                            "mcts_sims": mcts_sims,
                        }
                    )

            score = (wins + 0.5 * draws) / max(1, wins + draws + losses)
            score_lb = score_lower_bound_from_counts(wins, draws, losses, z=EVAL_SCORE_Z)
            elo = elo_diff_from_score(score)
            avg_plies = total_plies / max(1, num_games)

            result_q.put(
                {
                    "type": "eval_done",
                    "eval_id": eval_id,
                    "ok": True,
                    "res": {
                        "games": num_games,
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "score": float(score),
                        "score_lb": float(score_lb),
                        "elo_diff": float(elo),
                        "avg_plies": float(avg_plies),
                        "elapsed_s": float(time.time() - t0),
                        "mcts_sims": mcts_sims,
                        "best_path": best_path,
                        "candidate_path": cand_path,
                    },
                }
            )

        except Exception as e:
            result_q.put(
                {
                    "type": "eval_done",
                    "eval_id": eval_id,
                    "ok": False,
                    "err": str(e),
                    "trace": traceback.format_exc(),
                }
            )


def main():
    mp.freeze_support()

    # On Windows it's safer to enforce spawn explicitly.
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure dirs exist
    _ensure_parent_dir(CHECKPOINT_PATH)
    _ensure_parent_dir(BEST_CHECKPOINT_PATH)
    _ensure_parent_dir(REPLAY_PATH)
    _ensure_parent_dir(EVAL_CANDIDATE_PATH)
    os.makedirs(EVAL_TMP_DIR, exist_ok=True)

    # Candidate model
    model = AlphaZeroNet().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
            print(f"Loaded candidate checkpoint: {CHECKPOINT_PATH}")
        except Exception:
            print("WARNING: failed to load candidate checkpoint, starting fresh.")
            traceback.print_exc()

    model.eval()

    # Ensure baseline exists
    if not os.path.exists(BEST_CHECKPOINT_PATH):
        if os.path.exists(CHECKPOINT_PATH):
            _atomic_torch_save(torch.load(CHECKPOINT_PATH, map_location="cpu"), BEST_CHECKPOINT_PATH)
        else:
            _atomic_torch_save(model.state_dict(), BEST_CHECKPOINT_PATH)
        print(f"Initialized baseline: {BEST_CHECKPOINT_PATH}")

    # Replay buffer (singleton)
    if load_replay_buffer(REPLAY_PATH):
        print(f"Loaded replay buffer: {REPLAY_PATH}")
    rb = get_replay_buffer()
    print("Replay sizes:", rb.sizes())

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Multiprocessing comms
    stop_event = mp.Event()
    _install_sigint_handler(stop_event)

    request_q = mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 16)
    response_qs = [mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 16) for _ in range(NUM_SELFPLAY_WORKERS)]
    data_q = mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 8)

    # Eval process
    eval_task_q = mp.Queue(maxsize=8)
    eval_result_q = mp.Queue(maxsize=32)
    eval_proc = mp.Process(target=_eval_worker, args=(eval_task_q, eval_result_q), daemon=True)
    eval_proc.start()

    # Self-play workers
    workers = []
    for wid in range(NUM_SELFPLAY_WORKERS):
        p = mp.Process(
            target=_selfplay_worker,
            args=(wid, request_q, response_qs[wid], data_q, stop_event),
            daemon=True,
        )
        p.start()
        workers.append(p)
    print(f"Started {NUM_SELFPLAY_WORKERS} self-play workers.")

    # Stats
    games_seen = 0
    mates = 0
    draws = 0
    served_pos_5s = 0
    last_tick = time.time()

    last_100_finish_ts: list[float] = []

    # Evaluation state (gating)
    eval_in_flight = False
    eval_id_counter = 0
    eval_started_ts: float | None = None
    eval_last_progress_ts = 0.0
    last_eval_at_games = -10**9

    try:
        while not stop_event.is_set():
            # --------------------------
            # Central inference server
            # --------------------------
            batch = []
            meta = []
            t_batch_start = time.time()

            while len(batch) < SERVER_MAX_INFERENCE_POSITIONS:
                timeout = max(0.0, SERVER_BATCH_TIMEOUT_S - (time.time() - t_batch_start))
                if timeout == 0.0 and batch:
                    break
                try:
                    rid, obs = request_q.get(timeout=timeout)
                    batch.append(obs)
                    meta.append(rid)
                except queue.Empty:
                    break

            if batch:
                with torch.no_grad():
                    x = torch.from_numpy(np.stack(batch, axis=0)).to(device)
                    pol, val = model(x)
                    pol = pol.detach().cpu().numpy().astype(np.float32)
                    val = val.detach().cpu().numpy().astype(np.float32).reshape(-1)

                # route responses by worker_id embedded in rid (upper 16 bits)
                for rid, logits, v in zip(meta, pol, val):
                    worker_id = int(rid >> 48)
                    response_qs[worker_id].put((rid, logits, float(v)), block=True)

                served_pos_5s += len(batch)

            # --------------------------
            # Drain finished games
            # --------------------------
            drained = 0
            while drained < SERVER_DRAIN_GAMES_PER_TICK:
                try:
                    data, outcome, plies = data_q.get_nowait()
                except queue.Empty:
                    break

                drained += 1
                games_seen += 1

                # outcome stats (mate vs draw-ish)
                if outcome is not None and outcome.winner is not None and outcome.termination == chess.Termination.CHECKMATE:
                    mates += 1
                else:
                    draws += 1

                last_100_finish_ts.append(time.time())
                if len(last_100_finish_ts) > 100:
                    last_100_finish_ts = last_100_finish_ts[-100:]

                rb.add_game(data, outcome)

                # periodic persistence
                if games_seen % SAVE_MODEL_PER_GAMES == 0:
                    _atomic_torch_save(model.state_dict(), CHECKPOINT_PATH)

                if games_seen % SAVE_BUFFER_PER_GAMES == 0:
                    save_replay_buffer(REPLAY_PATH)

                # schedule evaluation
                if (not eval_in_flight) and (games_seen - last_eval_at_games >= EVAL_EVERY_GAMES):
                    # Snapshot candidate weights for evaluation so training can continue in parallel.
                    _atomic_torch_save(model.state_dict(), EVAL_CANDIDATE_PATH)

                    eval_id_counter += 1
                    eval_started_ts = time.time()
                    eval_last_progress_ts = eval_started_ts

                    eval_task_q.put(
                        {
                            "type": "eval",
                            "eval_id": eval_id_counter,
                            "best_path": BEST_CHECKPOINT_PATH,
                            "candidate_path": EVAL_CANDIDATE_PATH,
                            "num_games": EVAL_NUM_GAMES,
                            # Keep eval on CPU so it doesn't steal GPU from self-play/training
                            "device": "cpu",
                            "mcts_sims": EVAL_MCTS_SIMULATIONS,
                            "progress_every": 5,
                        }
                    )
                    eval_in_flight = True
                    last_eval_at_games = games_seen

                    print(
                        f"[eval] scheduled @ {time.strftime('%H:%M:%S')}: "
                        f"games={games_seen} num={EVAL_NUM_GAMES} mcts_sims={MCTS_SIMULATIONS} "
                        f"best='{BEST_CHECKPOINT_PATH}' cand='{EVAL_CANDIDATE_PATH}' "
                        f"promote_if_score_lb>={EVAL_PROMOTE_SCORE} (z={EVAL_SCORE_Z})"
                    )

            # --------------------------
            # Training steps (bounded)
            # --------------------------
            if drained > 0 and len(rb) >= max(MIN_REPLAY_SIZE, BATCH_SIZE):
                steps = int(TRAIN_STEPS_PER_GAME) * drained
                # keep inference responsive even if drained is large
                steps = min(steps, 64)

                model.train()
                for _ in range(steps):
                    obs_t, pi_t, z_t = rb.sample_torch(BATCH_SIZE, device=device)
                    _ = train_one_step(model, optimizer, (obs_t, pi_t, z_t))
                model.eval()

            # --------------------------
            # Collect eval results / progress (gating)
            # --------------------------
            while True:
                try:
                    msg = eval_result_q.get_nowait()
                except queue.Empty:
                    break

                mtype = msg.get("type", "")
                if mtype == "eval_progress":
                    done = int(msg.get("done", 0))
                    num_games = int(msg.get("num_games", 0))
                    w = int(msg.get("wins", 0))
                    d = int(msg.get("draws", 0))
                    l = int(msg.get("losses", 0))
                    score = float(msg.get("score", 0.0))
                    score_lb = float(msg.get("score_lb", 0.0))
                    elo = float(msg.get("elo_diff", 0.0))
                    elapsed = float(msg.get("elapsed_s", 0.0))
                    sims = int(msg.get("mcts_sims", EVAL_MCTS_SIMULATIONS))

                    eval_last_progress_ts = time.time()
                    print(
                        f"[eval] progress id={msg.get('eval_id', '?')} "
                        f"{done}/{num_games} W/D/L={w}/{d}/{l} "
                        f"score={score:.3f} score_lb={score_lb:.3f} elo={elo:+.1f} "
                        f"elapsed={elapsed:.1f}s sims={sims}"
                    )
                    continue

                if mtype == "eval_done":
                    ok = bool(msg.get("ok", False))
                    eval_in_flight = False
                    eval_started_ts = None

                    if not ok:
                        print(f"[eval] ERROR id={msg.get('eval_id', '?')}: {msg.get('err', 'unknown error')}")
                        tr = msg.get("trace")
                        if tr:
                            print(tr)
                        continue

                    res = msg.get("res", {})
                    w = int(res.get("wins", 0))
                    d = int(res.get("draws", 0))
                    l = int(res.get("losses", 0))
                    games_eval = int(res.get("games", 0))
                    score = float(res.get("score", 0.0))
                    score_lb = float(res.get("score_lb", 0.0))
                    elo = float(res.get("elo_diff", 0.0))
                    avg_plies = float(res.get("avg_plies", 0.0))
                    elapsed = float(res.get("elapsed_s", 0.0))
                    sims = int(res.get("mcts_sims", MCTS_SIMULATIONS))

                    promote = (games_eval > 0) and (score_lb >= float(EVAL_PROMOTE_SCORE))
                    print(
                        f"[eval] DONE id={msg.get('eval_id', '?')} games={games_eval} sims={sims} "
                        f"W/D/L={w}/{d}/{l} score={score:.3f} score_lb(z={EVAL_SCORE_Z})={score_lb:.3f} "
                        f"elo_vs_best={elo:+.1f} avg_plies={avg_plies:.1f} elapsed={elapsed:.1f}s "
                        f"{'PROMOTE -> best' if promote else 'no promote'}"
                    )

                    if promote:
                        try:
                            old_mtime = os.path.getmtime(BEST_CHECKPOINT_PATH) if os.path.exists(BEST_CHECKPOINT_PATH) else None
                        except Exception:
                            old_mtime = None

                        _atomic_torch_save(torch.load(EVAL_CANDIDATE_PATH, map_location="cpu"), BEST_CHECKPOINT_PATH)

                        try:
                            new_mtime = os.path.getmtime(BEST_CHECKPOINT_PATH)
                        except Exception:
                            new_mtime = None

                        print(
                            f"[eval] PROMOTED @ {time.strftime('%H:%M:%S')}: "
                            f"best updated -> {BEST_CHECKPOINT_PATH} (from {EVAL_CANDIDATE_PATH}); "
                            f"score={score:.3f} score_lb={score_lb:.3f} elo={elo:+.1f} "
                            f"mtime {old_mtime} -> {new_mtime}"
                        )
                    continue

                print(f"[eval] WARNING: unknown message: {msg}")

            # If eval is running and we haven't heard from it in a while, say so.
            if eval_in_flight and eval_started_ts is not None:
                now = time.time()
                if now - float(eval_last_progress_ts) > 60.0:
                    print(
                        f"[eval] still running... elapsed={now - float(eval_started_ts):.1f}s "
                        f"(no progress messages for {now - float(eval_last_progress_ts):.1f}s)"
                    )
                    eval_last_progress_ts = now

            # --------------------------
            # Tick log
            # --------------------------
            now = time.time()
            if now - last_tick >= SERVER_TICK_EVERY_S:
                gpm = 0.0
                if len(last_100_finish_ts) >= 2:
                    dt = last_100_finish_ts[-1] - last_100_finish_ts[0]
                    if dt > 0:
                        gpm = (len(last_100_finish_ts) / dt) * 60.0

                eval_elapsed = (now - float(eval_started_ts)) if (eval_in_flight and eval_started_ts is not None) else 0.0
                print(
                    f"[tick] games={games_seen} mates={mates} draws={draws} "
                    f"infer_pos/5s={served_pos_5s} gpm(100)={gpm:.1f} "
                    f"eval_in_flight={eval_in_flight} eval_elapsed={eval_elapsed:.1f}s "
                    f"replay={rb.sizes()}"
                )
                served_pos_5s = 0
                last_tick = now

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()

        # Stop eval process
        try:
            eval_task_q.put(None)
        except Exception:
            pass
        try:
            eval_proc.join(timeout=2.0)
        except Exception:
            pass

        # Workers are daemons; still try to join briefly for cleanliness.
        for p in workers:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass

        # Persist state
        try:
            _atomic_torch_save(model.state_dict(), CHECKPOINT_PATH)
        except Exception:
            pass
        try:
            save_replay_buffer(REPLAY_PATH)
        except Exception:
            pass

        print("Done.")


if __name__ == "__main__":
    main()
