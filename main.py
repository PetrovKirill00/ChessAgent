import os

# До импорта torch/numpy (уменьшает "шум" от Intel runtimes)
os.environ.setdefault("FOR_DISABLE_STACK_TRACE", "1")
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("KMP_SETTINGS", "0")

import time
import queue as _py_queue
import multiprocessing as mp
from dataclasses import dataclass
from collections import deque
import traceback

import numpy as np
import torch
import chess

from nw import CNNActorCritic
from constants import (
    CHECKPOINT_PATH,
    DEFAULT_REPLAY_PATH,
    SAVE_MODEL_PER_GAMES,
    SAVE_BUFFER_PER_GAMES,
    TRAINING_MAX_MOVES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_SELFPLAY_WORKERS,
    SERVER_MAX_INFERENCE_POSITIONS,
    SERVER_INFERENCE_WAIT_MS,
    SERVER_DRAIN_GAMES_PER_TICK,
    SERVER_TRAIN_STEPS_PER_GAME,
    SERVER_IDLE_SLEEP_MS,
)

from replay_buffer import get_replay_buffer, load_replay_buffer, save_replay_buffer
from agent import InferenceClient, self_play_game, train_one_step


# --- Windows Ctrl+C (console control) handler ---
_WIN_CTRL_HANDLER_REF = None


def _install_windows_ctrl_handler(on_ctrl):
    """
    Регистрирует SetConsoleCtrlHandler, чтобы перехватывать Ctrl+C на Windows.
    Возвращаем True -> событие считается обработанным и НЕ идёт дальше (в т.ч. forrtl).
    """
    global _WIN_CTRL_HANDLER_REF
    if os.name != "nt":
        return None

    import ctypes
    import ctypes.wintypes as wt

    HandlerRoutine = ctypes.WINFUNCTYPE(wt.BOOL, wt.DWORD)

    def _handler(ctrl_type):
        try:
            on_ctrl(int(ctrl_type))
        except Exception:
            pass
        return True  # поглощаем событие

    _WIN_CTRL_HANDLER_REF = HandlerRoutine(_handler)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(_WIN_CTRL_HANDLER_REF, True)
    return _WIN_CTRL_HANDLER_REF


@dataclass
class InferenceRequest:
    worker_id: int
    req_id: int
    obs_batch: np.ndarray


def _selfplay_worker(worker_id, request_q, response_q, data_q, log_q, stop_event, seed):
    """
    Worker: генерирует self-play партии.
    На Windows поглощаем Ctrl+C event, чтобы не было forrtl spam.
    """
    _install_windows_ctrl_handler(lambda _t: None)

    try:
        import numpy as _np
        import random as _random
        import torch as _torch

        _torch.set_num_threads(1)
        _random.seed(seed)
        _np.random.seed(seed)

        client = InferenceClient(
            worker_id=worker_id,
            request_q=request_q,
            response_q=response_q,
            stop_event=stop_event,
            log_q=log_q,
        )

        while not stop_event.is_set():
            data, outcome = self_play_game(
                inference_client=client,
                model=None,
                max_moves=TRAINING_MAX_MOVES,
                device="cpu",
            )

            if stop_event.is_set():
                break

            if not data:
                continue

            obs = _np.asarray([x[0] for x in data], dtype=_np.float32)
            pi = _np.asarray([x[1] for x in data], dtype=_np.float32)
            z = _np.asarray([x[2] for x in data], dtype=_np.float32)

            is_mate = (
                outcome is not None
                and outcome.winner is not None
                and outcome.termination == chess.Termination.CHECKMATE
            )

            if outcome is not None and outcome.winner is not None:
                winner = 1 if outcome.winner == chess.WHITE else -1
                termination = str(outcome.termination)
            else:
                winner = 0
                termination = str(outcome.termination) if outcome is not None else "UNKNOWN"

            data_q.put((obs, pi, z, bool(is_mate), int(winner), termination))

    except BaseException:
        if stop_event.is_set():
            return
        tb = traceback.format_exc()
        try:
            log_q.put(f"[worker {worker_id}] EXCEPTION:\n{tb}")
        except Exception:
            pass
        try:
            stop_event.set()
        except Exception:
            pass


def _collect_inference_requests(request_q, *, max_total_positions: int, wait_ms: int):
    reqs = []
    timeout_s = max(0.0, float(wait_ms) / 1000.0)

    try:
        worker_id, req_id, obs_batch = request_q.get(timeout=timeout_s)
    except _py_queue.Empty:
        return reqs

    total = int(obs_batch.shape[0])
    reqs.append(InferenceRequest(int(worker_id), int(req_id), obs_batch))

    deadline = time.perf_counter() + timeout_s
    while total < max_total_positions and time.perf_counter() < deadline:
        try:
            worker_id, req_id, obs_batch = request_q.get_nowait()
        except _py_queue.Empty:
            break

        b = int(obs_batch.shape[0])
        if total + b > max_total_positions:
            request_q.put((worker_id, req_id, obs_batch))
            break

        reqs.append(InferenceRequest(int(worker_id), int(req_id), obs_batch))
        total += b

    return reqs


def _serve_inference_batch(model, device, reqs, response_queues):
    if not reqs:
        return 0

    sizes = [int(r.obs_batch.shape[0]) for r in reqs]
    big_obs = np.concatenate([r.obs_batch for r in reqs], axis=0).astype(np.float32, copy=False)

    model.eval()
    obs_t = torch.as_tensor(big_obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits, v_pred = model(obs_t)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32, copy=False)
        values = v_pred.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

    offset = 0
    for r, b in zip(reqs, sizes):
        p = probs[offset:offset + b]
        v = values[offset:offset + b]
        offset += b
        response_queues[r.worker_id].put((r.req_id, p, v))

    return int(big_obs.shape[0])


def _try_qsize(q):
    try:
        return str(q.qsize())
    except Exception:
        return "NA"


def _gpm_100(finish_ts: deque) -> float:
    """
    Games Per Minute (GPM) по последним len(finish_ts) играм.
    Берём (N-1) интервалов между t_first и t_last:
        GPM = (N-1) * 60 / (t_last - t_first)
    """
    n = len(finish_ts)
    if n < 2:
        return 0.0
    dt = float(finish_ts[-1] - finish_ts[0])
    if dt <= 1e-9:
        return 0.0
    return (n - 1) * 60.0 / dt


def main(device="cuda"):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    stop_event = mp.Event()

    # В main тоже поглощаем Ctrl+C на уровне console control handler
    def _on_ctrl(_ctrl_type: int):
        try:
            stop_event.set()
        except Exception:
            pass

    _install_windows_ctrl_handler(_on_ctrl)

    model = CNNActorCritic().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    if load_replay_buffer(DEFAULT_REPLAY_PATH):
        print(f"Loaded replay buffer: {DEFAULT_REPLAY_PATH}")
    else:
        print("Replay buffer not found (starting empty).")

    rb = get_replay_buffer()
    print("Replay sizes:", rb.sizes())

    request_q = mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 8)
    response_queues = [mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 8) for _ in range(NUM_SELFPLAY_WORKERS)]
    data_q = mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 4)
    log_q = mp.Queue(maxsize=NUM_SELFPLAY_WORKERS * 256)

    workers = []
    base_seed = int(time.time()) & 0xFFFF_FFFF
    for wid in range(NUM_SELFPLAY_WORKERS):
        p = mp.Process(
            target=_selfplay_worker,
            args=(wid, request_q, response_queues[wid], data_q, log_q, stop_event, base_seed + wid),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(f"Started {len(workers)} self-play workers.")

    games_seen = 0
    train_budget = 0

    games_mate = 0
    games_draw = 0
    games_white = 0
    games_black = 0

    # ✅ времена завершения последних 100 игр -> считаем GPM по ним
    last_finish_ts = deque(maxlen=100)

    served_positions_since_log = 0
    train_steps_since_log = 0
    last_log_t = time.time()

    stopping_printed = False

    try:
        while not stop_event.is_set():
            did_work = False

            # drain worker logs
            for _ in range(200):
                try:
                    msg = log_q.get_nowait()
                except _py_queue.Empty:
                    break
                print(msg)
                did_work = True

            # detect dead workers
            for i, p in enumerate(workers):
                if not p.is_alive() and not stop_event.is_set():
                    print(f"[main] worker[{i}] died. exitcode={p.exitcode}")
                    stop_event.set()
                    raise RuntimeError("a worker died")

            # pull finished games
            for _ in range(int(SERVER_DRAIN_GAMES_PER_TICK)):
                try:
                    obs, pi, z, is_mate, winner, termination = data_q.get_nowait()
                except _py_queue.Empty:
                    break

                # фиксируем время завершения игры
                last_finish_ts.append(time.time())

                if bool(is_mate):
                    rb.mate.add_many((obs, pi, z))
                    games_mate += 1
                    if int(winner) == 1:
                        games_white += 1
                    elif int(winner) == -1:
                        games_black += 1
                else:
                    rb.draw.add_many((obs, pi, z))
                    games_draw += 1

                games_seen += 1
                train_budget += int(SERVER_TRAIN_STEPS_PER_GAME)
                did_work = True

                res = "MATE" if bool(is_mate) else "DRAW"
                who = "W" if int(winner) == 1 else ("B" if int(winner) == -1 else "-")
                print(
                    f"[game] #{games_seen} result={res} winner={who} termination={termination} "
                    f"positions={len(z)} gpm_100={_gpm_100(last_finish_ts):.2f}"
                )

                if SAVE_MODEL_PER_GAMES and games_seen % int(SAVE_MODEL_PER_GAMES) == 0:
                    torch.save(model.state_dict(), CHECKPOINT_PATH)
                    print(f"[save] model -> {CHECKPOINT_PATH} (games={games_seen})")

                if SAVE_BUFFER_PER_GAMES and games_seen % int(SAVE_BUFFER_PER_GAMES) == 0:
                    save_replay_buffer(DEFAULT_REPLAY_PATH)
                    print(f"[save] buffer -> {DEFAULT_REPLAY_PATH} (games={games_seen})")

            # serve inference
            reqs = _collect_inference_requests(
                request_q,
                max_total_positions=int(SERVER_MAX_INFERENCE_POSITIONS),
                wait_ms=int(SERVER_INFERENCE_WAIT_MS),
            )
            if reqs:
                served = _serve_inference_batch(model, device, reqs, response_queues)
                served_positions_since_log += served
                did_work = True

            # training (как было: учимся только если в этот тик нет inference запросов)
            if train_budget > 0 and not reqs:
                s = train_one_step(model, optimizer, device=device)
                if s.get("did_step"):
                    train_budget -= 1
                    train_steps_since_log += 1
                    did_work = True
                else:
                    train_budget = 0

            if time.time() - last_log_t > 5.0:
                last_log_t = time.time()
                alive = sum(1 for p in workers if p.is_alive())
                gpm100 = _gpm_100(last_finish_ts)

                print(
                    f"[tick] games={games_seen} train_budget={train_budget} alive_workers={alive} "
                    f"rq={_try_qsize(request_q)} dq={_try_qsize(data_q)} "
                    f"served_pos_5s={served_positions_since_log} train_steps_5s={train_steps_since_log} "
                    f"gpm_100={gpm100:.2f} "
                    f"mates={games_mate} draws={games_draw} (W={games_white},B={games_black}) "
                    f"sizes={rb.sizes()}"
                )
                served_positions_since_log = 0
                train_steps_since_log = 0

            if not did_work:
                time.sleep(float(SERVER_IDLE_SLEEP_MS) / 1000.0)

        stopping_printed = True
        print("\nStopping (Ctrl+C)...")

    except KeyboardInterrupt:
        if not stopping_printed:
            print("\nStopping (KeyboardInterrupt)...")
        stop_event.set()

    finally:
        stop_event.set()

        for p in workers:
            if p.is_alive():
                p.join(timeout=2.0)

        for p in workers:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
                p.join(timeout=2.0)

        try:
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Final model saved to {CHECKPOINT_PATH}")
        except Exception as e:
            print(f"[warn] failed to save model: {e}")

        try:
            save_replay_buffer(DEFAULT_REPLAY_PATH)
            print(f"Final replay buffer saved to {DEFAULT_REPLAY_PATH}")
        except Exception as e:
            print(f"[warn] failed to save buffer: {e}")


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=dev)
