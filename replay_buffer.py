# replay_buffer.py
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import chess  # needed to detect CHECKMATE for mate/draw split

from constants import (
    TOTAL_LAYERS,
    TOTAL_MOVES,
    REPLAY_CAPACITY,
    DEFAULT_REPLAY_PATH,
    MATE_BUFFER_FRACTION,
    P_MATE_IN_BATCH,
)


# WDL labels (from the perspective of side-to-move at `obs`)
WDL_WIN = 0
WDL_DRAW = 1
WDL_LOSS = 2


def _z_to_wdl(z_arr: np.ndarray) -> np.ndarray:
    """Convert old scalar z in [-1,0,1] into WDL labels."""
    z_arr = np.asarray(z_arr)
    out = np.full(z_arr.shape, WDL_DRAW, dtype=np.int8)
    out[z_arr > 0] = WDL_WIN
    out[z_arr < 0] = WDL_LOSS
    return out


class ReplayBuffer:
    """Dense replay buffer on preallocated numpy arrays.

    Stores (obs, pi, wdl):
      obs: float32 [TOTAL_LAYERS, 8, 8]
      pi : float32 [TOTAL_MOVES]
      wdl: int8 scalar label: 0=WIN, 1=DRAW, 2=LOSS
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, TOTAL_LAYERS, 8, 8), dtype=np.float32)
        self.pi = np.zeros((self.capacity, TOTAL_MOVES), dtype=np.float32)
        self.wdl = np.zeros((self.capacity,), dtype=np.int8)
        self.idx = 0
        self.sz = 0

    def __len__(self):
        return int(self.sz)

    def add_many(self, data: List[Tuple[np.ndarray, np.ndarray, int]] | Tuple[np.ndarray, np.ndarray, np.ndarray]):
        if not data:
            return

        # support both list[(obs,pi,wdl)] and (obs_batch,pi_batch,wdl_batch)
        if isinstance(data, tuple) and len(data) == 3 and isinstance(data[0], np.ndarray):
            obs_batch, pi_batch, wdl_batch = data
            obs_batch = obs_batch.astype(np.float32, copy=False)
            pi_batch = pi_batch.astype(np.float32, copy=False)
            wdl_batch = wdl_batch.astype(np.int8, copy=False)
        else:
            obs_batch = np.asarray([obs for obs, _, _ in data], dtype=np.float32)
            pi_batch = np.asarray([pi_vec for _, pi_vec, _ in data], dtype=np.float32)
            wdl_batch = np.asarray([wdl for _, _, wdl in data], dtype=np.int8)

        n = int(wdl_batch.shape[0])
        if n == 0:
            return

        # if more than capacity -> take the tail
        if n > self.capacity:
            obs_batch = obs_batch[-self.capacity:]
            pi_batch = pi_batch[-self.capacity:]
            wdl_batch = wdl_batch[-self.capacity:]
            n = self.capacity

        end = self.idx + n
        if end <= self.capacity:
            self.obs[self.idx:end] = obs_batch
            self.pi[self.idx:end] = pi_batch
            self.wdl[self.idx:end] = wdl_batch
        else:
            first = self.capacity - self.idx
            self.obs[self.idx:] = obs_batch[:first]
            self.pi[self.idx:] = pi_batch[:first]
            self.wdl[self.idx:] = wdl_batch[:first]

            remaining = n - first
            self.obs[:remaining] = obs_batch[first:first + remaining]
            self.pi[:remaining] = pi_batch[first:first + remaining]
            self.wdl[:remaining] = wdl_batch[first:first + remaining]

        self.idx = (self.idx + n) % self.capacity
        self.sz = min(self.capacity, self.sz + n)

    def sample_exact(self, batch_size: int):
        """Return exactly batch_size samples (without replacement)."""
        if self.sz < batch_size:
            raise ValueError(f"ReplayBuffer: not enough data to sample_exact({batch_size}), sz={self.sz}")

        idxs = np.random.choice(self.sz, size=batch_size, replace=False)
        return self.obs[idxs], self.pi[idxs], self.wdl[idxs]

    def sample(self, batch_size: int):
        """If buffer has fewer than batch_size, returns as many as there are."""
        batch_size = int(batch_size)
        if self.sz == 0:
            raise ValueError("ReplayBuffer пуст, нечего sample'ить")
        if self.sz <= batch_size:
            idxs = np.random.choice(self.sz, size=self.sz, replace=False)
        else:
            idxs = np.random.choice(self.sz, size=batch_size, replace=False)
        return self.obs[idxs], self.pi[idxs], self.wdl[idxs]


class DualReplayBuffer:
    """Stratified replay:

    - mate_buffer: positions from games that ended by CHECKMATE
    - draw_buffer: everything else

    sample(batch_size, p_mate):
      k ~ round(batch_size * p_mate) from mate_buffer
      batch_size-k from draw_buffer
      then shuffle.
    """

    def __init__(self, total_capacity: int, mate_fraction: float = 0.5):
        total_capacity = int(total_capacity)
        mate_fraction = float(mate_fraction)

        mate_cap = int(round(total_capacity * mate_fraction))
        mate_cap = max(1, min(mate_cap, total_capacity - 1))
        draw_cap = total_capacity - mate_cap

        self.mate = ReplayBuffer(mate_cap)
        self.draw = ReplayBuffer(draw_cap)

    def __len__(self):
        return len(self.mate) + len(self.draw)

    def sizes(self) -> dict:
        return {"mate": len(self.mate), "draw": len(self.draw), "total": len(self)}

    def add_game(self, data: List[Tuple[np.ndarray, np.ndarray, int]], outcome: Optional[chess.Outcome]):
        """Put the whole game either into mate buffer or into draw buffer."""
        is_mate = (
            outcome is not None
            and outcome.winner is not None
            and outcome.termination == chess.Termination.CHECKMATE
        )

        if is_mate:
            self.mate.add_many(data)
        else:
            self.draw.add_many(data)

    def sample(self, batch_size: int, p_mate: float = P_MATE_IN_BATCH):
        if len(self) == 0:
            raise ValueError("DualReplayBuffer пуст, нечего sample'ить")

        batch_size = int(batch_size)
        p_mate = float(p_mate)

        total = len(self)
        if total < batch_size:
            batch_size = total

        m = len(self.mate)
        d = len(self.draw)

        # want k from mate
        k = int(round(batch_size * p_mate))

        # clamp by availability
        if m == 0:
            k = 0
        elif d == 0:
            k = batch_size
        else:
            k = max(0, min(k, batch_size))
            k = min(k, m)
            k = max(k, batch_size - d)

        parts = []
        if k > 0:
            parts.append(self.mate.sample(k))
        if batch_size - k > 0:
            parts.append(self.draw.sample(batch_size - k))

        if not parts:
            raise ValueError("DualReplayBuffer: странное состояние, parts пуст")

        obs = np.concatenate([p[0] for p in parts], axis=0) if len(parts) > 1 else parts[0][0]
        pi = np.concatenate([p[1] for p in parts], axis=0) if len(parts) > 1 else parts[0][1]
        wdl = np.concatenate([p[2] for p in parts], axis=0) if len(parts) > 1 else parts[0][2]

        perm = np.random.permutation(obs.shape[0])
        return obs[perm], pi[perm], wdl[perm]

    # ---- Convenience helpers used by main.py ----
    def sample_torch(self, batch_size: int, device: str | torch.device, p_mate: float = P_MATE_IN_BATCH):
        """Sample and convert to torch tensors on `device`.

        Returns:
            obs_t: float32 (B, C, 8, 8)
            pi_t : float32 (B, A)
            wdl_t: int64   (B,) with {0,1,2}
        """
        obs, pi, wdl = self.sample(batch_size=batch_size, p_mate=p_mate)
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        obs_t = torch.from_numpy(obs).to(dev)
        pi_t = torch.from_numpy(pi).to(dev)
        wdl_t = torch.from_numpy(wdl.astype(np.int64, copy=False)).to(dev)
        return obs_t, pi_t, wdl_t

    def save(self, path: str = DEFAULT_REPLAY_PATH) -> None:
        save_replay_buffer(path)

    def load(self, path: str = DEFAULT_REPLAY_PATH) -> bool:
        return load_replay_buffer(path)


# ===== lazy singleton buffer (important for multiprocessing) =====
_replay_buffer: DualReplayBuffer | None = None


def get_replay_buffer() -> DualReplayBuffer:
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = DualReplayBuffer(REPLAY_CAPACITY, mate_fraction=MATE_BUFFER_FRACTION)
    return _replay_buffer


def save_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rb = get_replay_buffer()
    if len(rb) == 0:
        return

    np.savez_compressed(
        path,
        version=np.int64(3),

        mate_obs=rb.mate.obs[:rb.mate.sz],
        mate_pi=rb.mate.pi[:rb.mate.sz],
        mate_wdl=rb.mate.wdl[:rb.mate.sz],
        mate_idx=np.int64(rb.mate.idx),
        mate_sz=np.int64(rb.mate.sz),
        mate_capacity=np.int64(rb.mate.capacity),

        draw_obs=rb.draw.obs[:rb.draw.sz],
        draw_pi=rb.draw.pi[:rb.draw.sz],
        draw_wdl=rb.draw.wdl[:rb.draw.sz],
        draw_idx=np.int64(rb.draw.idx),
        draw_sz=np.int64(rb.draw.sz),
        draw_capacity=np.int64(rb.draw.capacity),
    )


def load_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> bool:
    if not os.path.exists(path):
        return False

    data = np.load(path, allow_pickle=False)
    rb = get_replay_buffer()

    version = int(data["version"]) if "version" in data else -1

    if version == 3:
        # mate
        rb.mate.obs[:] = 0
        rb.mate.pi[:] = 0
        rb.mate.wdl[:] = 0
        rb.mate.sz = 0
        rb.mate.idx = 0
        rb.mate.add_many((data["mate_obs"], data["mate_pi"], data["mate_wdl"]))

        # draw
        rb.draw.obs[:] = 0
        rb.draw.pi[:] = 0
        rb.draw.wdl[:] = 0
        rb.draw.sz = 0
        rb.draw.idx = 0
        rb.draw.add_many((data["draw_obs"], data["draw_pi"], data["draw_wdl"]))

        return True

    if version == 2:
        # Backward compatibility: old files stored scalar z.
        mate_wdl = _z_to_wdl(data["mate_z"])
        draw_wdl = _z_to_wdl(data["draw_z"])

        rb.mate.obs[:] = 0
        rb.mate.pi[:] = 0
        rb.mate.wdl[:] = 0
        rb.mate.sz = 0
        rb.mate.idx = 0
        rb.mate.add_many((data["mate_obs"], data["mate_pi"], mate_wdl))

        rb.draw.obs[:] = 0
        rb.draw.pi[:] = 0
        rb.draw.wdl[:] = 0
        rb.draw.sz = 0
        rb.draw.idx = 0
        rb.draw.add_many((data["draw_obs"], data["draw_pi"], draw_wdl))

        return True

    return False
