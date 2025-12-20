# replay_buffer.py
import os
from typing import List, Tuple, Optional

import numpy as np
import chess  # нужен для определения CHECKMATE при add_game

from constants import (
    TOTAL_LAYERS,
    TOTAL_MOVES,
    REPLAY_CAPACITY,
    DEFAULT_REPLAY_PATH,
    MATE_BUFFER_FRACTION,
    P_MATE_IN_BATCH,
)


class ReplayBuffer:
    """
    Плотный replay buffer на предвыделенных numpy-массивах.
    Хранит (obs, pi, z):
      obs: [TOTAL_LAYERS, 8, 8]
      pi : [TOTAL_MOVES]
      z  : scalar float
    """

    def __init__(self, capacity: int):
        self.capacity = int(capacity)

        self.obs = np.zeros((self.capacity, TOTAL_LAYERS, 8, 8), dtype=np.float32)
        self.pi = np.zeros((self.capacity, TOTAL_MOVES), dtype=np.float32)
        self.z = np.zeros((self.capacity,), dtype=np.float32)

        self.idx = 0
        self.sz = 0

    def __len__(self) -> int:
        return int(self.sz)

    def add_many(self, data: List[Tuple[np.ndarray, np.ndarray, float]]):
        if not data:
            return

        if (
            isinstance(data, tuple)
            and len(data) == 3
            and isinstance(data[0], np.ndarray)
        ):
            obs_batch, pi_batch, z_batch = data
            obs_batch = obs_batch.astype(np.float32, copy=False)
            pi_batch = pi_batch.astype(np.float32, copy=False)
            z_batch = z_batch.astype(np.float32, copy=False)
        else:
            obs_batch = np.asarray([obs for obs, _, _ in data], dtype=np.float32)
            pi_batch = np.asarray([pi_vec for _, pi_vec, _ in data], dtype=np.float32)
            z_batch = np.asarray([z for _, _, z in data], dtype=np.float32)

        n = int(z_batch.shape[0])
        if n == 0:
            return

        if n > self.capacity:
            obs_batch = obs_batch[-self.capacity:]
            pi_batch = pi_batch[-self.capacity:]
            z_batch = z_batch[-self.capacity:]
            n = self.capacity

        end = self.idx + n
        if end <= self.capacity:
            self.obs[self.idx:end] = obs_batch
            self.pi[self.idx:end] = pi_batch
            self.z[self.idx:end] = z_batch
        else:
            first = self.capacity - self.idx
            self.obs[self.idx:] = obs_batch[:first]
            self.pi[self.idx:] = pi_batch[:first]
            self.z[self.idx:] = z_batch[:first]

            remaining = n - first
            self.obs[:remaining] = obs_batch[first:first + remaining]
            self.pi[:remaining] = pi_batch[first:first + remaining]
            self.z[:remaining] = z_batch[first:first + remaining]

        self.idx = (self.idx + n) % self.capacity
        self.sz = min(self.capacity, self.sz + n)

    def sample_exact(self, batch_size: int):
        """
        Возвращает ровно batch_size элементов (без replacement).
        Вызывать только если self.sz >= batch_size.
        """
        if self.sz < batch_size:
            raise ValueError(f"ReplayBuffer: not enough data to sample_exact({batch_size}), sz={self.sz}")

        indices = np.random.choice(self.sz, size=batch_size, replace=False)
        return self.obs[indices], self.pi[indices], self.z[indices]


class DualReplayBuffer:
    """
    Два буфера:
      - mate_buffer: позиции из партий, закончившихся checkmate
      - draw_buffer: позиции из остальных партий (draw / exceeded move limit / etc)

    sample(batch_size) возвращает смешанный batch:
      k ~ round(batch_size * p_mate) из mate_buffer
      batch_size-k из draw_buffer
      затем shuffle.
    """

    def __init__(self, total_capacity: int, mate_fraction: float = 0.5):
        total_capacity = int(total_capacity)
        mate_fraction = float(mate_fraction)

        mate_cap = int(round(total_capacity * mate_fraction))
        mate_cap = max(1, min(mate_cap, total_capacity - 1))
        draw_cap = total_capacity - mate_cap

        self.mate = ReplayBuffer(mate_cap)
        self.draw = ReplayBuffer(draw_cap)

    def __len__(self) -> int:
        return len(self.mate) + len(self.draw)

    def sizes(self) -> dict:
        return {"mate": len(self.mate), "draw": len(self.draw), "total": len(self)}

    def add_game(self, data: List[Tuple[np.ndarray, np.ndarray, float]], outcome: Optional[chess.Outcome]):
        """
        Кладём ВСЮ партию либо в mate, либо в draw.
        mate: outcome.termination == CHECKMATE
        иначе: draw
        """
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
            # как и в твоём старом буфере: если мало данных, отдадим сколько есть
            batch_size = total

        m = len(self.mate)
        d = len(self.draw)

        # хотим k из mate
        k_desired = int(round(batch_size * p_mate))
        k = min(k_desired, m)
        rest = batch_size - k

        # добираем из draw
        take_d = min(rest, d)
        rest2 = rest - take_d

        # если draw не хватило, докидываем из mate (гарантировано хватит, т.к. total >= batch_size)
        take_m2 = rest2
        k_total = k + take_m2

        parts = []
        if k_total > 0:
            obs_m, pi_m, z_m = self.mate.sample_exact(k_total)
            parts.append((obs_m, pi_m, z_m))

        if take_d > 0:
            obs_d, pi_d, z_d = self.draw.sample_exact(take_d)
            parts.append((obs_d, pi_d, z_d))

        obs = np.concatenate([p[0] for p in parts], axis=0) if len(parts) > 1 else parts[0][0]
        pi = np.concatenate([p[1] for p in parts], axis=0) if len(parts) > 1 else parts[0][1]
        z = np.concatenate([p[2] for p in parts], axis=0) if len(parts) > 1 else parts[0][2]

        # shuffle внутри batch
        perm = np.random.permutation(obs.shape[0])
        return obs[perm], pi[perm], z[perm]


# Глобальный буфер (чтобы agent.py почти не трогать)
replay_buffer = DualReplayBuffer(REPLAY_CAPACITY, mate_fraction=MATE_BUFFER_FRACTION)


def save_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> None:
    """
    Сохраняем ДВА буфера в один .npz.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rb = replay_buffer

    if len(rb) == 0:
        return

    np.savez_compressed(
        path,
        version=np.int64(2),

        mate_obs=rb.mate.obs[:rb.mate.sz],
        mate_pi=rb.mate.pi[:rb.mate.sz],
        mate_z=rb.mate.z[:rb.mate.sz],
        mate_idx=np.int64(rb.mate.idx),
        mate_sz=np.int64(rb.mate.sz),
        mate_capacity=np.int64(rb.mate.capacity),

        draw_obs=rb.draw.obs[:rb.draw.sz],
        draw_pi=rb.draw.pi[:rb.draw.sz],
        draw_z=rb.draw.z[:rb.draw.sz],
        draw_idx=np.int64(rb.draw.idx),
        draw_sz=np.int64(rb.draw.sz),
        draw_capacity=np.int64(rb.draw.capacity),
    )


def load_replay_buffer(path: str = DEFAULT_REPLAY_PATH) -> bool:
    """
    Загружаем новый формат (version=2).
    """
    if not os.path.exists(path):
        return False

    data = np.load(path, allow_pickle=False)
    rb = replay_buffer

    # ---- новый формат ----
    if "version" in data and int(data["version"]) == 2:
        # mate
        mate_obs = data["mate_obs"]
        mate_pi = data["mate_pi"]
        mate_z = data["mate_z"]

        rb.mate.obs[:] = 0
        rb.mate.pi[:] = 0
        rb.mate.z[:] = 0
        rb.mate.sz = 0
        rb.mate.idx = 0
        rb.mate.add_many((mate_obs, mate_pi, mate_z))

        # draw
        draw_obs = data["draw_obs"]
        draw_pi = data["draw_pi"]
        draw_z = data["draw_z"]

        rb.draw.obs[:] = 0
        rb.draw.pi[:] = 0
        rb.draw.z[:] = 0
        rb.draw.sz = 0
        rb.draw.idx = 0
        rb.draw.add_many((draw_obs, draw_pi, draw_z))

        return True

    # непонятный файл
    return False
