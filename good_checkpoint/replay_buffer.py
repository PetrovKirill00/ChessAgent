# replay_buffer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from constants import TOTAL_LAYERS, TOTAL_MOVES, REPLAY_CAPACITY, REPLAY_TARGET_MATE_FRACTION

# Replay format:
# version=3
#   mate_obs: (N, C, 8, 8) uint8
#   mate_pi:  (N, A) float32
#   mate_wdl: (N,) uint8  {0,1,2}
#   draw_obs / draw_pi / draw_wdl similarly


def _as_uint8_obs(obs: np.ndarray) -> np.ndarray:
    """Store observations compactly. Expect planes in {0,1} (or float close to that)."""
    if obs.dtype == np.uint8:
        return obs
    return (obs > 0.5).astype(np.uint8)


@dataclass
class ReplayBuffer:
    """Replay buffer optimized for high-throughput self-play.

    We store samples in a single contiguous array, but enforce a *target* mate/draw
    split by capacity: once a class reaches its quota, new samples of that class
    overwrite old samples of the same class (so the stored distribution does not drift).

    Storage dtypes:
      - obs: uint8   (binary planes)
      - pi:  float16 (to reduce RAM; converted to float32 on sampling)
      - wdl: uint8
      - is_mate: uint8  {0,1}
    """

    capacity: int = REPLAY_CAPACITY
    # allocate lazily; we grow until `capacity`
    initial_alloc: int = 8192

    def __post_init__(self) -> None:
        cap = int(self.capacity)
        frac = float(REPLAY_TARGET_MATE_FRACTION)
        if not np.isfinite(frac):
            frac = 0.5
        frac = max(0.0, min(1.0, frac))
        self.mate_quota = int(round(cap * frac))
        self.mate_quota = max(0, min(self.mate_quota, cap))
        self.draw_quota = cap - self.mate_quota

        alloc = int(max(1, min(cap, self.initial_alloc)))
        self._alloc_cap = alloc

        self.obs = np.zeros((alloc, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        self.pi = np.zeros((alloc, TOTAL_MOVES), dtype=np.float16)
        self.wdl = np.zeros((alloc,), dtype=np.uint8)
        self.is_mate = np.zeros((alloc,), dtype=np.uint8)

        self.filled = 0
        self.mate_count = 0
        self.draw_count = 0

        # Per-class overwrite cursors (indices into [0:filled) or [0:capacity) when full)
        self.write_pos_mate = 0
        self.write_pos_draw = 0

    # -----------------------------
    # Sizes / stats
    # -----------------------------
    def sizes(self) -> Dict[str, int]:
        return {
            "mate": int(self.mate_count),
            "draw": int(self.draw_count),
            "total": int(self.filled),
        }

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _grow_to(self, new_cap: int) -> None:
        new_cap = int(min(self.capacity, max(new_cap, self._alloc_cap + 1)))
        if new_cap <= self._alloc_cap:
            return

        new_obs = np.zeros((new_cap, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        new_pi = np.zeros((new_cap, TOTAL_MOVES), dtype=np.float16)
        new_wdl = np.zeros((new_cap,), dtype=np.uint8)
        new_is_mate = np.zeros((new_cap,), dtype=np.uint8)

        if self.filled > 0:
            new_obs[: self.filled] = self.obs[: self.filled]
            new_pi[: self.filled] = self.pi[: self.filled]
            new_wdl[: self.filled] = self.wdl[: self.filled]
            new_is_mate[: self.filled] = self.is_mate[: self.filled]

        self.obs, self.pi, self.wdl, self.is_mate = new_obs, new_pi, new_wdl, new_is_mate
        self._alloc_cap = new_cap

    def _maybe_grow_for(self, add_n: int) -> None:
        """Ensure backing arrays are large enough for upcoming append, but never exceed configured capacity."""
        if self.filled >= self.capacity:
            return
        add_n = int(add_n)
        need = min(self.capacity, self.filled + add_n)
        if need <= self._alloc_cap:
            return
        new_cap = min(self.capacity, max(self._alloc_cap * 2, need))
        self._grow_to(new_cap)

    def _select_overwrite_indices(self, flag: int, n: int) -> np.ndarray:
        """Select `n` indices to overwrite for the given class flag (1=mate, 0=draw)."""
        if n <= 0:
            return np.empty((0,), dtype=np.int64)

        if self.filled <= 0:
            return np.empty((0,), dtype=np.int64)

        # Safety: avoid infinite loops if the class is absent.
        if flag == 1 and self.mate_count <= 0:
            return np.empty((0,), dtype=np.int64)
        if flag == 0 and self.draw_count <= 0:
            return np.empty((0,), dtype=np.int64)

        L = int(self.filled) if self.filled < self.capacity else int(self.capacity)
        out = np.empty((n,), dtype=np.int64)

        pos = int(self.write_pos_mate if flag == 1 else self.write_pos_draw) % max(1, L)
        got = 0
        while got < n:
            if int(self.is_mate[pos]) == int(flag):
                out[got] = pos
                got += 1
            pos += 1
            if pos >= L:
                pos = 0

        if flag == 1:
            self.write_pos_mate = pos
        else:
            self.write_pos_draw = pos

        return out

    # -----------------------------
    # Add / sample
    # -----------------------------
    def add(self, obs: np.ndarray, pi: np.ndarray, wdl: int, is_mate: bool) -> None:
        """Add a single (obs, pi, wdl_label) sample."""
        self.add_game([(obs, pi, int(wdl))], is_mate=is_mate)

    def add_game(self, samples, *, is_mate: bool) -> None:
        """Add many samples from one game (fast path).

        Enforces target mate/draw capacity split using per-class overwrite once a quota is reached.
        """
        if not samples:
            return

        flag = 1 if is_mate else 0
        quota = int(self.mate_quota if flag == 1 else self.draw_quota)
        if quota <= 0:
            return  # this class is disabled by configuration

        k = int(len(samples))

        # Prepare contiguous batch arrays
        obs_b = np.empty((k, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        pi_b = np.empty((k, TOTAL_MOVES), dtype=np.float16)
        wdl_b = np.empty((k,), dtype=np.uint8)

        for i, (obs, pi, wdl) in enumerate(samples):
            obs_b[i] = _as_uint8_obs(obs)
            pi_b[i] = np.asarray(pi, dtype=np.float16)
            wdl_b[i] = np.uint8(int(wdl))

        cur_count = int(self.mate_count if flag == 1 else self.draw_count)
        # Append as much as we can (only while we have free capacity AND class quota not reached)
        can_append = min(k, quota - cur_count, int(self.capacity - self.filled))
        can_append = max(0, int(can_append))

        if can_append > 0:
            self._maybe_grow_for(can_append)
            start = int(self.filled)
            end = start + can_append
            if end > self._alloc_cap:
                self._grow_to(end)

            self.obs[start:end] = obs_b[:can_append]
            self.pi[start:end] = pi_b[:can_append]
            self.wdl[start:end] = wdl_b[:can_append]
            self.is_mate[start:end] = np.uint8(flag)

            self.filled = end
            if flag == 1:
                self.mate_count += can_append
            else:
                self.draw_count += can_append

            # Heuristic: start draw overwrite cursor at the beginning of draw block (avoid scanning mates)
            if flag == 0 and self.write_pos_draw == 0 and self.mate_count > 0:
                self.write_pos_draw = int(self.mate_count)

        # Overwrite the rest inside the same class quota
        rest = k - can_append
        if rest <= 0:
            return

        # If we still don't have any sample of this class stored, we can't overwrite yet.
        if flag == 1 and self.mate_count <= 0:
            return
        if flag == 0 and self.draw_count <= 0:
            return

        idx = self._select_overwrite_indices(flag, rest)
        if idx.size == 0:
            return

        src = slice(can_append, can_append + idx.size)
        self.obs[idx] = obs_b[src]
        self.pi[idx] = pi_b[src]
        self.wdl[idx] = wdl_b[src]
        # is_mate stays the same by construction

    def sample(
        self,
        batch_size: int,
        *,
        p_mate: float,
        device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        sizes = self.sizes()
        if sizes["total"] < batch_size:
            return None

        # Decide how many mate/draw to take.
        n_mate = int(round(batch_size * float(p_mate)))
        n_mate = max(0, min(n_mate, batch_size))
        n_draw = batch_size - n_mate

        if sizes["mate"] == 0:
            n_mate = 0
            n_draw = batch_size
        if sizes["draw"] == 0:
            n_draw = 0
            n_mate = batch_size

        # Rejection sampling by flag (fast; avoids building huge index lists)
        def _sample_flag(flag: int, n: int) -> np.ndarray:
            if n <= 0:
                return np.empty((0,), dtype=np.int64)
            out = np.empty((n,), dtype=np.int64)
            got = 0
            while got < n:
                need = n - got
                # oversample candidates; adjust factor if heavily imbalanced
                k = max(64, int(need * 3))
                cand = np.random.randint(0, self.filled, size=k, dtype=np.int64)
                mask = (self.is_mate[cand] == flag)
                sel = cand[mask]
                if sel.size == 0:
                    continue
                take = min(sel.size, need)
                out[got : got + take] = sel[:take]
                got += take
            return out

        idx_m = _sample_flag(1, n_mate)
        idx_d = _sample_flag(0, n_draw)
        idx = np.concatenate([idx_m, idx_d], axis=0)
        np.random.shuffle(idx)

        obs = self.obs[idx]  # uint8
        pi = self.pi[idx].astype(np.float32, copy=False)  # float32 for stable training
        wdl = self.wdl[idx].astype(np.int64, copy=False)  # CE target

        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32, non_blocking=True)
        pi_t = torch.from_numpy(pi).to(device=device, dtype=torch.float32, non_blocking=True)
        wdl_t = torch.from_numpy(wdl).to(device=device, dtype=torch.long, non_blocking=True)
        return obs_t, pi_t, wdl_t

    # -----------------------------
    # Save / load (format v3, compatible)
    # -----------------------------
    def save(self, path: str, compressed: bool = True) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.filled == 0:
            mate_obs = np.zeros((0, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
            mate_pi = np.zeros((0, TOTAL_MOVES), dtype=np.float32)
            mate_wdl = np.zeros((0,), dtype=np.uint8)
            draw_obs = np.zeros((0, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
            draw_pi = np.zeros((0, TOTAL_MOVES), dtype=np.float32)
            draw_wdl = np.zeros((0,), dtype=np.uint8)
        else:
            flags = self.is_mate[: self.filled].astype(bool, copy=False)
            mate_mask = flags
            draw_mask = ~flags

            mate_obs = self.obs[: self.filled][mate_mask]
            mate_pi = self.pi[: self.filled][mate_mask].astype(np.float32, copy=False)
            mate_wdl = self.wdl[: self.filled][mate_mask]

            draw_obs = self.obs[: self.filled][draw_mask]
            draw_pi = self.pi[: self.filled][draw_mask].astype(np.float32, copy=False)
            draw_wdl = self.wdl[: self.filled][draw_mask]

        if compressed:
            np.savez_compressed(
                path,
                version=np.int32(3),
                mate_obs=mate_obs,
                mate_pi=mate_pi,
                mate_wdl=mate_wdl,
                draw_obs=draw_obs,
                draw_pi=draw_pi,
                draw_wdl=draw_wdl,
            )
        else:
            np.savez(
                path,
                version=np.int32(3),
                mate_obs=mate_obs,
                mate_pi=mate_pi,
                mate_wdl=mate_wdl,
                draw_obs=draw_obs,
                draw_pi=draw_pi,
                draw_wdl=draw_wdl,
            )

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        data = np.load(path, allow_pickle=False)
        version = int(data.get("version", 0))
        if version != 3:
            raise RuntimeError(
                f"Unsupported replay buffer version: {version} (expected 3). "
                "Delete buffer and restart."
            )

        mate_obs = data["mate_obs"]
        mate_pi = data["mate_pi"]
        mate_wdl = data["mate_wdl"]
        draw_obs = data["draw_obs"]
        draw_pi = data["draw_pi"]
        draw_wdl = data["draw_wdl"]

        nm = int(mate_obs.shape[0])
        nd = int(draw_obs.shape[0])
        if nm + nd <= 0:
            self.__post_init__()
            return True

        # Strict capacity split by quota (e.g. 50/50 when REPLAY_TARGET_MATE_FRACTION=0.5)
        keep_m = min(nm, int(self.mate_quota))
        keep_d = min(nd, int(self.draw_quota))

        mate_obs = mate_obs[-keep_m:] if keep_m > 0 else mate_obs[:0]
        mate_pi = mate_pi[-keep_m:] if keep_m > 0 else mate_pi[:0]
        mate_wdl = mate_wdl[-keep_m:] if keep_m > 0 else mate_wdl[:0]

        draw_obs = draw_obs[-keep_d:] if keep_d > 0 else draw_obs[:0]
        draw_pi = draw_pi[-keep_d:] if keep_d > 0 else draw_pi[:0]
        draw_wdl = draw_wdl[-keep_d:] if keep_d > 0 else draw_wdl[:0]

        total = keep_m + keep_d
        if total <= 0:
            self.__post_init__()
            return True

        # Reset, then allocate enough space and load.
        self.__post_init__()
        self._maybe_grow_for(total)
        if total > self._alloc_cap:
            self._grow_to(total)

        # Layout: mates first, then draws (simple and fast).
        if keep_m > 0:
            self.obs[:keep_m] = mate_obs.astype(np.uint8, copy=False)
            self.pi[:keep_m] = mate_pi.astype(np.float16, copy=False)
            self.wdl[:keep_m] = mate_wdl.astype(np.uint8, copy=False)
            self.is_mate[:keep_m] = np.uint8(1)

        if keep_d > 0:
            s = keep_m
            e = keep_m + keep_d
            self.obs[s:e] = draw_obs.astype(np.uint8, copy=False)
            self.pi[s:e] = draw_pi.astype(np.float16, copy=False)
            self.wdl[s:e] = draw_wdl.astype(np.uint8, copy=False)
            self.is_mate[s:e] = np.uint8(0)

        self.filled = total
        self.mate_count = keep_m
        self.draw_count = keep_d

        self.write_pos_mate = 0
        self.write_pos_draw = int(keep_m) if keep_m > 0 else 0

        return True
