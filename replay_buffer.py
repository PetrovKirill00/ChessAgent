# replay_buffer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from constants import TOTAL_LAYERS, TOTAL_MOVES, REPLAY_CAPACITY

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
    # be robust to float32 planes
    return (obs > 0.5).astype(np.uint8)


@dataclass
class ReplayBuffer:
    """Replay buffer optimized for high-throughput self-play.

    Key idea: keep a single contiguous storage and use a ring buffer once the
    configured capacity is reached. This avoids O(N^2) reallocations from
    repeated np.concatenate and avoids per-element trimming.

    Storage dtypes:
      - obs: uint8  (binary planes)
      - pi:  float16 (to reduce RAM; converted to float32 on sampling)
      - wdl: uint8
      - is_mate: uint8  {0,1}
    """

    capacity: int = REPLAY_CAPACITY
    # allocate lazily; we grow until `capacity`, then become a ring buffer
    initial_alloc: int = 8192

    def __post_init__(self) -> None:
        cap = int(max(1, min(self.capacity, self.initial_alloc)))
        self._alloc_cap = cap

        self.obs = np.zeros((cap, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        self.pi = np.zeros((cap, TOTAL_MOVES), dtype=np.float16)
        self.wdl = np.zeros((cap,), dtype=np.uint8)
        self.is_mate = np.zeros((cap,), dtype=np.uint8)

        self.filled = 0
        self.write_pos = 0  # where the next write starts (ring mode only)
        self.mate_count = 0
        self.draw_count = 0

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

        # Before reaching full capacity, we are in append-only mode, so data is in [0:filled).
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

        # still append-only unless we reached full configured capacity AND filled==capacity
        if self.filled == self.capacity:
            self.write_pos = 0
        else:
            self.write_pos = self.filled

    def _maybe_grow_for(self, add_n: int) -> None:
        """Ensure backing arrays are large enough for upcoming append, but never exceed configured capacity."""
        if self.filled >= self.capacity:
            return  # ring mode; fixed size
        add_n = int(add_n)
        # We never allocate beyond the configured capacity.
        need = min(self.capacity, self.filled + add_n)
        if need <= self._alloc_cap:
            return
        new_cap = min(self.capacity, max(self._alloc_cap * 2, need))
        self._grow_to(new_cap)

    def _write_block(self, start: int, obs_b: np.ndarray, pi_b: np.ndarray, wdl_b: np.ndarray, is_mate_flag: int) -> None:
        k = int(obs_b.shape[0])
        end = start + k

        if self.filled == self.capacity:
            # Overwrite counts for the range being overwritten
            old_flags = self.is_mate[start:end]
            old_mates = int(old_flags.sum())
            self.mate_count -= old_mates
            self.draw_count -= (k - old_mates)

        # Write payload
        self.obs[start:end] = obs_b
        self.pi[start:end] = pi_b
        self.wdl[start:end] = wdl_b
        self.is_mate[start:end] = is_mate_flag

        # Add new counts
        if is_mate_flag:
            self.mate_count += k
        else:
            self.draw_count += k

    # -----------------------------
    # Add / sample
    # -----------------------------
    def add(self, obs: np.ndarray, pi: np.ndarray, wdl: int, is_mate: bool) -> None:
        """Add a single (obs, pi, wdl_label) sample."""
        self.add_game([(obs, pi, int(wdl))], is_mate=is_mate)

    def add_game(self, samples, *, is_mate: bool) -> None:
        """Add many samples from one game (fast path).

        Important: this function must be safe when a game crosses the buffer capacity boundary.
        We fill until capacity, then switch to ring overwrite mode for the remainder.
        """
        if not samples:
            return

        is_mate_flag = 1 if is_mate else 0
        k = int(len(samples))
        self._maybe_grow_for(k)

        # Prepare contiguous batch arrays
        obs_b = np.empty((k, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        pi_b = np.empty((k, TOTAL_MOVES), dtype=np.float16)
        wdl_b = np.empty((k,), dtype=np.uint8)

        for i, (obs, pi, wdl) in enumerate(samples):
            obs_b[i] = _as_uint8_obs(obs)
            # store as float16 to reduce RAM; convert to float32 when sampling
            pi_b[i] = np.asarray(pi, dtype=np.float16)
            wdl_b[i] = np.uint8(int(wdl))

        cap = int(self.capacity)

        # Ensure full capacity allocated once we enter ring mode.
        if self.filled >= cap and self._alloc_cap < cap:
            self._grow_to(cap)

        # Append mode until full configured capacity is reached; then ring overwrite mode.
        if self.filled < cap:
            start = int(self.filled)
            space = cap - start
            if k <= space:
                # Pure append without crossing capacity.
                if start + k > self._alloc_cap:
                    self._grow_to(start + k)
                self._write_block(start, obs_b, pi_b, wdl_b, is_mate_flag)
                self.filled = start + k
                self.write_pos = 0 if self.filled == cap else self.filled
                return

            # We cross the capacity boundary: write tail to the end, then wrap to 0.
            if space > 0:
                if cap > self._alloc_cap:
                    self._grow_to(cap)
                self._write_block(start, obs_b[:space], pi_b[:space], wdl_b[:space], is_mate_flag)

            self.filled = cap
            pos = 0
            idx = space
            while idx < k:
                chunk = min(cap - pos, k - idx)
                self._write_block(pos, obs_b[idx:idx+chunk], pi_b[idx:idx+chunk], wdl_b[idx:idx+chunk], is_mate_flag)
                idx += chunk
                pos = (pos + chunk) % cap
            self.write_pos = pos
            return

        # Ring overwrite mode (fixed size).
        pos = int(self.write_pos) % cap
        idx = 0
        while idx < k:
            chunk = min(cap - pos, k - idx)
            self._write_block(pos, obs_b[idx:idx+chunk], pi_b[idx:idx+chunk], wdl_b[idx:idx+chunk], is_mate_flag)
            idx += chunk
            pos = (pos + chunk) % cap
        self.filled = cap
        self.write_pos = pos

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
                # oversample candidates; adjust factor based on expected class imbalance
                need = n - got
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
            mate_mask = np.zeros((0,), dtype=bool)
            draw_mask = np.zeros((0,), dtype=bool)
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

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return

        data = np.load(path, allow_pickle=False)
        version = int(data.get("version", 0))
        if version != 3:
            raise RuntimeError(f"Unsupported replay buffer version: {version} (expected 3). Delete buffer and restart.")

        mate_obs = data["mate_obs"]
        mate_pi = data["mate_pi"]
        mate_wdl = data["mate_wdl"]
        draw_obs = data["draw_obs"]
        draw_pi = data["draw_pi"]
        draw_wdl = data["draw_wdl"]

        nm = int(mate_obs.shape[0])
        nd = int(draw_obs.shape[0])
        total = nm + nd
        if total <= 0:
            # reset to empty
            self.__post_init__()
            return

        # Keep at most self.capacity samples; preserve rough mate/draw proportion.
        if total > self.capacity:
            keep_m = int(round(self.capacity * (nm / total))) if total > 0 else 0
            keep_m = max(0, min(keep_m, nm))
            keep_d = self.capacity - keep_m
            keep_d = max(0, min(keep_d, nd))

            mate_obs = mate_obs[-keep_m:] if keep_m > 0 else mate_obs[:0]
            mate_pi = mate_pi[-keep_m:] if keep_m > 0 else mate_pi[:0]
            mate_wdl = mate_wdl[-keep_m:] if keep_m > 0 else mate_wdl[:0]

            draw_obs = draw_obs[-keep_d:] if keep_d > 0 else draw_obs[:0]
            draw_pi = draw_pi[-keep_d:] if keep_d > 0 else draw_pi[:0]
            draw_wdl = draw_wdl[-keep_d:] if keep_d > 0 else draw_wdl[:0]

            nm = int(mate_obs.shape[0])
            nd = int(draw_obs.shape[0])
            total = nm + nd

        # Allocate just enough (grow lazily later).
        self.__post_init__()
        target_alloc = int(max(self.initial_alloc, total))
        self._grow_to(min(self.capacity, target_alloc))

        # Fill
        pos = 0
        if nm > 0:
            self.obs[pos : pos + nm] = mate_obs.astype(np.uint8, copy=False)
            self.pi[pos : pos + nm] = mate_pi.astype(np.float16, copy=False)
            self.wdl[pos : pos + nm] = mate_wdl.astype(np.uint8, copy=False)
            self.is_mate[pos : pos + nm] = 1
            pos += nm

        if nd > 0:
            self.obs[pos : pos + nd] = draw_obs.astype(np.uint8, copy=False)
            self.pi[pos : pos + nd] = draw_pi.astype(np.float16, copy=False)
            self.wdl[pos : pos + nd] = draw_wdl.astype(np.uint8, copy=False)
            self.is_mate[pos : pos + nd] = 0
            pos += nd

        self.filled = total
        self.mate_count = nm
        self.draw_count = nd

        if self.filled == self.capacity:
            self.write_pos = 0
        else:
            self.write_pos = self.filled

