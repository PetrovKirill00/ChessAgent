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
    capacity: int = REPLAY_CAPACITY

    def __post_init__(self) -> None:
        self.mate_obs = np.zeros((0, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        self.mate_pi = np.zeros((0, TOTAL_MOVES), dtype=np.float32)
        self.mate_wdl = np.zeros((0,), dtype=np.uint8)

        self.draw_obs = np.zeros((0, TOTAL_LAYERS, 8, 8), dtype=np.uint8)
        self.draw_pi = np.zeros((0, TOTAL_MOVES), dtype=np.float32)
        self.draw_wdl = np.zeros((0,), dtype=np.uint8)

    # -----------------------------
    # Sizes / stats
    # -----------------------------
    def sizes(self) -> Dict[str, int]:
        return {
            "mate": int(self.mate_obs.shape[0]),
            "draw": int(self.draw_obs.shape[0]),
            "total": int(self.mate_obs.shape[0] + self.draw_obs.shape[0]),
        }

    # -----------------------------
    # Add / sample
    # -----------------------------
    def add(self, obs: np.ndarray, pi: np.ndarray, wdl: int, is_mate: bool) -> None:
        """Add a single (obs, pi, wdl_label) sample into mate/draw partition."""
        obs_u8 = _as_uint8_obs(obs)
        pi_f = pi.astype(np.float32, copy=False)
        wdl_u8 = np.uint8(int(wdl))

        if is_mate:
            self.mate_obs = np.concatenate([self.mate_obs, obs_u8[None]], axis=0)
            self.mate_pi = np.concatenate([self.mate_pi, pi_f[None]], axis=0)
            self.mate_wdl = np.concatenate([self.mate_wdl, np.array([wdl_u8], dtype=np.uint8)], axis=0)
        else:
            self.draw_obs = np.concatenate([self.draw_obs, obs_u8[None]], axis=0)
            self.draw_pi = np.concatenate([self.draw_pi, pi_f[None]], axis=0)
            self.draw_wdl = np.concatenate([self.draw_wdl, np.array([wdl_u8], dtype=np.uint8)], axis=0)

        self._trim()

    def add_game(self, samples, *, is_mate: bool) -> None:
        """Add many samples from one game, assuming they share the same is_mate flag."""
        for obs, pi, wdl in samples:
            self.add(obs, pi, int(wdl), is_mate=is_mate)

    def _trim(self) -> None:
        """Keep total capacity bounded (simple proportional trimming)."""
        total = self.sizes()["total"]
        if total <= self.capacity:
            return

        # Trim both partitions proportionally but keep at least 1 element if present
        excess = total - self.capacity
        # Trim from the larger partition first
        for _ in range(excess):
            if self.draw_obs.shape[0] >= self.mate_obs.shape[0] and self.draw_obs.shape[0] > 0:
                self.draw_obs = self.draw_obs[1:]
                self.draw_pi = self.draw_pi[1:]
                self.draw_wdl = self.draw_wdl[1:]
            elif self.mate_obs.shape[0] > 0:
                self.mate_obs = self.mate_obs[1:]
                self.mate_pi = self.mate_pi[1:]
                self.mate_wdl = self.mate_wdl[1:]
            else:
                break

    def sample(self, batch_size: int, *, p_mate: float, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        sizes = self.sizes()
        if sizes["total"] < batch_size:
            return None

        n_mate = int(round(batch_size * p_mate))
        n_draw = batch_size - n_mate

        # If one partition is empty, fall back to the other.
        if sizes["mate"] == 0:
            n_mate, n_draw = 0, batch_size
        if sizes["draw"] == 0:
            n_mate, n_draw = batch_size, 0

        idx_m = np.random.randint(0, sizes["mate"], size=n_mate) if n_mate > 0 else np.array([], dtype=np.int64)
        idx_d = np.random.randint(0, sizes["draw"], size=n_draw) if n_draw > 0 else np.array([], dtype=np.int64)

        obs = []
        pi = []
        wdl = []

        if n_mate > 0:
            obs.append(self.mate_obs[idx_m])
            pi.append(self.mate_pi[idx_m])
            wdl.append(self.mate_wdl[idx_m])
        if n_draw > 0:
            obs.append(self.draw_obs[idx_d])
            pi.append(self.draw_pi[idx_d])
            wdl.append(self.draw_wdl[idx_d])

        obs = np.concatenate(obs, axis=0)
        pi = np.concatenate(pi, axis=0)
        wdl = np.concatenate(wdl, axis=0)

        # Shuffle mixed batch
        perm = np.random.permutation(batch_size)
        obs = obs[perm]
        pi = pi[perm]
        wdl = wdl[perm]

        obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32, non_blocking=True)
        pi_t = torch.from_numpy(pi).to(device=device, dtype=torch.float32, non_blocking=True)
        wdl_t = torch.from_numpy(wdl.astype(np.int64)).to(device=device, dtype=torch.long, non_blocking=True)

        return obs_t, pi_t, wdl_t

    # -----------------------------
    # Save / load
    # -----------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            version=np.int32(3),
            mate_obs=self.mate_obs,
            mate_pi=self.mate_pi,
            mate_wdl=self.mate_wdl,
            draw_obs=self.draw_obs,
            draw_pi=self.draw_pi,
            draw_wdl=self.draw_wdl,
        )

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False

        try:
            data = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"[replay] failed to load '{path}': {e}")
            return False

        # Hard stop on legacy buffers (we intentionally removed auto-migration).
        if "version" not in data:
            print("[replay] legacy buffer detected (no version). Please delete the file and restart to rebuild.")
            return False

        version = int(data["version"])
        if version != 3:
            print(f"[replay] legacy buffer version={version}. Please delete the file and restart to rebuild.")
            return False

        required = ["mate_obs", "mate_pi", "mate_wdl", "draw_obs", "draw_pi", "draw_wdl"]
        for k in required:
            if k not in data:
                print(f"[replay] invalid buffer: missing key '{k}'. Delete and rebuild.")
                return False

        self.mate_obs = data["mate_obs"].astype(np.uint8, copy=False)
        self.mate_pi = data["mate_pi"].astype(np.float32, copy=False)
        self.mate_wdl = data["mate_wdl"].astype(np.uint8, copy=False)

        self.draw_obs = data["draw_obs"].astype(np.uint8, copy=False)
        self.draw_pi = data["draw_pi"].astype(np.float32, copy=False)
        self.draw_wdl = data["draw_wdl"].astype(np.uint8, copy=False)

        return True
