# agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
import collections
import queue
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import chess

from constants import *
from nw import AlphaZeroNet


# -----------------------------
# Observation encoding
# -----------------------------

def _softmax(x: 'np.ndarray') -> 'np.ndarray':
    """Stable softmax for 1-D numpy arrays."""
    x = x.astype(np.float64, copy=False)
    if x.size:
        x = x - float(np.max(x))
    e = np.exp(x)
    s = float(np.sum(e))
    if s <= 0.0:
        return np.ones_like(e, dtype=np.float64) / max(1, e.size)
    return (e / s).astype(np.float64, copy=False)

def board_to_obs(board: chess.Board, position_deque: Deque[chess.Board]) -> np.ndarray:
    """obs: (TOTAL_LAYERS, 8, 8) float32 (0/1 planes)

    Layout:
      - LAST_POSITIONS * 12 piece-planes (6 white + 6 black) for each of last positions
      - en-passant (1 plane)
      - castling rights (4 planes)
      - side to move (1 plane)

    TOTAL_LAYERS = 8*12 + 1 + 4 + 1 = 102
    """
    obs = np.zeros((TOTAL_LAYERS, 8, 8), dtype=np.float32)

    def put_pieces(dst_offset: int, b: chess.Board) -> None:
        # planes order: WP WN WB WR WQ WK BP BN BB BR BQ BK
        piece_to_plane = {
            chess.Piece(chess.PAWN, chess.WHITE): 0,
            chess.Piece(chess.KNIGHT, chess.WHITE): 1,
            chess.Piece(chess.BISHOP, chess.WHITE): 2,
            chess.Piece(chess.ROOK, chess.WHITE): 3,
            chess.Piece(chess.QUEEN, chess.WHITE): 4,
            chess.Piece(chess.KING, chess.WHITE): 5,
            chess.Piece(chess.PAWN, chess.BLACK): 6,
            chess.Piece(chess.KNIGHT, chess.BLACK): 7,
            chess.Piece(chess.BISHOP, chess.BLACK): 8,
            chess.Piece(chess.ROOK, chess.BLACK): 9,
            chess.Piece(chess.QUEEN, chess.BLACK): 10,
            chess.Piece(chess.KING, chess.BLACK): 11,
        }
        for square, piece in b.piece_map().items():
            p = piece_to_plane.get(piece)
            if p is None:
                continue
            r = 7 - chess.square_rank(square)
            c = chess.square_file(square)
            obs[dst_offset + p, r, c] = 1.0

    # last positions (including current)
    # NOTE: in self-play we append current board to deque before calling MCTS.
    # Here we iterate in chronological order, but any fixed order is OK as long as consistent with training.
    boards = list(position_deque)[-LAST_POSITIONS:]
    # pad with current board if history is shorter
    if len(boards) < LAST_POSITIONS:
        boards = [board] * (LAST_POSITIONS - len(boards)) + boards

    for i, b in enumerate(boards):
        put_pieces(i * BOARD_LAYERS, b)

    meta_off = LAST_POSITIONS * BOARD_LAYERS

    # en-passant
    if board.ep_square is not None:
        f = chess.square_file(board.ep_square)
        obs[meta_off + 0, :, f] = 1.0

    # castling rights
    obs[meta_off + 1, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    obs[meta_off + 2, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    obs[meta_off + 3, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    obs[meta_off + 4, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # turn
    obs[meta_off + 5, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    return obs


# -----------------------------
# Action encoding (move -> index)
# -----------------------------
# IMPORTANT: this matches your current trained network / checkpoints.
# It is a compact 4672-action encoding used in many AlphaZero-like chess repos:
#   - 64*73 "from-square + move-type"
# Your previous *_working.py already used this mapping; we keep it unchanged here.

_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to [0..TOTAL_MOVES)."""
    from_sq = move.from_square
    to_sq = move.to_square

    fr = chess.square_rank(from_sq)
    ff = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tf = chess.square_file(to_sq)

    dr = tr - fr
    df = tf - ff

    # Knight moves (8)
    knight_deltas = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1),
    ]
    if (dr, df) in knight_deltas:
        k = knight_deltas.index((dr, df))
        move_type = 56 + k  # after 56 queen-like moves
        return from_sq * 73 + move_type

    # Underpromotions (to N/B/R) on forward diagonals/files (3 directions * 3 pieces = 9)
    if move.promotion is not None and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        # direction from White's perspective; python-chess uses absolute squares
        # We just encode relative delta:
        # file delta: -1,0,+1 ; rank delta should be +1 for white promotions, -1 for black promotions.
        # We'll normalize by side-to-move not here (we encode absolute deltas).
        dir_map = {(-1): 0, 0: 1, 1: 2}
        if df not in dir_map:
            raise ValueError("bad promotion delta")
        p_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[move.promotion]
        move_type = 64 + dir_map[df] * 3 + p_idx
        return from_sq * 73 + move_type

    # Queen-like moves: 8 directions * up to 7 squares = 56
    dirs = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1),
    ]
    for d_idx, (d_r, d_f) in enumerate(dirs):
        for dist in range(1, 8):
            if dr == d_r * dist and df == d_f * dist:
                move_type = d_idx * 7 + (dist - 1)
                return from_sq * 73 + move_type

    # Queen promotion is encoded as normal queen-like move (promotion flag ignored).
    if move.promotion == chess.QUEEN:
        # This should have been matched above by queen-like delta.
        # If not, something is off.
        pass

    raise ValueError(f"Unsupported move encoding: {move}")


def policy_to_pi_vector(board: chess.Board, policy: Dict[chess.Move, float]) -> np.ndarray:
    """Dense π over TOTAL_MOVES with zeros for illegal moves."""
    pi = np.zeros((TOTAL_MOVES,), dtype=np.float32)
    s = 0.0
    for mv, p in policy.items():
        try:
            idx = move_to_index(mv)
        except Exception:
            continue
        pi[idx] += float(p)
        s += float(p)
    if s > 0:
        pi /= s
    else:
        # fallback: uniform over legal moves
        legal = list(board.legal_moves)
        if legal:
            for mv in legal:
                pi[move_to_index(mv)] = 1.0 / len(legal)
    return pi


# -----------------------------
# Central inference I/O
# -----------------------------
@dataclass
class InferenceResult:
    rid: int
    policy_logits: np.ndarray   # (TOTAL_MOVES,) float32
    value: float                # scalar float



class InferenceClient:
    """IPC client for self-play workers.

    Supports *multiple* in-flight inference requests per worker (needed for
    leaf-parallel MCTS). Requests go to a shared request queue; responses arrive
    on a per-worker response queue.

    Request payloads:
      - (worker_id, rid, obs)
      - (worker_id, rid, obs, legal_move_indices)                  if IPC_SEND_ONLY_LEGAL
      - (worker_id, rid, obs, model_id)                           if model_id != 0
      - (worker_id, rid, obs, legal_move_indices, model_id)       if IPC_SEND_ONLY_LEGAL and model_id != 0

    Where model_id is used by the main-process inference server to route requests
    to different models (e.g., candidate vs best during eval).

    If model_id == 0, the legacy 3/4-tuple formats are used for compatibility.

    Request payloads (legacy):
      - (worker_id, rid, obs)                       or
      - (worker_id, rid, obs, legal_move_indices)   if IPC_SEND_ONLY_LEGAL

    Response payload:
      - (rid, logits, value)

    Notes:
      - logits may be full (TOTAL_MOVES,) or a *subset* aligned with the passed
        legal_move_indices (if IPC_SEND_ONLY_LEGAL).
      - value is scalar from side-to-move perspective.
    """

    
    def __init__(
        self,
        request_q,
        response_q,
        worker_id: int,
        stop_event=None,
        *,
        model_id: int = 0,
        pause_event=None,
        pause_sleep_s: float = SELFPLAY_PAUSE_SLEEP_S,
        suppress_first_print: bool = False,
    ):
        self.request_q = request_q
        self.response_q = response_q
        self.worker_id = int(worker_id)
        self.stop_event = stop_event

        self.model_id = int(model_id)
        self.pause_event = pause_event
        self.pause_sleep_s = float(pause_sleep_s)
        self._suppress_first_print = bool(suppress_first_print)

        self._rid = 0
        self._stash: Dict[int, Tuple[np.ndarray, float]] = {}
        self._printed_first = False

        # Exposed counters for logging from self_play_game
        self.prof = {"calls": 0, "wait_s": 0.0}

    def _should_stop(self) -> bool:
        return bool(self.stop_event is not None and self.stop_event.is_set())

    def _next_rid(self) -> int:
        self._rid += 1
        return self._rid

    def submit(self, obs: np.ndarray, legal_idxs: Optional[List[int]] = None, *, _suppress_first_print: bool = False) -> int:
        """Submit one request and return rid (does not wait for response)."""
        if self._should_stop():
            raise KeyboardInterrupt

        # Cooperative pause (exclusive eval): block new submits while paused.
        if self.pause_event is not None:
            while self.pause_event.is_set():
                if self._should_stop():
                    raise KeyboardInterrupt
                time.sleep(self.pause_sleep_s)

        rid = self._next_rid()

        if IPC_SEND_ONLY_LEGAL and legal_idxs is None:
            if not getattr(self, "_warned_missing_legal_idxs", False):
                print(f"[worker {self.worker_id}] WARNING: IPC_SEND_ONLY_LEGAL=True but legal_idxs is None; sending full logits.", flush=True)
                self._warned_missing_legal_idxs = True

        if (not self._printed_first) and (not self._suppress_first_print) and (not _suppress_first_print):
            print(f"[worker {self.worker_id}] first inference request: batch=1", flush=True)
            self._printed_first = True

        # Optional request payload minimization
        obs_send = obs
        if IPC_OBS_UINT8:
            # obs is binary planes (0/1); send as uint8 to reduce IPC payload.
            # (safe even if obs is float32)
            obs_send = (obs_send > 0.0).astype(np.uint8, copy=False)

        if IPC_SEND_ONLY_LEGAL and legal_idxs is not None:
            idxs_send = np.asarray(legal_idxs, dtype=np.uint16)
            payload = (self.worker_id, int(rid), obs_send, idxs_send, self.model_id) if self.model_id != 0 else (self.worker_id, int(rid), obs_send, idxs_send)
        else:
            payload = (self.worker_id, int(rid), obs_send, self.model_id) if self.model_id != 0 else (self.worker_id, int(rid), obs_send)

        # Non-busy wait put loop (interrupt-friendly)
        while True:
            try:
                self.request_q.put(payload, block=True, timeout=INFER_CLIENT_PUT_TIMEOUT_S)
                break
            except queue.Full:
                if self._should_stop():
                    raise KeyboardInterrupt

        self.prof["calls"] += 1
        return rid

    def wait(self, rid: int) -> Tuple[np.ndarray, float]:
        """Wait for (logits, value) for the given rid (handles out-of-order)."""
        if rid in self._stash:
            logits, value = self._stash.pop(rid)
            return logits, float(value)

        t0 = time.perf_counter()
        warned_at = 0.0

        while True:
            if self._should_stop():
                raise KeyboardInterrupt

            try:
                msg = self.response_q.get(block=True, timeout=INFER_CLIENT_GET_TIMEOUT_S)
            except queue.Empty:
                # Optional latency warning (helps catch deadlocks / slowdowns)
                if WORKER_INFER_WARN_AFTER_S > 0:
                    waited = time.perf_counter() - t0
                    if waited >= WORKER_INFER_WARN_AFTER_S and waited - warned_at >= WORKER_INFER_WARN_AFTER_S:
                        warned_at = waited
                        print(f"[worker {self.worker_id}] WARN: inference wait {waited:.3f}s (rid={rid})", flush=True)
                continue

            # Server may send either a single (rid, logits, value) tuple, or a batch list of such tuples.
            items = msg if isinstance(msg, list) else [msg]

            for j, it in enumerate(items):
                try:
                    rrid, logits, value = it
                except Exception:
                    continue
                rrid_i = int(rrid)
                if rrid_i == int(rid):
                    self.prof["wait_s"] += (time.perf_counter() - t0)
                    # Stash any additional results that arrived in the same message.
                    for it2 in items[j+1:]:
                        try:
                            rrid2, logits2, value2 = it2
                        except Exception:
                            continue
                        self._stash[int(rrid2)] = (logits2, float(value2))
                    return logits, float(value)
            
                # out-of-order: stash
                self._stash[rrid_i] = (logits, float(value))

            # Not found in this message (maybe stashed already)
            if rid in self._stash:
                logits, value = self._stash.pop(rid)
                self.prof["wait_s"] += (time.perf_counter() - t0)
                return logits, float(value)

    def infer(self, obs: np.ndarray, legal_idxs: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        rid = self.submit(obs, legal_idxs)
        return self.wait(rid)

    def infer_many(self, obs_list: List[np.ndarray], legal_idxs_list: Optional[List[Optional[List[int]]]] = None) -> List[Tuple[np.ndarray, float]]:
        """Submit many requests, then wait them back in the same order."""
        if not obs_list:
            return []

        if legal_idxs_list is None:
            legal_idxs_list = [None] * len(obs_list)
        assert len(legal_idxs_list) == len(obs_list)

        if not self._printed_first:
            print(f"[worker {self.worker_id}] first inference request: batch={len(obs_list)}", flush=True)
            self._printed_first = True

        rids: List[int] = []
        for obs, idxs in zip(obs_list, legal_idxs_list):
            rids.append(self.submit(obs, idxs, _suppress_first_print=True))

        out: List[Tuple[np.ndarray, float]] = []
        for rid in rids:
            out.append(self.wait(rid))
        return out


@dataclass(slots=True)
class Node:
    prior: float
    value_sum: float = 0.0
    visit_count: int = 0
    vloss: int = 0
    children: Optional[Dict[chess.Move, "Node"]] = None

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTS:
    """AlphaZero-style MCTS with leaf-parallelism when infer_client is provided."""

    def __init__(self, model: AlphaZeroNet, infer_client: Optional[InferenceClient] = None, *, simulations: Optional[int] = None):
        self.model = model
        self.infer_client = infer_client
        self.simulations = int(simulations) if simulations is not None else int(MCTS_SIMULATIONS)

        self._root_hash: Optional[str] = None
        self._root: Optional[Node] = None

    def reset(self) -> None:
        self._root_hash = None
        self._root = None

    def advance_root(self, board_before: chess.Board, move_played: chess.Move) -> None:
        """Reuse subtree after playing a move (optional optimization)."""
        if self._root is None or self._root_hash != board_before.fen():
            self._root = None
            self._root_hash = None
            return

        root = self._root
        if root.children is None:
            self._root = None
            self._root_hash = None
            return

        child = root.children.get(move_played)
        if child is None:
            self._root = None
            self._root_hash = None
            return

        # detach and promote
        self._root = child
        b = board_before.copy(stack=False)
        b.push(move_played)
        self._root_hash = b.fen()

    def _attach_children(
        self,
        node: Node,
        legal_moves: List[chess.Move],
        logits: np.ndarray,
        legal_idxs: Optional[List[int]] = None,
    ) -> None:
        if not legal_moves:
            node.children = {}
            return

        # If server sent only legal logits: logits aligns with legal_moves / legal_idxs ordering.
        if IPC_SEND_ONLY_LEGAL and legal_idxs is not None and logits.shape[0] == len(legal_moves):
            logits_sub = logits.astype(np.float32, copy=False)
        else:
            if legal_idxs is None:
                legal_idxs = [move_to_index(mv) for mv in legal_moves]
            logits_sub = logits[np.asarray(legal_idxs, dtype=np.int32)]

        priors = _softmax(logits_sub)
        children: Dict[chess.Move, Node] = {}
        for mv, p in zip(legal_moves, priors):
            children[mv] = Node(prior=float(p), children=None)
        node.children = children

    def _expand(self, node: Node, board: chess.Board, position_deque: Deque[chess.Board]) -> float:
        """Attach children to *this* node and return NN value (side-to-move)."""
        obs = board_to_obs(board, position_deque)
        legal = list(board.legal_moves)
        if not legal:
            node.children = {}
            return 0.0

        if self.infer_client is not None:
            legal_idxs = [move_to_index(mv) for mv in legal] if IPC_SEND_ONLY_LEGAL else None
            logits, value = self.infer_client.infer(obs, legal_idxs)
        else:
            with torch.no_grad():
                device = next(self.model.parameters()).device
                x = torch.from_numpy(obs).unsqueeze(0).to(device)
                out = self.model(x)
                logits = out[0].squeeze(0).detach().cpu().numpy()
                wdl_logits = out[2].squeeze(0).detach().cpu().numpy()
                probs = np.exp(wdl_logits - np.max(wdl_logits))
                probs = probs / (np.sum(probs) + 1e-8)
                value = float(probs[0] - probs[2] + CONTEMPT_DRAW * probs[1])

            legal_idxs = [move_to_index(mv) for mv in legal]
        self._attach_children(node, legal, logits, legal_idxs if IPC_SEND_ONLY_LEGAL else None)
        return float(value)

    def run(
        self,
        board: chess.Board,
        position_deque: Deque[chess.Board],
        *,
        add_dirichlet_noise: bool = False,
        reuse_tree: bool = True,
    ) -> Tuple[Dict[chess.Move, float], Dict[chess.Move, int]]:
        """Run MCTS and return (policy_dict, visit_counts_dict)."""
        h = board.fen()
        if not (reuse_tree and self._root_hash == h and self._root is not None):
            self._root = Node(prior=1.0, children=None)
            self._root_hash = h

        root = self._root

        # Expand root once
        if root.children is None:
            _ = self._expand(root, board, position_deque)

        # Dirichlet noise (root only)
        if add_dirichlet_noise and root.children:
            moves = list(root.children.keys())
            priors = np.array([root.children[m].prior for m in moves], dtype=np.float32)
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves)).astype(np.float32)
            priors = (1.0 - DIRICHLET_EPS) * priors + DIRICHLET_EPS * noise
            priors = priors / (priors.sum() + 1e-8)
            for mv, p in zip(moves, priors):
                root.children[mv].prior = float(p)

        leaf_parallel = int(MCTS_LEAF_PARALLELISM) if (self.infer_client is not None) else 1
        leaf_parallel = max(1, leaf_parallel)

        sims_done = 0
        while sims_done < self.simulations:
            k = min(leaf_parallel, self.simulations - sims_done)

            tasks: List[Tuple[Node, chess.Board, Deque[chess.Board], List[Node]]] = []
            # 1) select k leaves (with virtual loss so we diversify paths)
            for _ in range(k):
                b = board.copy(stack=False)
                hist: Deque[chess.Board] = collections.deque(position_deque, maxlen=LAST_POSITIONS)

                node = root
                path: List[Node] = [node]

                while node.children is not None and len(node.children) > 0:
                    mv, node = self._select_child(node)
                    b.push(mv)
                    hist.append(b.copy(stack=False))
                    path.append(node)

                if leaf_parallel > 1:
                    for n in path:
                        n.vloss += 1

                tasks.append((node, b, hist, path))

            values: List[Optional[float]] = [None] * k
            req_nodes: List[Tuple[int, Node, List[chess.Move], Optional[List[int]], np.ndarray]] = []

            # 2) terminal check & build inference batch
            for i, (node, b, hist, path) in enumerate(tasks):
                outcome = b.outcome(claim_draw=True)
                if outcome is not None:
                    if outcome.winner is None:
                        values[i] = 0.0
                    else:
                        values[i] = 1.0 if outcome.winner == b.turn else -1.0
                    continue

                if node.children is None:
                    obs = board_to_obs(b, hist)
                    legal = list(b.legal_moves)
                    if self.infer_client is not None and IPC_SEND_ONLY_LEGAL:
                        legal_idxs = [move_to_index(mv) for mv in legal]
                    else:
                        legal_idxs = None
                    req_nodes.append((i, node, legal, legal_idxs, obs))
                else:
                    # leaf is already expanded but has no children (rare)
                    values[i] = 0.0

            # 3) run batched inference (many in-flight)
            if req_nodes:
                if self.infer_client is None:
                    # Local inference fallback (rare path)
                    with torch.no_grad():
                        device = next(self.model.parameters()).device
                        obs_batch = np.stack([x[-1] for x in req_nodes], axis=0).astype(np.float32, copy=False)
                        x = torch.from_numpy(obs_batch).to(device)
                        out = self.model(x)
                        logits_b = out[0].detach().cpu().numpy()
                        wdl_b = out[2].detach().cpu().numpy()
                        # map WDL -> scalar value (side-to-move)
                        probs = np.exp(wdl_b - np.max(wdl_b, axis=1, keepdims=True))
                        probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-8)
                        values_b = probs[:, 0] - probs[:, 2] + CONTEMPT_DRAW * probs[:, 1]

                    for row, (i, node, legal, legal_idxs, _obs) in enumerate(req_nodes):
                        self._attach_children(node, legal, logits_b[row], legal_idxs)
                        values[i] = float(values_b[row])
                else:
                    # Submit all first
                    rids: List[int] = []
                    for (i, node, legal, legal_idxs, obs) in req_nodes:
                        rids.append(self.infer_client.submit(obs, legal_idxs, _suppress_first_print=True))

                    # Wait and attach
                    for (info, rid) in zip(req_nodes, rids):
                        i, node, legal, legal_idxs, _obs = info
                        logits, value = self.infer_client.wait(rid)
                        self._attach_children(node, legal, logits, legal_idxs)
                        values[i] = float(value)

            # 4) backprop each simulation & clear vloss
            for i, (_node, _b, _hist, path) in enumerate(tasks):
                if leaf_parallel > 1:
                    for n in path:
                        n.vloss -= 1

                value = float(values[i] if values[i] is not None else 0.0)
                for n in reversed(path):
                    n.visit_count += 1
                    n.value_sum += value
                    value = -value

            sims_done += k

        # Build policy from visit counts
        if root.children is None or len(root.children) == 0:
            return {}, {}

        visit_counts = {m: c.visit_count for m, c in root.children.items()}
        total = sum(visit_counts.values())
        if total <= 0:
            policy = {m: 1.0 / len(visit_counts) for m in visit_counts} if visit_counts else {}
        else:
            policy = {m: vc / total for m, vc in visit_counts.items()}

        return policy, visit_counts

    def _select_child(self, node: Node) -> Tuple[chess.Move, Node]:
        """PUCT selection from parent perspective.

        child.value() is from side-to-move at child (opponent relative to parent),
        so parent-Q uses -child.value().
        """
        assert node.children is not None and len(node.children) > 0

        best_score = -1e9
        best_move = None
        best_child = None

        # Include virtual loss in exploration term so multiple in-flight leaves diversify.
        parent_n = node.visit_count + node.vloss
        sqrt_visits = math.sqrt(parent_n + 1.0)

        for mv, child in node.children.items():
            child_n = child.visit_count + child.vloss
            q_parent = -child.value()
            u = MCTS_C_PUCT * child.prior * (sqrt_visits / (1.0 + child_n))
            score = q_parent + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child

        return best_move, best_child



def apply_temperature(probs: np.ndarray, tau: float) -> np.ndarray:
    probs = probs.astype(np.float64, copy=False)
    probs = probs / (probs.sum() + 1e-12)
    if tau <= 1e-6:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out.astype(np.float32)
    p = np.power(probs, 1.0 / tau)
    p = p / (p.sum() + 1e-12)
    return p.astype(np.float32)


# -----------------------------
# Self-play
# -----------------------------

def temperature_tau(move_idx: int) -> float:
    if move_idx >= TEMPERATURE_MOVES:
        return 0.0
    # linear decay
    if TEMPERATURE_MOVES <= 1:
        return TEMPERATURE_TAU_END
    t = move_idx / max(1, (TEMPERATURE_MOVES - 1))
    return float(TEMPERATURE_TAU_START * (1.0 - t) + TEMPERATURE_TAU_END * t)

# Backward-compat alias for a common typo
try:
    temperateru_tau
except NameError:
    def temperateru_tau(move_idx: int) -> float:
        return temperature_tau(move_idx)

def self_play_game(
    model: AlphaZeroNet,
    infer_client: InferenceClient,
    worker_id: int,
    *,
    max_moves: int = MAX_GAME_LENGTH,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], chess.Outcome, int, str]:
    """Play one game with MCTS+Dirichlet noise. Returns (samples, outcome, moves_cnt, draw_reason).

    samples items: (obs, pi_vec, wdl_label)
      wdl_label is from side-to-move at that obs:
        0=WIN, 1=DRAW, 2=LOSS
    """
    board = chess.Board()
    position_deque: Deque[chess.Board] = collections.deque(maxlen=LAST_POSITIONS)

    mcts = MCTS(model, infer_client=infer_client)

    traj: List[Tuple[np.ndarray, np.ndarray, bool]] = []  # (obs, pi, player_turn)

    moves_cnt = 0
    prof = {'mcts_s': 0.0, 'infer_wait_s': 0.0, 'infer_calls': 0, 'moves': 0}
    while not board.is_game_over(claim_draw=True) and moves_cnt < max_moves:
        position_deque.append(board.copy(stack=False))

        _mcts_t0 = time.perf_counter()

        _c0 = infer_client.prof.get('calls', 0)

        _w0 = infer_client.prof.get('wait_s', 0.0)

        policy, _visit_counts = mcts.run(
            board,
            position_deque,
            add_dirichlet_noise=True,
            reuse_tree=True,
        )

        _dt = time.perf_counter() - _mcts_t0
        prof['mcts_s'] += _dt
        prof['infer_calls'] += infer_client.prof.get('calls', 0) - _c0
        prof['infer_wait_s'] += infer_client.prof.get('wait_s', 0.0) - _w0
        # Build π
        pi_vec = policy_to_pi_vector(board, policy)

        # Choose action
        if moves_cnt < TEMPERATURE_MOVES:
            moves_list = list(policy.keys())
            probs = np.array([policy[m] for m in moves_list], dtype=np.float32)
            tau = temperature_tau(moves_cnt)
            probs = apply_temperature(probs, tau=tau)
            move = np.random.choice(moves_list, p=probs)
        else:
            move = max(policy.items(), key=lambda kv: kv[1])[0] if policy else random.choice(list(board.legal_moves))

        obs = board_to_obs(board, position_deque)
        traj.append((obs, pi_vec, board.turn))

        # Advance board + tree
        board_before = board.copy(stack=False)
        board.push(move)
        mcts.advance_root(board_before, move)

        moves_cnt += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Forced stop -> treat as draw with explicit reason
        outcome = chess.Outcome(termination=chess.Termination.VARIANT_DRAW, winner=None)

    draw_reason = ""
    if outcome.winner is None:
        draw_reason = str(outcome.termination)

    # Convert trajectory into labeled samples
    samples: List[Tuple[np.ndarray, np.ndarray, int]] = []
    if outcome.winner is None:
        wdl_all = 1  # DRAW
        for obs, pi_vec, _player in traj:
            samples.append((obs, pi_vec, wdl_all))
    else:
        for obs, pi_vec, player in traj:
            if outcome.winner == player:
                wdl = 0  # WIN for side-to-move at that position
            else:
                wdl = 2  # LOSS
            samples.append((obs, pi_vec, wdl))

    # One-shot worker print: first finished game
    if DEBUG_PRINT_WORKER_FIRST_GAME and worker_id >= 0:
        # Worker process can keep its own state externally; we don't store it here.
        pass

    prof['moves'] = moves_cnt

    return samples, outcome, moves_cnt, draw_reason, prof


# -----------------------------
# Training
# -----------------------------
def train_one_step(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
) -> Dict[str, float]:
    """One SGD step. Returns losses for logging."""
    model.train()
    obs, pi, wdl = batch  # obs: (B,C,8,8), pi:(B,A), wdl:(B,)
    obs = obs.to(device, non_blocking=True)
    pi = pi.to(device, non_blocking=True)
    wdl = wdl.to(device, non_blocking=True)

    out = model(obs)
    pred_pi = out[0]
    wdl_logits = out[2]

    # policy loss: cross-entropy with target distribution π
    logp = torch.log_softmax(pred_pi, dim=1)
    loss_pi = -(pi * logp).sum(dim=1).mean()

    # value loss: CE over WDL classes
    loss_v = torch.nn.functional.cross_entropy(wdl_logits, wdl)

    loss = loss_pi + loss_v

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(loss_pi.detach().cpu().item()),
        "value_loss": float(loss_v.detach().cpu().item()),
    }