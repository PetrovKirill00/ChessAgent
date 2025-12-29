# agent.py
# -*- coding: utf-8 -*-
"""Self-play, MCTS and evaluation utilities.

This file is intentionally self-contained: it includes
- observation encoding (board -> 102x8x8)
- action encoding (move -> index in [0..4671])
- MCTS (with transposition table + optional batched inference)
- self-play game generation
- single training step (policy + value losses)
- candidate-vs-baseline evaluation used for gating.

The *_working.py version had a correct MCTS (tree actually grows).
The previous "new" version regressed into a 1-ply search because
leaf expansions were not attached to the tree. This file restores the
working MCTS logic and keeps the newer gating metric (score with draw=0.5).
"""

from __future__ import annotations

import math
import random
import collections
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import chess
import chess.polyglot

from constants import (
    # encoding
    TOTAL_LAYERS, TOTAL_MOVES, LAST_POSITIONS,
    # mcts
    MCTS_SIMULATIONS, MCTS_C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPS, INFERENCE_BATCH_SIZE,
    TEMPERATURE_MOVES, TEMPERATURE_TAU_START, TEMPERATURE_TAU_END,
    # training shaping
    THREEFOLD,
    # eval/gating
    EVAL_SCORE_Z, EVAL_MAX_GAME_LENGTH, DEBUG_PRINT_EVAL_TERMINATIONS, ELO_FROM_SCORE_EPS,
    # optimization
    GRAD_CLIP_NORM,
)

from nw import AlphaZeroNet


# -----------------------------
# Observation encoding
# -----------------------------
_PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_planes(board: chess.Board) -> np.ndarray:
    """12x8x8 piece planes (absolute board coordinates).

    Plane order:
      0..5   : white [P,N,B,R,Q,K]
      6..11  : black [P,N,B,R,Q,K]
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = 7 - chess.square_rank(sq)   # rank 8 -> row 0
        c = chess.square_file(sq)       # file a -> col 0
        base = _PIECE_TO_PLANE[piece.piece_type]
        idx = base + (0 if piece.color == chess.WHITE else 6)
        planes[idx, r, c] = 1.0
    return planes


def board_to_obs(board: chess.Board, history: Deque[np.ndarray]) -> np.ndarray:
    """Build the full 102x8x8 observation.

    `history` stores 12x8x8 planes of the last positions (including current).
    """
    # Take the last LAST_POSITIONS entries; pad with zeros on the left if needed.
    hist = list(history)[-LAST_POSITIONS:]
    if len(hist) < LAST_POSITIONS:
        pad = [np.zeros((12, 8, 8), dtype=np.float32) for _ in range(LAST_POSITIONS - len(hist))]
        hist = pad + hist

    stack = np.concatenate(hist, axis=0)  # (LAST_POSITIONS*12, 8, 8)

    # Side to move
    stm = np.full((1, 8, 8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32)

    # Castling rights (WK, WQ, BK, BQ)
    cr = np.zeros((4, 8, 8), dtype=np.float32)
    cr[0, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    cr[1, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    cr[2, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    cr[3, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En-passant (one-hot square, else all zeros)
    ep = np.zeros((1, 8, 8), dtype=np.float32)
    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        ep[0, r, c] = 1.0

    obs = np.concatenate([stack, stm, cr, ep], axis=0)

    # Hard-guard: the network stem expects TOTAL_LAYERS channels.
    if obs.shape[0] != TOTAL_LAYERS:
        if obs.shape[0] < TOTAL_LAYERS:
            obs = np.concatenate([obs, np.zeros((TOTAL_LAYERS - obs.shape[0], 8, 8), dtype=np.float32)], axis=0)
        else:
            obs = obs[:TOTAL_LAYERS]
    return obs.astype(np.float32, copy=False)


# -----------------------------
# Action encoding (8x8x73 = 4672)
# -----------------------------
# 56 queen-like: 8 directions * 7 steps
_Q_DIRS = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]

# 8 knight moves
_KNIGHT_DELTAS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# AlphaZero 73 planes convention:
#   - planes 0..55: queen-like moves (any piece, incl. normal queen promotions)
#   - planes 56..63: knight moves
#   - planes 64..72: UNDERpromotions only (N,B,R) * (forward, capture-left, capture-right)
_UNDERPROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
_UNDERPROMO_DIRS_REL = [(1, 0), (1, -1), (1, 1)]  # relative to side-to-move


def move_to_index(board: chess.Board, move: chess.Move) -> int:
    """Encode a move into [0..4671].

    Important: for underpromotions we use direction *relative* to side-to-move,
    otherwise black underpromotions would not fit into the 73-plane scheme.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    fr = chess.square_rank(from_sq)
    fc = chess.square_file(from_sq)
    tr = chess.square_rank(to_sq)
    tc = chess.square_file(to_sq)

    dr = tr - fr
    dc = tc - fc

    plane: Optional[int] = None

    # Queen-like (including king moves, pawn pushes/captures, queen promotions).
    if dr == 0 or dc == 0 or abs(dr) == abs(dc):
        step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
        step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
        if (step_r, step_c) in _Q_DIRS:
            steps = max(abs(dr), abs(dc))
            if 1 <= steps <= 7:
                dir_idx = _Q_DIRS.index((step_r, step_c))
                plane = dir_idx * 7 + (steps - 1)  # 0..55

    # Knight moves
    if plane is None and (dr, dc) in _KNIGHT_DELTAS:
        plane = 56 + _KNIGHT_DELTAS.index((dr, dc))  # 56..63

    # Underpromotions: 9 extra planes.
    if move.promotion is not None and move.promotion in _UNDERPROMO_PIECES:
        # Make the delta relative to side-to-move (so "forward" is always +1 rank).
        rel_dr, rel_dc = dr, dc
        if board.turn == chess.BLACK:
            rel_dr, rel_dc = -dr, -dc

        # rel_dr must be +1 for promotions.
        # (If it's not, we still guard to avoid crashing on corrupted moves.)
        dir_key = (1, 0) if rel_dc == 0 else (1, -1) if rel_dc < 0 else (1, 1)
        if rel_dr == 1 and dir_key in _UNDERPROMO_DIRS_REL:
            dir_i = _UNDERPROMO_DIRS_REL.index(dir_key)  # 0..2
            piece_i = _UNDERPROMO_PIECES.index(move.promotion)  # 0..2
            plane = 64 + dir_i * 3 + piece_i  # 64..72

    if plane is None:
        # This should not happen for legal moves in our 73-plane scheme.
        # Keep a deterministic fallback instead of crashing a whole worker.
        plane = 0

    index = (fr * 8 + fc) * 73 + plane
    return int(index)


def policy_to_pi_vector(board: chess.Board, policy: Dict[chess.Move, float]) -> np.ndarray:
    """Convert visit-count policy dict into a normalized pi vector (4672,)."""
    pi = np.zeros((TOTAL_MOVES,), dtype=np.float32)
    if not policy:
        return pi

    for mv, v in policy.items():
        pi[move_to_index(board, mv)] += float(v)

    s = float(pi.sum())
    if s > 0:
        pi /= s
    return pi


# -----------------------------
# Temperature helper
# -----------------------------
def temperature_tau(ply: int) -> float:
    # Linear decay from start to end for the first TEMPERATURE_MOVES plies.
    if TEMPERATURE_MOVES <= 1:
        return float(TEMPERATURE_TAU_END)
    t = min(max(ply, 0), TEMPERATURE_MOVES - 1) / float(TEMPERATURE_MOVES - 1)
    return float(TEMPERATURE_TAU_START * (1.0 - t) + TEMPERATURE_TAU_END * t)


def apply_temperature(probs: np.ndarray, tau: float) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 1e-12)
    if tau <= 1e-6:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out.astype(np.float64)

    # p_i^(1/tau) / sum
    p = probs ** (1.0 / float(tau))
    p /= p.sum()
    return p


# -----------------------------
# Inference (worker-side)
# -----------------------------
class InferenceClient:
    """Thin client over multiprocessing queues.

    Protocol (per request):
      request_q:  (rid:int, obs:np.ndarray[float32,C,8,8])
      response_q: (rid:int, logits:np.ndarray[float32,4672], value:float)
    """

    def __init__(self, request_q, response_q, worker_id: int):
        self.request_q = request_q
        self.response_q = response_q
        self.worker_id = int(worker_id)
        self._rid = 0
        self._first = True

    def _next_rid(self) -> int:
        rid = (self.worker_id << 48) | (self._rid & ((1 << 48) - 1))
        self._rid += 1
        return rid

    def infer(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        rid = self._next_rid()
        if self._first:
            # Helpful for debugging: proves the worker->server path is alive.
            self._first = False

        self.request_q.put((rid, obs), block=True)

        while True:
            r2, logits, value = self.response_q.get(block=True)
            if r2 == rid:
                return logits, float(value)

    def infer_many(self, obs_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Send many requests and wait for all replies (order preserved)."""
        if not obs_list:
            return np.zeros((0, TOTAL_MOVES), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        rids = [self._next_rid() for _ in obs_list]
        for rid, obs in zip(rids, obs_list):
            self.request_q.put((rid, obs), block=True)

        got: Dict[int, Tuple[np.ndarray, float]] = {}
        while len(got) < len(rids):
            r2, logits, value = self.response_q.get(block=True)
            if r2 in got:
                continue
            got[r2] = (logits, float(value))

        logits_batch = np.stack([got[r][0] for r in rids], axis=0).astype(np.float32, copy=False)
        values = np.asarray([got[r][1] for r in rids], dtype=np.float32)
        return logits_batch, values


# -----------------------------
# MCTS (transposition-table based)
# -----------------------------
class Node:
    def __init__(self, player_to_move: bool):
        self.player_to_move = bool(player_to_move)
        self.number_of_visits: int = 0
        self.value_sum: float = 0.0
        self.mean_value: float = 0.0

        # move -> (child_hash, prior)
        self.children: Dict[chess.Move, Tuple[int, float]] = {}


class MCTS:
    def __init__(
        self,
        *,
        model: Optional[AlphaZeroNet] = None,
        inference_client: Optional[InferenceClient] = None,
        device: str | torch.device = "cpu",
        c_puct: float = MCTS_C_PUCT,
        inference_batch_size: int = INFERENCE_BATCH_SIZE,
    ):
        assert (model is not None) or (inference_client is not None)

        self.model = model
        self.inference_client = inference_client
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.c_puct = float(c_puct)
        self.inference_batch_size = int(inference_batch_size)

        # root tracking for reuse_tree
        self._root_hash: Optional[int] = None

        # transposition table: hash -> Node
        self.ttable: Dict[int, Node] = {}

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    @staticmethod
    def _board_hash(board: chess.Board) -> int:
        return int(chess.polyglot.zobrist_hash(board))

    def _get_or_create_node(self, state_hash: int, player_to_move: bool) -> Node:
        node = self.ttable.get(state_hash)
        if node is None:
            node = Node(player_to_move=player_to_move)
            self.ttable[state_hash] = node
        return node

    def _ucb_score(self, parent: Node, child: Node, prior: float) -> float:
        q = 0.0 if child.number_of_visits == 0 else child.mean_value
        u = self.c_puct * float(prior) * math.sqrt(parent.number_of_visits + 1e-8) / (1.0 + child.number_of_visits)
        return q + u

    def _infer_many(self, obs_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (policy_logits[B,4672], values[B])."""
        if self.inference_client is not None:
            return self.inference_client.infer_many(obs_list)

        assert self.model is not None
        with torch.no_grad():
            x = torch.from_numpy(np.stack(obs_list, axis=0)).to(self.device)
            pol, val, _ = self.model(x)
            pol = pol.detach().cpu().numpy().astype(np.float32)
            val = val.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return pol, val

    def _terminal_value(self, board: chess.Board, player_to_move: bool) -> Optional[float]:
        outcome = board.outcome(claim_draw=THREEFOLD)
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        # Value from the perspective of the side to move at this node.
        return 1.0 if outcome.winner == player_to_move else -1.0

    def _expand_from_logits(self, node: Node, board: chess.Board, logits: np.ndarray):
        """Populate node.children using NN logits (softmax over legal moves)."""
        legal = list(board.legal_moves)
        if not legal:
            return

        idxs = [move_to_index(board, mv) for mv in legal]
        l = logits[np.asarray(idxs, dtype=np.int64)].astype(np.float64)

        # Softmax (stable)
        l = l - float(np.max(l))
        p = np.exp(l)
        s = float(p.sum())
        if s <= 0.0 or not np.isfinite(s):
            p = np.ones_like(p) / float(len(p))
        else:
            p = p / s

        # Attach children with priors and ensure child nodes exist in ttable.
        for mv, prior in zip(legal, p):
            board.push(mv)
            child_h = self._board_hash(board)
            board.pop()

            self._get_or_create_node(child_h, player_to_move=(not node.player_to_move))
            node.children[mv] = (child_h, float(prior))

    def _apply_dirichlet_to_root(self, root: Node):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        priors = np.array([root.children[m][1] for m in moves], dtype=np.float64)
        new_priors = (1.0 - DIRICHLET_EPS) * priors + DIRICHLET_EPS * noise
        for mv, p in zip(moves, new_priors):
            child_h, _ = root.children[mv]
            root.children[mv] = (child_h, float(p))

    def _backup(self, path: List[Node], value: float):
        """Backup alternating signs along the path (value is for leaf.player_to_move)."""
        v = float(value)
        for node in path:
            node.number_of_visits += 1
            node.value_sum += v
            node.mean_value = node.value_sum / float(node.number_of_visits)
            v = -v

    def _select_action_from_root(self, root: Node) -> chess.Move:
        # Pick the move with maximum visits.
        best_move = None
        best_visits = -1
        for mv, (child_h, _prior) in root.children.items():
            child = self.ttable[child_h]
            if child.number_of_visits > best_visits:
                best_visits = child.number_of_visits
                best_move = mv
        assert best_move is not None
        return best_move

    def run(
        self,
        root_state: chess.Board,
        history: Deque[np.ndarray],
        *,
        add_dirichlet_noise: bool,
        reuse_tree: bool,
        number_of_simulations: int = MCTS_SIMULATIONS,
    ) -> Tuple[chess.Move, Dict[chess.Move, float]]:
        """Run MCTS and return (best_move, policy_visits_dict)."""
        root_hash = self._board_hash(root_state)
        if (not reuse_tree) or (self._root_hash is None) or (self._root_hash != root_hash):
            self._root_hash = root_hash

        root_node = self._get_or_create_node(root_hash, player_to_move=root_state.turn)

        # Ensure root is expanded at least once.
        if not root_node.children:
            obs0 = board_to_obs(root_state, history)
            logits0, values0 = self._infer_many([obs0])
            self._expand_from_logits(root_node, root_state, logits0[0])
            self._backup([root_node], float(values0[0]))

        if add_dirichlet_noise:
            self._apply_dirichlet_to_root(root_node)

        pending = []

        for _ in range(int(number_of_simulations)):
            board = root_state.copy(stack=False)
            hist = deque(history, maxlen=LAST_POSITIONS)
            path: List[Node] = []

            node = root_node
            path.append(node)

            # Traverse until leaf or terminal
            while node.children:
                # If terminal, stop traversal early.
                tv = self._terminal_value(board, node.player_to_move)
                if tv is not None:
                    break

                # Select move by UCB
                best_score = -1e9
                best_move = None
                best_child_h = None
                best_prior = 0.0

                for mv, (child_h, prior) in node.children.items():
                    child = self.ttable[child_h]
                    score = self._ucb_score(node, child, prior)
                    if score > best_score:
                        best_score = score
                        best_move = mv
                        best_child_h = child_h
                        best_prior = prior

                assert best_move is not None and best_child_h is not None

                # Advance
                board.push(best_move)
                hist.append(board_to_planes(board))
                node = self.ttable[best_child_h]
                path.append(node)

            # Leaf handling
            tv = self._terminal_value(board, node.player_to_move)
            if tv is not None:
                self._backup(path, tv)
                continue

            # Need NN eval + expansion.
            pending.append((board, hist, node, path))
            if len(pending) >= self.inference_batch_size:
                self._flush_pending(pending)
                pending = []

        if pending:
            self._flush_pending(pending)

        # Build visit-count policy from root children
        policy: Dict[chess.Move, float] = {}
        for mv, (child_h, _prior) in root_node.children.items():
            policy[mv] = float(self.ttable[child_h].number_of_visits)

        move = self._select_action_from_root(root_node)
        return move, policy

    def _flush_pending(self, pending_items):
        obs_list = [board_to_obs(b, h) for (b, h, _node, _path) in pending_items]
        logits_b, values_b = self._infer_many(obs_list)

        for (item, logits, value) in zip(pending_items, logits_b, values_b):
            board, hist, node, path = item
            if not node.children:
                self._expand_from_logits(node, board, logits)
            self._backup(path, float(value))


# -----------------------------
# Self-play game generation
# -----------------------------
def self_play_game(
    model: AlphaZeroNet,
    infer_client: InferenceClient,
    worker_id: int,
    *,
    max_moves: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], chess.Outcome, int]:
    """Generate one self-play game.

    Returns:
        data: list[(obs, pi_vec, wdl)]
            wdl is from the perspective of the side-to-move at `obs`:
                0 = WIN, 1 = DRAW, 2 = LOSS
        outcome: chess.Outcome
        plies: number of half-moves played
    """
    _ = worker_id  # used only for logging on the worker side

    board = chess.Board()
    history: Deque[np.ndarray] = deque(maxlen=LAST_POSITIONS)
    history.append(board_to_planes(board))

    mcts = MCTS(model=None, inference_client=infer_client, device="cpu")  # weights live in central server

    trajectory: List[Tuple[np.ndarray, np.ndarray, bool]] = []
    moves_cnt = 0

    while not board.is_game_over(claim_draw=THREEFOLD) and moves_cnt < int(max_moves):
        obs = board_to_obs(board, history)

        move, policy = mcts.run(
            board,
            history,
            add_dirichlet_noise=True,
            reuse_tree=True,
            number_of_simulations=MCTS_SIMULATIONS,
        )

        # Normalized visit-count vector
        pi_vec = policy_to_pi_vector(board, policy)

        # For early plies: sample from policy with temperature
        if moves_cnt < TEMPERATURE_MOVES and policy:
            moves_list = list(policy.keys())
            counts = np.asarray([policy[m] for m in moves_list], dtype=np.float64)
            if counts.sum() <= 0:
                probs = np.ones_like(counts) / float(len(counts))
            else:
                probs = counts / counts.sum()

            tau = temperature_tau(moves_cnt)
            probs = apply_temperature(probs, tau=tau)
            move = np.random.choice(moves_list, p=probs)

        # Store player-to-move to assign WDL later.
        trajectory.append((obs, pi_vec, board.turn))

        board.push(move)
        history.append(board_to_planes(board))
        moves_cnt += 1

    outcome = board.outcome(claim_draw=THREEFOLD)
    if outcome is None:
        # forced stop -> treat as draw
        outcome = chess.Outcome(termination=chess.Termination.VARIANT_DRAW, winner=None)

    # Build (obs,pi,wdl) tuples WITHOUT any draw shaping: draws are honest draws.
    data: List[Tuple[np.ndarray, np.ndarray, int]] = []
    if outcome.winner is None:
        for obs, pi_vec, _player in trajectory:
            data.append((obs, pi_vec, 1))  # DRAW
    else:
        winner_is_white = (outcome.winner == chess.WHITE)
        for obs, pi_vec, player_is_white in trajectory:
            # perspective of side-to-move at this position
            if bool(player_is_white) == bool(winner_is_white):
                data.append((obs, pi_vec, 0))  # WIN
            else:
                data.append((obs, pi_vec, 2))  # LOSS

    return data, outcome, moves_cnt


# -----------------------------
# Training step
# -----------------------------
def train_one_step(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> float:
    """One SGD step on a batch sampled from replay.

    Value target is WDL (win/draw/loss) from the perspective of the side-to-move:
      0 = WIN, 1 = DRAW, 2 = LOSS
    """
    model.train()
    obs, pi, wdl = batch  # obs:(B,C,8,8), pi:(B,A), wdl:(B,)

    policy_logits, _value_scalar, wdl_logits = model(obs)

    logp = F.log_softmax(policy_logits, dim=1)
    loss_pi = -(pi * logp).sum(dim=1).mean()

    wdl_target = wdl.to(torch.long)
    loss_v = F.cross_entropy(wdl_logits, wdl_target)

    loss = loss_pi + loss_v

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(GRAD_CLIP_NORM))
    optimizer.step()

    return float(loss.detach().cpu().item())


# -----------------------------
# Evaluation / gating
# -----------------------------
def elo_diff_from_score(p: float) -> float:
    """Convert average score p in (0,1) to Elo difference (logistic model)."""
    p = float(min(max(p, 0.0), 1.0))
    p = min(max(p, ELO_FROM_SCORE_EPS), 1.0 - ELO_FROM_SCORE_EPS)
    return 400.0 * math.log10(p / (1.0 - p))


def score_lower_bound_from_counts(wins: int, draws: int, losses: int, *, z: float = 1.96) -> float:
    """Lower CI bound for score = (W + 0.5D)/N using normal approximation."""
    n = int(wins) + int(draws) + int(losses)
    if n <= 0:
        return 0.0

    sum_x = wins + 0.5 * draws
    sum_x2 = wins + 0.25 * draws  # 1^2 for wins, 0.5^2 for draws, 0 for losses

    mean = sum_x / n
    if n == 1:
        var = 0.0
    else:
        var = (sum_x2 - (sum_x * sum_x) / n) / (n - 1)
        var = max(var, 0.0)

    se = math.sqrt(var / n) if n > 0 else 0.0
    lb = mean - float(z) * se
    return float(min(max(lb, 0.0), 1.0))


def play_game_models(
    model_white: AlphaZeroNet,
    model_black: AlphaZeroNet,
    *,
    max_moves: int,
    mcts_sims: int = MCTS_SIMULATIONS,
) -> Tuple[chess.Outcome, int, str]:
    """Play one game model-vs-model (CPU-only, no dirichlet)."""
    board = chess.Board()
    history: Deque[np.ndarray] = deque(maxlen=LAST_POSITIONS)
    history.append(board_to_planes(board))

    mcts_w = MCTS(model=model_white, device="cpu", inference_batch_size=INFERENCE_BATCH_SIZE)
    mcts_b = MCTS(model=model_black, device="cpu", inference_batch_size=INFERENCE_BATCH_SIZE)

    plies = 0
    while not board.is_game_over(claim_draw=THREEFOLD) and plies < int(max_moves):
        if board.turn == chess.WHITE:
            move, _pol = mcts_w.run(
                board, history, add_dirichlet_noise=False, reuse_tree=True, number_of_simulations=mcts_sims
            )
        else:
            move, _pol = mcts_b.run(
                board, history, add_dirichlet_noise=False, reuse_tree=True, number_of_simulations=mcts_sims
            )

        board.push(move)
        history.append(board_to_planes(board))
        plies += 1

    outcome = board.outcome(claim_draw=THREEFOLD)
    if outcome is None:
        outcome = chess.Outcome(termination=chess.Termination.VARIANT_DRAW, winner=None)

    term = str(outcome.termination) if outcome.termination is not None else "UNKNOWN"
    return outcome, plies, term


def evaluate_vs_baseline(
    *,
    best_path: str,
    candidate_path: str,
    num_games: int,
    device: str = "cpu",
) -> dict:
    """Arena evaluation: candidate vs best.

    Metric returned for gating:
        score = (W + 0.5D) / N
        score_lb = lower CI bound of score
    """
    device_t = torch.device(device)

    best = AlphaZeroNet().to(device_t)
    cand = AlphaZeroNet().to(device_t)

    best.load_state_dict(torch.load(best_path, map_location=device_t))
    cand.load_state_dict(torch.load(candidate_path, map_location=device_t))

    best.eval()
    cand.eval()

    wins = draws = losses = 0
    plies_sum = 0
    term_counts: Dict[str, int] = collections.Counter()

    for g in range(int(num_games)):
        # Alternate colors to reduce first-move bias.
        cand_is_white = (g % 2 == 0)
        if cand_is_white:
            outcome, plies, term = play_game_models(cand, best, max_moves=EVAL_MAX_GAME_LENGTH)
        else:
            outcome, plies, term = play_game_models(best, cand, max_moves=EVAL_MAX_GAME_LENGTH)

        plies_sum += int(plies)
        term_counts[term] += 1

        if outcome.winner is None:
            draws += 1
        else:
            cand_won = (outcome.winner == (chess.WHITE if cand_is_white else chess.BLACK))
            if cand_won:
                wins += 1
            else:
                losses += 1

        if DEBUG_PRINT_EVAL_TERMINATIONS and (g == 0 or (g + 1) == num_games):
            # log first and last termination only (keeps noise low)
            pass

    games = wins + draws + losses
    score_sum = wins + 0.5 * draws
    p = (score_sum / games) if games > 0 else 0.0
    win_rate = (wins / games) if games > 0 else 0.0
    draw_rate = (draws / games) if games > 0 else 0.0
    score_lb = score_lower_bound_from_counts(wins, draws, losses, z=EVAL_SCORE_Z) if games > 0 else 0.0
    elo = elo_diff_from_score(p) if games > 0 else 0.0
    avg_plies = (plies_sum / games) if games > 0 else 0.0

    return {
        "games": int(games),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "draw_rate": float(draw_rate),
        "score": float(p),
        "score_lb": float(score_lb),
        "elo_diff": float(elo),
        "avg_plies": float(avg_plies),
        "terminations": dict(term_counts),
    }
