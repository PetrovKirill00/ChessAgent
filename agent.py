# agent.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import random
import collections
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
    """
    Worker-side sync client.

    Request schema (to shared request_q):
        (worker_id: int, rid: int, obs_u8: np.ndarray)                      # legacy
        (worker_id: int, rid: int, obs_u8: np.ndarray, idxs_u16: np.ndarray) # sparse logits for legal moves

    Response schema (to per-worker response_q):
        (rid: int, logits: np.ndarray, value: float)
        - logits is either full-length (TOTAL_MOVES,) or sparse (len(idxs_u16),) depending on request.
    """
    def __init__(self, request_q, response_q, worker_id: int, stop_event=None, **_ignored):
        self.request_q = request_q
        self.response_q = response_q
        self.worker_id = int(worker_id)
        self.stop_event = stop_event
        self._rid = 0
        self._printed_first = False

    def infer(self, obs_u8: np.ndarray, idxs_u16: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
        import queue as _queue
        import time as _time

        self._rid += 1
        rid = self._rid

        if idxs_u16 is None:
            self.request_q.put((self.worker_id, rid, obs_u8), block=True)
        else:
            # ensure compact dtypes for IPC
            if obs_u8.dtype != np.uint8:
                obs_u8 = obs_u8.astype(np.uint8, copy=False)
            if idxs_u16.dtype != np.uint16:
                idxs_u16 = idxs_u16.astype(np.uint16, copy=False)
            self.request_q.put((self.worker_id, rid, obs_u8, idxs_u16), block=True)

        if DEBUG_PRINT_WORKER_FIRST_INFER and (not self._printed_first):
            self._printed_first = True
            print(f"[worker {self.worker_id}] first inference request: batch=1", flush=True)

        # Wait response with timeout so we can exit cleanly when stop_event is set.
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise KeyboardInterrupt()

            try:
                rrid, logits, value = self.response_q.get(timeout=0.25)
            except _queue.Empty:
                continue

            if rrid == rid:
                return logits, float(value)

            # Out-of-order shouldn't happen; drop anything else.


# -----------------------------
# MCTS
# -----------------------------
@dataclass
class Node:
    prior: float
    value_sum: float = 0.0
    visit_count: int = 0
    children: Optional[Dict[chess.Move, "Node"]] = None

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    # logits may arrive as float16 from the GPU server; do softmax in float32 for stability and speed
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


class MCTS:
    """AlphaZero-style MCTS with correct expansion and single NN call per leaf."""

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
        """Enable tree reuse across real moves."""
        if self._root is None or self._root_hash != board_before.fen():
            self.reset()
            return
        if self._root.children is None:
            self.reset()
            return
        child = self._root.children.get(move_played)
        if child is None:
            self.reset()
            return
        b = board_before.copy(stack=False)
        b.push(move_played)
        self._root = child
        self._root_hash = b.fen()

    def run(
        self,
        board: chess.Board,
        position_deque: Deque[chess.Board],
        *,
        add_dirichlet_noise: bool,
        reuse_tree: bool,
    ) -> Tuple[Dict[chess.Move, float], Dict[chess.Move, int]]:
        """Return (policy_probs, visit_counts) at root.
        NOTE: we do NOT pick the final move here, because self-play may sample with temperature.
        """
        h = board.fen()
        if not (reuse_tree and self._root_hash == h and self._root is not None):
            self._root = Node(prior=1.0, children=None)
            self._root_hash = h

        root = self._root

        # Expand root once (attach to the real tree).
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

        # Simulations
        for _ in range(self.simulations):
            b = board.copy(stack=False)
            hist: Deque[chess.Board] = collections.deque(position_deque, maxlen=LAST_POSITIONS)

            node = root
            path: List[Node] = [node]

            # Selection
            while node.children is not None and len(node.children) > 0:
                mv, node = self._select_child(node)
                b.push(mv)
                hist.append(b.copy(stack=False))
                path.append(node)

            # Evaluate leaf
            outcome = b.outcome(claim_draw=False)
            if outcome is not None:
                if outcome.winner is None:
                    # Draw: use 0 at terminal (draw bias is handled via CONTEMPT in NN mapping)
                    value = 0.0
                else:
                    value = 1.0 if outcome.winner == b.turn else -1.0
            else:
                value = self._expand(node, b, hist)

            # Backprop (flip perspective each ply)
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += float(value)
                value = -value

        # Build policy from visit counts
        if root.children is None or len(root.children) == 0:
            # no legal moves (should only happen in terminal)
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

        sqrt_visits = math.sqrt(node.visit_count + 1.0)
        for mv, child in node.children.items():
            q_parent = -child.value()
            u = MCTS_C_PUCT * child.prior * (sqrt_visits / (1.0 + child.visit_count))
            score = q_parent + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child

        return best_move, best_child

    def _expand(self, node: Node, board: chess.Board, position_deque: Deque[chess.Board]) -> float:
        """Attach children to *this* node and return NN value (side-to-move)."""
        obs = board_to_obs(board, position_deque)

        legal = list(board.legal_moves)
        if not legal:
            node.children = {}
            _logits, value = self._infer(obs, None)
            return float(value)

        idxs = np.array([move_to_index(mv) for mv in legal], dtype=np.uint16)
        logits_legal, value = self._infer(obs, idxs)
        probs = _softmax(logits_legal)

        node.children = {}
        for mv, p in zip(legal, probs):
            node.children[mv] = Node(prior=float(p), children=None)

        return float(value)
    def _infer(self, obs: np.ndarray, idxs: np.ndarray | None = None) -> Tuple[np.ndarray, float]:
        """Return (policy_logits, value_scalar) for side-to-move."""
        if self.infer_client is not None:
            # Compact obs for IPC
            if obs.dtype != np.uint8:
                obs_u8 = (obs > 0.5).astype(np.uint8)
            else:
                obs_u8 = obs
            idxs_u16 = None if idxs is None else idxs.astype(np.uint16, copy=False)
            logits, value = self.infer_client.infer(obs_u8, idxs_u16)
            return logits, float(value)

        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0)
            out = self.model(x)
            pol_logits = out[0]
            wdl_logits = out[2]
            pol_np = pol_logits.squeeze(0).cpu().numpy().astype(np.float32)

            # Contempt-aware scalar value mapping
            probs = torch.softmax(wdl_logits, dim=1)  # (1,3)
            p_win = float(probs[0, 0].cpu().item())
            p_draw = float(probs[0, 1].cpu().item())
            p_loss = float(probs[0, 2].cpu().item())
            value = (p_win - p_loss) + CONTEMPT_DRAW * p_draw

        return pol_np, float(value)


# -----------------------------
# Temperature schedule
# -----------------------------
def temperature_tau(move_idx: int) -> float:
    if move_idx >= TEMPERATURE_MOVES:
        return 0.0
    # linear decay
    if TEMPERATURE_MOVES <= 1:
        return TEMPERATURE_TAU_END
    t = move_idx / max(1, (TEMPERATURE_MOVES - 1))
    return float(TEMPERATURE_TAU_START * (1.0 - t) + TEMPERATURE_TAU_END * t)


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
    while not board.is_game_over(claim_draw=True) and moves_cnt < max_moves:
        position_deque.append(board.copy(stack=False))

        policy, _visit_counts = mcts.run(
            board,
            position_deque,
            add_dirichlet_noise=True,
            reuse_tree=True,
        )

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

    return samples, outcome, moves_cnt, draw_reason


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