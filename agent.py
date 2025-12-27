import copy
import math
import numpy as np
from collections import deque
import queue as _py_queue

import chess
import chess.polyglot
import torch
import torch.nn.functional as F

from env import board_to_planes, board_to_obs, move_to_id
from nw import CNNActorCritic
from constants import *

position_deque = deque(maxlen=LAST_POSITIONS)


class InferenceClient:
    """
    Синхронный клиент для центрального GPU-инференса.

    Worker отправляет на request_q: (worker_id, req_id, obs_batch)
    Получает с response_q: (req_id, probs, values)

    probs: [B, TOTAL_MOVES] float32
    values: [B] float32
    """

    def __init__(self, worker_id: int, request_q, response_q, stop_event=None, log_q=None):
        self.worker_id = int(worker_id)
        self.request_q = request_q
        self.response_q = response_q
        self.stop_event = stop_event
        self.log_q = log_q
        self._next_req_id = 0

    def _log(self, msg: str):
        if self.log_q is not None:
            try:
                self.log_q.put(msg)
            except Exception:
                pass

    def heartbeat(self, msg: str):
        self._log(msg)

    def predict(self, obs_batch: np.ndarray):
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        req_id = self._next_req_id
        self._next_req_id += 1

        if req_id == 0:
            self._log(f"[worker {self.worker_id}] first inference request: batch={obs_batch.shape[0]}")

        self.request_q.put((self.worker_id, req_id, obs_batch))

        while True:
            try:
                rid, probs, values = self.response_q.get(timeout=0.2)
            except _py_queue.Empty:
                if self.stop_event is not None and self.stop_event.is_set():
                    raise RuntimeError("InferenceClient: stop_event set")
                continue

            if rid == req_id:
                return probs, values

            # крайне маловероятно, но если ответы перепутались — вернём назад
            self.response_q.put((rid, probs, values))


def temperature_tau(ply: int) -> float:
    t = min(max(ply, 0), TEMPERATURE_DECAY_PLY)
    return float(TEMPERATURE_TAU_START + (TEMPERATURE_TAU_END - TEMPERATURE_TAU_START) * (t / TEMPERATURE_DECAY_PLY))


def apply_temperature(probs, tau: float):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 1e-12)
    if tau <= 1e-6:
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out
    out = probs ** (1.0 / tau)
    out /= out.sum()
    return out


def policy_to_pi_vector(board: chess.Board, policy: dict) -> np.ndarray:
    pi = np.zeros(TOTAL_MOVES, dtype=np.float32)
    for move, prob in policy.items():
        action_id = move_to_id(board, move)
        assert 0 <= action_id < TOTAL_MOVES, "move_to_id вернул некорректный id"
        pi[action_id] = prob
    return pi


def self_play_game(
    *,
    inference_client: InferenceClient | None = None,
    model: CNNActorCritic | None = None,
    max_moves=200,
    device="cpu",
):
    assert (model is not None) or (inference_client is not None)

    mcts = MCTS(
        model=model,
        inference_client=inference_client,
        device=device,
        inference_batch_size=INFERENCE_BATCH_SIZE,
    )

    board = chess.Board()
    position_deque.clear()
    position_deque.append(board_to_planes(board))

    trajectory = []
    moves_cnt = 0

    while moves_cnt < max_moves:
        player = board.turn
        obs = board_to_obs(board, position_deque)

        nos = TRAINING_MCTS_SIMULATIONS
        if THINK_LONGER_AS_GAME_GOES:
            nos = max(nos, int(moves_cnt * THINK_COEFF))

        _best_action, policy0 = mcts.run(
            board,
            add_dirichlet_noise=True,
            reuse_tree=True,
            number_of_simulations=nos,
        )

        if not policy0:
            break

        root_hash = int(chess.polyglot.zobrist_hash(board))
        root_node = mcts.ttable.get(root_hash)

        policy = dict(policy0)

        def _renorm_inplace(p: dict[chess.Move, float]) -> None:
            if not p:
                return
            if len(p) == 1:
                mv = next(iter(p))
                p.clear()
                p[mv] = 1.0
                return
            s = float(sum(p.values()))
            if s <= 1e-12:
                n = len(p)
                w = 1.0 / n
                for mv in list(p.keys()):
                    p[mv] = w
                return
            inv = 1.0 / s
            for mv in list(p.keys()):
                p[mv] = float(p[mv]) * inv

        while True:
            if len(policy) == 1:
                move = next(iter(policy))
            else:
                if moves_cnt < TEMPERATURE_MOVES:
                    moves_list = list(policy.keys())
                    probs = np.array([policy[m] for m in moves_list], dtype=np.float64)
                    tau = temperature_tau(moves_cnt)
                    probs = apply_temperature(probs, tau=tau)
                    move = moves_list[np.random.choice(len(moves_list), p=probs)]
                else:
                    move = max(policy, key=policy.get)

            board.push(move)

            if len(policy) > 1 and board.can_claim_threefold_repetition():
                board.pop()
                policy.pop(move, None)
                _renorm_inplace(policy)
                continue

            board.pop()
            _renorm_inplace(policy)
            pi_vec = policy_to_pi_vector(board, policy)

            board.push(move)

            if root_node is not None and move in root_node.children:
                child_h, _prior = root_node.children[move]
                mcts._root_hash = child_h
            else:
                mcts._root_hash = int(chess.polyglot.zobrist_hash(board))

            break

        trajectory.append((obs, pi_vec, player))
        position_deque.append(board_to_planes(board))
        moves_cnt += 1

        if board.is_game_over(claim_draw=THREEFOLD):
            break

    outcome = board.outcome(claim_draw=THREEFOLD)

    data = []
    if outcome is None:
        for obs, pi_vec, _player in trajectory:
            data.append((obs, pi_vec, CONTEMPT_AGAINST_DRAW))
    elif outcome.winner is None:
        is_repetition = outcome.termination in (
            chess.Termination.THREEFOLD_REPETITION,
            chess.Termination.FIVEFOLD_REPETITION,
        )
        z = REPETITION_PENALTY if is_repetition else CONTEMPT_AGAINST_DRAW
        for obs, pi_vec, _player in trajectory:
            data.append((obs, pi_vec, z))
    else:
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
        for obs, pi_vec, player in trajectory:
            z = z_white if player == chess.WHITE else -z_white
            data.append((obs, pi_vec, z))

    return data, outcome


class Node:
    def __init__(self, player_to_move: bool):
        self.player_to_move = player_to_move
        self.number_of_visits = 0
        self.value_sum = 0.0
        self.mean_value = 0.0
        self.children: dict[chess.Move, tuple[int, float]] = {}


class MCTS:
    def __init__(
        self,
        c_puct: float = 1.5,
        model: CNNActorCritic | None = None,
        inference_client: InferenceClient | None = None,
        device: str = "cpu",
        inference_batch_size: int = 8,
    ):
        self.c_puct = float(c_puct)
        self.model = model
        self.inference_client = inference_client
        self.device = device
        self.inference_batch_size = int(inference_batch_size)

        # чтобы не падать на первом run()
        self._root_hash: int | None = None

        self.ttable: dict[int, Node] = {}

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    def _board_hash(self, board: chess.Board) -> int:
        return int(chess.polyglot.zobrist_hash(board))

    def _get_or_create_node(self, state_hash: int, player_to_move: bool) -> Node:
        node = self.ttable.get(state_hash)
        if node is None:
            node = Node(player_to_move=player_to_move)
            self.ttable[state_hash] = node
        return node

    def _ucb_score(self, parent: Node, child: Node, prior: float) -> float:
        q = 0.0 if child.number_of_visits == 0 else child.mean_value
        u = self.c_puct * prior * math.sqrt(parent.number_of_visits + 1e-8) / (1.0 + child.number_of_visits)
        return q + u

    # =======================
    # ✅ FIXED batched MCTS.run
    # =======================
    def run(
        self,
        root_state: chess.Board,
        position_history=None,
        add_dirichlet_noise: bool = False,
        reuse_tree: bool = False,
        number_of_simulations=TRAINING_MCTS_SIMULATIONS,
    ):
        assert (self.model is not None) or (self.inference_client is not None)

        root_hash = self._board_hash(root_state)
        if (not reuse_tree) or (self._root_hash is None) or (self._root_hash != root_hash):
            self._root_hash = root_hash

        root_node = self._get_or_create_node(root_hash, player_to_move=root_state.turn)

        # гарантируем, что у root есть дети (иначе policy пустой)
        if not root_node.children:
            leaf_info = self._traverse_to_leaf(root_state, position_history)
            if leaf_info.get("terminal", False):
                self._backup(leaf_info["path"], leaf_info["value"])
            else:
                self._eval_expand_backup_batch([leaf_info])

        if add_dirichlet_noise:
            self._add_dirichlet_noise_to_root(root_state, root_node)

        # batched симуляции: считаем симуляции по размеру batch
        sims_done = 0
        target = int(number_of_simulations)

        while sims_done < target:
            batch = []

            # набираем batch не больше inference_batch_size и не больше оставшихся симуляций
            while len(batch) < self.inference_batch_size and (sims_done + len(batch)) < target:
                leaf_info = self._traverse_to_leaf(root_state, position_history)

                if leaf_info.get("terminal", False):
                    # terminal симуляция считается как 1
                    self._backup(leaf_info["path"], leaf_info["value"])
                    sims_done += 1
                    continue

                batch.append(leaf_info)

            if batch:
                self._eval_expand_backup_batch(batch)
                sims_done += len(batch)

        policy = self._policy_from_root(root_node)
        best_action = self._select_action_from_root(root_node)
        return best_action, policy

    def _add_dirichlet_noise_to_root(self, board: chess.Board, root: Node):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        priors = np.array([root.children[m][1] for m in moves], dtype=np.float64)
        new_priors = (1 - DIRICHLET_EPSILON) * priors + DIRICHLET_EPSILON * noise
        for m, p in zip(moves, new_priors):
            child_h, _old_p = root.children[m]
            root.children[m] = (child_h, float(p))

    def _traverse_to_leaf(self, root_state: chess.Board, position_history=None):
        board = root_state.copy()

        base_hist = position_history if position_history is not None else position_deque
        hist = deque(base_hist, maxlen=LAST_POSITIONS)

        root_hash = self._board_hash(board)
        root_node = self._get_or_create_node(root_hash, player_to_move=board.turn)

        path = [root_node]
        node = root_node

        while True:
            if board.is_game_over(claim_draw=THREEFOLD):
                outcome = board.outcome(claim_draw=THREEFOLD)
                if outcome is None or outcome.winner is None:
                    value = CONTEMPT_AGAINST_DRAW
                else:
                    value = 1.0 if outcome.winner == root_node.player_to_move else -1.0
                return {"terminal": True, "value": float(value), "path": path}

            if not node.children:
                break

            best_score = float("-inf")
            best_move = None
            best_child = None

            for move, (child_hash, prior) in node.children.items():
                child = self.ttable[child_hash]
                score = self._ucb_score(node, child, prior)
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_child = child

            if best_move is None:
                break

            board.push(best_move)
            hist.append(board_to_planes(board))

            node = best_child
            path.append(node)

        obs = board_to_obs(board, hist)
        return {"obs": obs, "board": board, "leaf_node": node, "path": path}

    def _eval_expand_backup_batch(self, batch):
        obs_batch = np.stack([x["obs"] for x in batch], axis=0).astype(np.float32, copy=False)

        if self.model is not None:
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits, v_pred = self.model(obs_t)
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                values = v_pred.detach().cpu().numpy().reshape(-1)
        else:
            probs, values = self.inference_client.predict(obs_batch)

        for i, item in enumerate(batch):
            leaf = item["leaf_node"]
            board = item["board"]

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                self._backup(item["path"], float(values[i]))
                continue

            priors = {}
            total_p = 0.0
            for mv in legal_moves:
                a = move_to_id(board, mv)
                p = float(probs[i][a])
                priors[mv] = p
                total_p += p

            if total_p <= 1e-12:
                p0 = 1.0 / len(legal_moves)
                for mv in legal_moves:
                    priors[mv] = p0
            else:
                inv = 1.0 / total_p
                for mv in legal_moves:
                    priors[mv] *= inv

            for mv in legal_moves:
                board.push(mv)
                child_h = self._board_hash(board)
                board.pop()

                self._get_or_create_node(child_h, player_to_move=(not leaf.player_to_move))
                leaf.children[mv] = (child_h, float(priors[mv]))

            self._backup(item["path"], float(values[i]))

    def _backup(self, path, value: float):
        v = float(value)
        for node in path:
            node.number_of_visits += 1
            node.value_sum += v
            node.mean_value = node.value_sum / node.number_of_visits
            v = -v

    def _select_action_from_root(self, root: Node) -> chess.Move:
        best_visits = -1
        best_move = None
        for mv, (child_h, _p) in root.children.items():
            child = self.ttable[child_h]
            if child.number_of_visits > best_visits:
                best_visits = child.number_of_visits
                best_move = mv
        return best_move

    def _policy_from_root(self, root: Node):
        visits = {}
        total = 0
        for mv, (child_h, _p) in root.children.items():
            c = self.ttable[child_h].number_of_visits
            visits[mv] = c
            total += c
        if total == 0:
            n = len(visits) if visits else 1
            return {mv: 1.0 / n for mv in visits}
        return {mv: c / total for mv, c in visits.items()}


def train_one_step(model: CNNActorCritic, optimizer: torch.optim.Optimizer, *, device: str = "cpu"):
    from replay_buffer import get_replay_buffer
    rb = get_replay_buffer()

    if len(rb) < max(1, min(MIN_REPLAY_SIZE, BATCH_SIZE)):
        return {"did_step": False, "replay_sizes": rb.sizes(), "loss": None}

    obs_batch, pi_batch, z_batch = rb.sample(BATCH_SIZE, p_mate=P_MATE_IN_BATCH)

    model.train()
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    pi_t = torch.as_tensor(pi_batch, dtype=torch.float32, device=device)
    z_t = torch.as_tensor(z_batch, dtype=torch.float32, device=device)

    logits, v_pred = model(obs_t)
    value_loss = F.mse_loss(v_pred, z_t)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
    loss = value_loss + policy_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
    optimizer.step()

    return {
        "did_step": True,
        "replay_sizes": rb.sizes(),
        "loss": float(loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
    }
