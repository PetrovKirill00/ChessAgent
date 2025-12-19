import copy
import math
import random
import numpy as np
from env import board_to_planes, board_to_obs, action_id_to_move, move_to_id
import chess, chess.polyglot
import torch
import torch.nn.functional as F
from nw import CNNActorCritic
from collections import deque
from constants import (
    LAST_POSITIONS,
    TOTAL_MOVES,
    PIECE_VALUES,
    BATCH_SIZE,
    MIN_REPLAY_SIZE,
    TRAIN_STEPS_PER_ITER,
    INFERENCE_BATCH_SIZE,
    TEMPERATURE_MOVES,
    TRAINING_MCTS_SIMULATIONS,
    DIRICHLET_ALPHA,
    DIRICHLET_EPSILON,
    CONTEMPT_AGAINST_DRAW,
    THREEFOLD,
    TEMPERATURE_TAU_START,
    TEMPERATURE_TAU_END,
    TEMPERATURE_DECAY_PLY,
    REPETITION_PENALTY,
    GRADIENT_CLIP_NORM,
)
from replay_buffer import replay_buffer

position_deque = deque(maxlen=LAST_POSITIONS)

def temperature_tau(ply: int) -> float:
    # линейный спад: START -> END за DECAY_PLY полуходов
    t = min(max(ply, 0), TEMPERATURE_DECAY_PLY)
    tau = TEMPERATURE_TAU_START + (TEMPERATURE_TAU_END - TEMPERATURE_TAU_START) * (t / TEMPERATURE_DECAY_PLY)
    return float(tau)

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


def policy_to_pi_vector(board: chess.Board, policy:dict) -> np.ndarray:
    """
    Преобразуем policy в виде {chess.Move: prob} в вектор длины TOTAL_MOVES.
    """

    pi = np.zeros(TOTAL_MOVES, dtype=np.float32)
    for move, prob in policy.items():
        action_id = move_to_id(board, move)
        assert 0 <= action_id < TOTAL_MOVES, "encode_action вернул некорректный id"
        pi[action_id] = prob
    return pi

def _material_diff(board: chess.Board) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    diff = 0
    for piece in board.piece_map().values():
        v = values[piece.piece_type]
        diff += v if piece.color == chess.WHITE else -v
    return diff

def self_play_game(model: CNNActorCritic,
                   num_simulations=64,
                   max_moves=200,
                   device="cpu"):
    """
    Играем партию MCTS vs MCTS.
    Возвращаем список (obs, pi_vec, z) для каждого хода.
      obs   : np.ndarray [TOTAL_LAYERS, 8, 8]
      pi_vec: np.ndarray [TOTAL_MOVES]
      z     : скаляр в {-1, 0, +1} с точки зрения игрока, который делал ход.
    """
    mcts = MCTS(number_of_simulations=num_simulations,
                model=model,
                device=device,
                inference_batch_size=INFERENCE_BATCH_SIZE)
    board = chess.Board()
    position_deque.clear()
    position_deque.append(board_to_planes(board))

    trajectory = []
    moves_cnt = 0

    forbidden = set()

    while moves_cnt < max_moves:
        player = board.turn
        obs = board_to_obs(board, position_deque)

        move0, policy0 = mcts.run(board, add_dirichlet_noise=True, reuse_tree=True)

        # 1) фильтруем forbidden
        policy = {mv: w for mv, w in policy0.items() if mv not in forbidden}

        # 2) если всё запретили — сбрасываем запреты (иначе бесконечный pop/continue)
        cleared = False
        if not policy:
            cleared = True
            forbidden.clear()
            policy = dict(policy0)

        # π логируем по текущей policy (после фильтра)
        pi_vec = policy_to_pi_vector(board, policy)

        # 3) выбираем move ТОЛЬКО из текущей policy
        if moves_cnt < TEMPERATURE_MOVES:
            moves_list = list(policy.keys())
            probs = np.array([policy[m] for m in moves_list], dtype=np.float64)
            tau = temperature_tau(moves_cnt)
            probs = apply_temperature(probs, tau=tau)
            move = moves_list[np.random.choice(len(moves_list), p=probs)]
        else:
            move = max(policy, key=policy.get)

        board.push(move)

        # 4) обрубаем ходы, которые делают repetition claimable
        if not cleared and board.can_claim_threefold_repetition():
            board.pop()
            forbidden.add(move)
            continue

        forbidden.clear()

        trajectory.append((obs, pi_vec, player))
        position_deque.append(board_to_planes(board))
        moves_cnt += 1

        # если хочешь — можно оставить этот break как “нормальный” конец партии
        if board.is_game_over(claim_draw=THREEFOLD):
            break

    outcome = board.outcome(claim_draw=THREEFOLD)
    data = []

    if outcome is None:
        # считаем это "обычной" ничьёй
        z = CONTEMPT_AGAINST_DRAW
        for obs, pi_vec, player in trajectory:
            data.append((obs, pi_vec, z))
        return data, outcome
    elif outcome.winner is None:
        # НИЧЬЯ: делим на повторение и остальные причины
        is_repetition = outcome.termination in (
            chess.Termination.THREEFOLD_REPETITION,
            chess.Termination.FIVEFOLD_REPETITION,
        )

        z = REPETITION_PENALTY if is_repetition else CONTEMPT_AGAINST_DRAW

        # тут ты можешь решать: хочешь ли одинаковый z для обоих,
        # или антисимметрию как раньше. Сейчас — одинаковый для обоих:
        for obs, pi_vec, player in trajectory:
            data.append((obs, pi_vec, z))
    else:
        # ПОБЕДА/ПОРАЖЕНИЕ: оставляем старую логику
        z_white = 1.0 if outcome.winner == chess.WHITE else -1.0
        for obs, pi_vec, player in trajectory:
            z = z_white if player == chess.WHITE else -z_white
            data.append((obs, pi_vec, z))

    return data, outcome


def train_one_iteration(model: CNNActorCritic,
                        optimizer: torch.optim.Optimizer,
                        num_simulations=TRAINING_MCTS_SIMULATIONS,
                        max_moves=200,
                        device="cpu"):
    """
    Одна итерация обучения:
      1) генерируем одну self-play партию (self_play_game),
      2) добавляем все позиции в replay buffer,
      3) делаем несколько SGD- шагов по случайным минибатчам из буфера.

    Возвращает словарь со статистикой.
    """

    # 1. self-play: одна партия
    data, outcome = self_play_game(model=model,
                                   num_simulations=num_simulations,
                                   max_moves=max_moves,
                                   device=device)
    if data is None or len(data) == 0:
        return None

    # 2. добавляем позиции партии в буфер
    replay_buffer.add_many(data)
    buffer_size = len(replay_buffer)

    # Если данных совсем мало — просто копим буфер, почти не обучаясь
    if buffer_size < BATCH_SIZE:
        return {
            "loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "num_positions": len(data),   # сколько позиций добавили
            "buffer_size": buffer_size,
            "train_steps": 0,
            "positions_used_for_training": 0,
            "outcome": outcome
        }

    # Сколько шагов SGD делаем на этой итерации
    if buffer_size < MIN_REPLAY_SIZE:
        # тёплый старт: буфер ещё небольшой → один шаг
        train_steps = 1
    else:
        train_steps = TRAIN_STEPS_PER_ITER

    last_loss = 0.0
    last_value_loss = 0.0
    last_policy_loss = 0.0
    total_positions_used = 0

    model.train()

    for _ in range(train_steps):
        obs_batch, pi_batch, z_batch = replay_buffer.sample(BATCH_SIZE)
        total_positions_used += len(z_batch)

        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        pi_t = torch.as_tensor(pi_batch, dtype=torch.float32, device=device)
        z_t = torch.as_tensor(z_batch, dtype=torch.float32, device=device)

        logits, v_pred = model(obs_t)

        value_loss = F.mse_loss(v_pred, z_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        policy_loss = -(pi_t * log_probs).sum(dim=-1).mean()
        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        last_loss = float(loss.detach().item())
        last_value_loss = float(value_loss.detach().item())
        last_policy_loss = float(policy_loss.detach().item())

    return {
        "loss": last_loss,
        "value_loss": last_value_loss,
        "policy_loss": last_policy_loss,
        "num_positions": len(data),                 # добавлено в буфер
        "buffer_size": buffer_size,                 # размер буфера после добавления
        "train_steps": train_steps,                 # сколько SGD-шагов
        "positions_used_for_training": total_positions_used,
        "outcome": outcome,
    }

class Node:
    def __init__(self, player_to_move: bool):
        self.player_to_move = player_to_move
        self.number_of_visits = 0
        self.value_sum = 0.0
        self.mean_value = 0.0
        # children: move -> (child_hash, prior_on_edge)
        self.children: dict[chess.Move, tuple[int, float]] = {}


class MCTS:
    def __init__(
        self,
        number_of_simulations: int = 100,
        c_puct: float = 1.5,
        model: CNNActorCritic | None = None,
        device: str = "cpu",
        inference_batch_size: int = 64,   # <<< главное: размер батча для NN
    ):
        self.number_of_simulations = int(number_of_simulations)
        self.c_puct = float(c_puct)
        self.model = model
        self.device = device
        self.inference_batch_size = int(inference_batch_size)

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        self.ttable: dict[int, Node] = {}

    def reset_tree(self):
        self.ttable.clear()
        self._root_hash = None

    def reroot(self, board: chess.Board):
        """Сделать текущую позицию новым root, НЕ очищая таблицу."""
        h = self._board_hash(board)
        self._root_hash = h
        self._get_or_create_node(h, player_to_move=board.turn)

    def _board_hash(self, board: chess.Board) -> int:
        return int(chess.polyglot.zobrist_hash(board))

    def _get_or_create_node(self, state_hash: int, player_to_move: bool) -> Node:
        node = self.ttable.get(state_hash)
        if node is None:
            node = Node(player_to_move=player_to_move)
            self.ttable[state_hash] = node
        return node

    def _ucb_score(self, parent: Node, child: Node, prior_edge: float) -> float:
        if child.number_of_visits == 0:
            q = 0.0
        else:
            q = -child.mean_value

        eps = 1e-8
        u = (
            self.c_puct
            * float(prior_edge)
            * math.sqrt(parent.number_of_visits + eps)
            / (1.0 + child.number_of_visits)
        )
        return q + u

    def _make_root_prior_override(self, root: Node, alpha: float = DIRICHLET_ALPHA, epsilon: float = DIRICHLET_EPSILON):
        # children: move -> (child_hash, prior)
        moves = list(root.children.keys())
        if not moves:
            return None

        priors = np.array([root.children[mv][1] for mv in moves], dtype=np.float64)
        priors_sum = priors.sum()
        if priors_sum > 0:
            priors /= priors_sum
        else:
            priors[:] = 1.0 / len(moves)

        noise = np.random.dirichlet([alpha] * len(moves))
        mixed = (1.0 - epsilon) * priors + epsilon * noise

        return {mv: float(p) for mv, p in zip(moves, mixed)}

    def run(self, root_state: chess.Board,
            position_history=None,
            add_dirichlet_noise: bool = False,
            reuse_tree: bool = False):
        assert self.model is not None, "MCTS: model is None"

        root_hash = self._board_hash(root_state)

        if (not reuse_tree) or (not self.ttable) or (getattr(self, "_root_hash", None) != root_hash):
            # либо не переиспользуем, либо дерево пустое, либо другая позиция → стартуем заново
            self.ttable.clear()
            self._root_hash = root_hash

        root = self._get_or_create_node(root_hash, player_to_move=root_state.turn)
        root_prior_override = None
        if add_dirichlet_noise and root.children:
            root_prior_override = self._make_root_prior_override(root)

        # ---- батчим только evaluation/expansion ----
        pending = []  # список задач на оценку нейросетью

        for _ in range(self.number_of_simulations):
            res = self._traverse_to_leaf(root_state, root, position_history=position_history, root_prior_override=root_prior_override)

            if res["terminal"]:
                # терминал: можем сразу backprop
                value = res["value"]
                self._backup(res["path"], value)
                continue

            # нетерминальный лист -> откладываем в батч
            pending.append(res)

            # если батч набрался — делаем inference пачкой
            if len(pending) >= self.inference_batch_size:
                self._eval_expand_backup_batch(pending)
                # если root был листом и только что расширился, то override создастся сейчас
                if add_dirichlet_noise and root_prior_override is None and root.children:
                    root_prior_override = self._make_root_prior_override(root)
                pending.clear()

        # добиваем хвост
        if pending:
            self._eval_expand_backup_batch(pending)
            # если root был листом и только что расширился, то override создастся сейчас
            if add_dirichlet_noise and root_prior_override is None and root.children:
                root_prior_override = self._make_root_prior_override(root)

            pending.clear()

        best_action = self._select_action_from_root(root)
        policy = self._policy_from_root(root)

        if reuse_tree and best_action is not None and best_action in root.children:
            child_h, _p = root.children[best_action]
            self._root_hash = child_h  # новый root — выбранный ребёнок

        return best_action, policy

    def _traverse_to_leaf(self, state: chess.Board, root_node: Node, position_history=None, root_prior_override=None):
        board = state.copy()

        base_hist = position_history if position_history is not None else position_deque
        hist = deque(base_hist, maxlen=LAST_POSITIONS)

        path: list[Node] = [root_node]
        node = root_node

        while True:
            if board.is_game_over(claim_draw=THREEFOLD):
                outcome = board.outcome(claim_draw=THREEFOLD)
                if outcome is None or outcome.winner is None:
                    value = 0.0
                else:
                    value = 1.0 if outcome.winner == root_node.player_to_move else -1.0
                return {"terminal": True, "value": float(value), "path": path}

            if not node.children:
                # нашли лист (не расширен)
                break

            best_score = float("-inf")
            best_move = None
            best_child = None

            for move, (child_hash, prior) in node.children.items():
                child = self.ttable[child_hash]
                prior_used = prior
                if root_prior_override is not None and node is root_node:
                    prior_used = root_prior_override.get(move, prior)
                s = self._ucb_score(node, child, prior_used)
                if s > best_score:
                    best_score = s
                    best_move = move
                    best_child = child

            assert best_move is not None and best_child is not None

            board.push(best_move)
            hist.append(board_to_planes(board))
            node = best_child
            path.append(node)

        # лист не терминальный → готовим obs/легальные ходы
        obs = board_to_obs(board, hist)  # np.ndarray
        legal_moves = list(board.legal_moves)
        return {
            "terminal": False,
            "board": board,
            "obs": obs,
            "legal_moves": legal_moves,
            "leaf_node": node,
            "path": path,
            "is_root_leaf": (node is root_node),
        }

    def _eval_expand_backup_batch(self, batch):
        # batch: list of dict from _traverse_to_leaf
        obs_batch = np.stack([x["obs"] for x in batch], axis=0)
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits, v_pred = self.model(obs_t)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            values = v_pred.detach().cpu().numpy().reshape(-1)

        # если нужно — применяем дирихле к root priors (только для тех элементов batch, где leaf==root)
        # на практике root расширяется в первом батче, поэтому это работает.
        for i, item in enumerate(batch):
            leaf = item["leaf_node"]
            board = item["board"]
            legal_moves = item["legal_moves"]

            # expand leaf: посчитать priors по probs[i]
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

            # записываем детей как ребра move -> (hash, prior)
            for mv in legal_moves:
                board.push(mv)
                child_h = self._board_hash(board)
                board.pop()

                self._get_or_create_node(child_h, player_to_move=(not leaf.player_to_move))
                leaf.children[mv] = (child_h, float(priors[mv]))

            value = float(values[i])
            self._backup(item["path"], value)

    def _backup(self, path: list[Node], value: float):
        v = float(value)
        for node in path:
            node.number_of_visits += 1
            node.value_sum += v
            node.mean_value = node.value_sum / node.number_of_visits
            v = -v

    def _select_action_from_root(self, root: Node):
        if not root.children:
            return None
        best_move = None
        best_visits = -1
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
