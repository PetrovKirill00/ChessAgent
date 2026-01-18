from __future__ import annotations

import os
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from constants import (
    CHECKPOINT_PATH,
    LAST_POSITIONS,
    WEB_MCTS_SIMULATIONS,
    CONTEMPT_DRAW,
    USE_FP16_INFERENCE,
    USE_CHANNELS_LAST,
)
from nw import AlphaZeroNet as CNNActorCritic
from agent import MCTS


app = FastAPI(
    title="Chess Agent Web API",
    description="Игра в шахматы против агента. Есть web UI и API.",
)


@app.on_event("shutdown")
def _on_shutdown():
    # Stop pondering threads (important for uvicorn --reload and restarts).
    with games_lock:
        for _gid, _g in list(games.items()):
            _stop_pondering(_g, join_timeout_s=0.1)
        games.clear()

# Optional: don't crash if static assets aren't present (API-only mode).
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    print("[WARN] 'static/' directory not found; Web UI assets will not be served.")


@dataclass
class GameState:
    id: str
    human_color: str
    board: chess.Board
    history: deque[chess.Board]
    mcts: MCTS = field(repr=False, compare=False)
    moves_san: list[str] = field(default_factory=list)
    moves_uci: list[str] = field(default_factory=list)
    status: str = "ongoing"
    result: str | None = None
    termination: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # --- concurrency / pondering ---
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    ponder_thread: threading.Thread | None = field(default=None, repr=False, compare=False)

    # --- per-engine-move simulation budget (minimum) ---
    engine_target_fen: str | None = None
    engine_target_sims: int | None = None


games: dict[str, GameState] = {}
games_lock = threading.RLock()


# -------- pondering helpers (web) --------
# We "ponder" (run MCTS continuously) even when it's not the engine's turn.
# For each engine move we sample a *minimum* simulation budget:
#   sims ~ Normal(mu=WEB_MCTS_SIMULATIONS, sigma=WEB_MCTS_SIMULATIONS/10)
# The engine may exceed this minimum if the opponent takes long (pondering accumulates extra sims).
_WEB_MCTS_SIGMA = float(WEB_MCTS_SIMULATIONS) / 10.0

# Small chunks keep the lock hold time short so UI requests stay responsive.
_PONDER_CHUNK_SIMS = 16
_PICK_CHUNK_SIMS = 64

# Yield after each ponder chunk to avoid starving other threads.
_PONDER_YIELD_S = 0.0


def _engine_color_bool(game: GameState) -> bool:
    return (not _human_color_bool(game.human_color))


def _sample_engine_budget_sims() -> int:
    # Truncated normal at >= 1. (Negative values are astronomically unlikely but still possible.)
    x = float(np.random.normal(loc=float(WEB_MCTS_SIMULATIONS), scale=_WEB_MCTS_SIGMA))
    n = int(round(x))
    return max(1, n)


def _ensure_history_current(game: GameState) -> None:
    b = game.board
    if not game.history or game.history[-1].fen() != b.fen():
        game.history.append(b.copy(stack=False))


def _ensure_engine_target(game: GameState) -> int:
    # Must be called under game.lock.
    if game.board.turn != _engine_color_bool(game):
        # Not engine to move -> target is irrelevant right now.
        return 0
    fen = game.board.fen()
    if game.engine_target_fen != fen or game.engine_target_sims is None:
        game.engine_target_fen = fen
        game.engine_target_sims = _sample_engine_budget_sims()
    return int(game.engine_target_sims)


def _mcts_run(game: GameState, sims: int) -> tuple[dict[chess.Move, float], dict[chess.Move, int]]:
    # Must be called under game.lock.
    _ensure_history_current(game)
    game.mcts.simulations = int(sims)
    return game.mcts.run(game.board, game.history, add_dirichlet_noise=False, reuse_tree=True)


def _ponder_loop(game: GameState) -> None:
    # Daemon thread: runs until stop_event or game finishes.
    while not game.stop_event.is_set():
        with game.lock:
            if game.status != "ongoing":
                break
            if game.board.is_game_over(claim_draw=True):
                break

            # Keep target synced for the current engine-to-move position (if applicable).
            _ = _ensure_engine_target(game)

            # One small chunk of search.
            _mcts_run(game, _PONDER_CHUNK_SIMS)

        # Yield so request handlers can grab the lock quickly.
        time.sleep(_PONDER_YIELD_S)


def _start_pondering(game: GameState) -> None:
    # Must be called once right after game creation.
    with game.lock:
        if game.ponder_thread is not None and game.ponder_thread.is_alive():
            return
        game.stop_event.clear()
        t = threading.Thread(target=_ponder_loop, args=(game,), daemon=True, name=f"ponder-{game.id[:8]}")
        game.ponder_thread = t
        t.start()


def _stop_pondering(game: GameState, *, join_timeout_s: float = 0.2):
    """Signal the pondering thread to stop and optionally join briefly.

    We keep this lightweight so request handlers won't block for long.
    """
    try:
        game.stop_event.set()
    except Exception:
        pass
    t = getattr(game, "ponder_thread", None)
    if t is not None and t.is_alive():
        try:
            t.join(timeout=join_timeout_s)
        except Exception:
            pass



# -------- model / mcts --------
def _pick_device() -> str:
    dev = os.getenv("DEVICE", "cpu")
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


device = torch.device(_pick_device())
model = CNNActorCritic().to(device).eval()

if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
else:
    # чтобы сервис запускался даже без весов
    print(f"[WARN] checkpoint not found: {CHECKPOINT_PATH}. Using random weights.")
    model.eval()


class LocalInferenceClient:
    """Local (in-process) inference client that mimics the central inference server.

    Why it exists:
      - agent.MCTS expects `position_deque` to contain chess.Board objects.
      - agent.MCTS supports sparse logits for legal moves only *only* via infer_client.
      - agent.MCTS local (infer_client=None) path does not handle idxs slicing.

    So the web API uses this client to keep compatibility and to work on CPU/GPU.
    """

    def __init__(self, model_: CNNActorCritic, device_: torch.device):
        self.model = model_
        self.device = device_
        self._lock = threading.Lock()
        # Leaf-parallel MCTS expects InferenceClient-like API (submit/wait) + prof counters.
        self._rq_lock = threading.Lock()
        self._rid = 0
        self._stash = {}  # rid -> (logits, value)
        self.prof = {"calls": 0, "wait_s": 0.0}



    def infer(self, obs_u8: np.ndarray, idxs_u16: np.ndarray | list[int] | None = None):
        # obs_u8 is (C,8,8) uint8 planes in {0,1}
        obs_f = obs_u8.astype(np.float32, copy=False)
        x = torch.from_numpy(obs_f).unsqueeze(0).to(self.device, dtype=torch.float32)
        if self.device.type == "cuda" and USE_CHANNELS_LAST:
            x = x.contiguous(memory_format=torch.channels_last)

        with self._lock, torch.no_grad():
            if self.device.type == "cuda" and USE_FP16_INFERENCE:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pol_logits, _v_unused, wdl_logits = self.model(x)
            else:
                pol_logits, _v_unused, wdl_logits = self.model(x)

            # Contempt-aware scalar value (matches main.py server mapping)
            probs = torch.softmax(wdl_logits.float(), dim=1)  # (1,3)
            value_t = (probs[:, 0] - probs[:, 2] + (CONTEMPT_DRAW * probs[:, 1]))
            value = float(value_t.squeeze(0).item())

            pol_row = pol_logits.squeeze(0)
            if idxs_u16 is None:
                logits_t = pol_row
            else:
                idxs_np = (idxs_u16.astype(np.int64, copy=False) if isinstance(idxs_u16, np.ndarray)
                           else np.asarray(idxs_u16, dtype=np.int64))
                idxs_t = torch.as_tensor(idxs_np, device=pol_row.device, dtype=torch.long)
                logits_t = pol_row.index_select(0, idxs_t)

            logits = logits_t.detach().cpu().numpy()

        return logits, value



    def _next_rid(self) -> int:
        with self._rq_lock:
            self._rid += 1
            return self._rid

    def submit(self, obs: np.ndarray, legal_idxs: np.ndarray | list[int] | None = None, *, _suppress_first_print: bool = False) -> int:
        """Submit one inference request and return rid.

        web_api runs inference locally; this exists to satisfy leaf-parallel MCTS
        which calls infer_client.submit()/wait().
        """
        rid = self._next_rid()
        t0 = time.perf_counter()
        logits, value = self.infer(obs, legal_idxs)
        dt = time.perf_counter() - t0
        with self._rq_lock:
            self.prof["calls"] += 1
            self.prof["wait_s"] += float(dt)
            self._stash[rid] = (logits, float(value))
        return rid

    def wait(self, rid: int):
        """Return (logits, value) for previously submitted rid."""
        with self._rq_lock:
            if rid in self._stash:
                return self._stash.pop(rid)
        raise KeyError(f"Unknown rid={rid}. Did you call submit()?")

infer_client = LocalInferenceClient(model, device)


def _engine_pick_move(game: GameState) -> tuple[chess.Move | None, dict[chess.Move, float]]:
    """Pick best move for the engine.

    Guarantees a per-move *minimum* MCTS budget sampled from Normal(WEB_MCTS_SIMULATIONS, WEB_MCTS_SIMULATIONS/10),
    but the actual number of simulations may be higher if pondering accumulated extra search.
    """
    with game.lock:
        board = game.board
        if board.is_game_over(claim_draw=True):
            return None, {}

        engine_bool = _engine_color_bool(game)
        if board.turn != engine_bool:
            # Not engine's turn (caller error).
            return None, {}

        # Ensure target for this exact position.
        target = _ensure_engine_target(game)

        # Query current search stats WITHOUT adding simulations.
        policy, visit_counts = _mcts_run(game, sims=0)
        sims_done = int(sum(visit_counts.values()))

    # Add more simulations until we reach the sampled minimum.
    while sims_done < target:
        with game.lock:
            if game.board.is_game_over(claim_draw=True):
                return None, {}
            if game.board.turn != _engine_color_bool(game):
                return None, {}

            remain = int(target - sims_done)
            chunk = int(min(_PICK_CHUNK_SIMS, remain))
            policy, visit_counts = _mcts_run(game, sims=chunk)
            sims_done = int(sum(visit_counts.values()))

    # Deterministic best move from final policy.
    if policy:
        mv = max(policy.items(), key=lambda kv: kv[1])[0]
    else:
        # Fallback: random legal move.
        with game.lock:
            legal = list(game.board.legal_moves)
        mv = random.choice(legal) if legal else None

    return mv, policy


# -------- schemas --------
class NewGameRequest(BaseModel):
    human_color: str = "w"  # "w" or "b"


class MoveRequest(BaseModel):
    game_id: str
    from_square: str
    to_square: str
    promotion: str | None = None  # "q/r/b/n" or None


class EngineMoveRequest(BaseModel):
    game_id: str


class MoveResponse(BaseModel):
    game_id: str
    board_fen: str
    engine_move: str | None
    moves_san: list[str]
    game_over: bool
    result: str | None
    termination: str | None



def _promotion_piece(p: str | None):
    if not p:
        return None
    promo_map = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
    out = promo_map.get(p.lower())
    if out is None:
        raise HTTPException(status_code=400, detail="Некорректная фигура для превращения (q/r/b/n)")
    return out



def _outcome_info(board: chess.Board):
    if not board.is_game_over(claim_draw=True):
        return False, None, None
    outcome = board.outcome(claim_draw=True)
    result = board.result(claim_draw=True)
    termination = outcome.termination.name if outcome is not None else None
    return True, result, termination


def _human_color_bool(human_color: str) -> bool:
    if human_color not in ("w", "b"):
        raise HTTPException(status_code=400, detail="human_color must be 'w' or 'b'")
    return chess.WHITE if human_color == "w" else chess.BLACK


def _new_game_state(human_color: str) -> GameState:
    board = chess.Board()
    hist = deque(maxlen=LAST_POSITIONS)
    hist.append(board.copy(stack=False))

    # Each game gets its own MCTS instance so tree reuse doesn't leak between games.
    mcts_inst = MCTS(model=model, infer_client=infer_client, simulations=WEB_MCTS_SIMULATIONS)
    game = GameState(id=uuid4().hex, human_color=human_color, board=board, history=hist, mcts=mcts_inst)

    # Start continuous pondering immediately.
    _start_pondering(game)
    return game


def _save_move(game: GameState, board: chess.Board, mv: chess.Move):
    # Note: game.lock is an RLock; it's OK if caller already holds it.
    with game.lock:
        san = board.san(mv)
        board_before = board.copy(stack=False)
        board.push(mv)

        # Keep MCTS root in sync for fast reuse.
        if hasattr(game.mcts, "advance_root"):
            try:
                game.mcts.advance_root(board_before, mv)
            except Exception:
                # If tree reuse fails for any reason, just reset.
                if hasattr(game.mcts, "reset"):
                    game.mcts.reset()

        # New position => new target (will be sampled when engine is to move).
        game.engine_target_fen = None
        game.engine_target_sims = None

        game.moves_san.append(san)
        game.moves_uci.append(mv.uci())
        game.history.append(board.copy(stack=False))


def _update_game_status(game: GameState, board: chess.Board):
    with game.lock:
        game_over, result, termination = _outcome_info(board)
        if game_over:
            game.status = "finished"
            game.result = result
            game.termination = termination
            try:
                game.stop_event.set()
            except Exception:
                pass
        else:
            game.status = "ongoing"
            game.result = None
            game.termination = None
        return game_over, result, termination


def _get_game(game_id: str) -> GameState:
    with games_lock:
        game = games.get(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


# -------- endpoints --------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/new_game", response_model=MoveResponse)
def new_game(req: NewGameRequest):
    human_color = req.human_color if req.human_color in ("w", "b") else "w"
    human_bool = _human_color_bool(human_color)

    # This web UI expects a single active game. Stop old pondering threads so they
    # don't keep consuming inference/GPU and slowing down the new game.
    with games_lock:
        for _gid, _g in list(games.items()):
            _stop_pondering(_g)
        games.clear()

    game = _new_game_state(human_color)
    board = game.board

    engine_move_uci = None

    # Если человек играет чёрными — движок делает первый ход сразу
    if human_bool == chess.BLACK:
        mv, _policy = _engine_pick_move(game)
        if mv is None:
            raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")
        _save_move(game, board, mv)
        engine_move_uci = mv.uci()

    game_over, result, termination = _update_game_status(game, board)
    with games_lock:
        games[game.id] = game

    with game.lock:
        fen = board.fen()
        moves_san = list(game.moves_san)

    return MoveResponse(
        game_id=game.id,
        board_fen=fen,
        engine_move=engine_move_uci,
        moves_san=moves_san,
        game_over=game_over,
        result=result,
        termination=termination,
    )


@app.post("/make_move", response_model=MoveResponse)
def make_move(req: MoveRequest):
    game = _get_game(req.game_id)
    board = game.board

    with game.lock:
        if board.is_game_over(claim_draw=True):
            game_over, result, termination = _update_game_status(game, board)
            return MoveResponse(
                game_id=game.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=list(game.moves_san),
                game_over=game_over,
                result=result,
                termination=termination,
            )

        human_bool = _human_color_bool(game.human_color)
        if board.turn != human_bool:
            raise HTTPException(status_code=400, detail="Сейчас не ход человека")

        try:
            from_sq = chess.parse_square(req.from_square)
            to_sq = chess.parse_square(req.to_square)
        except ValueError:
            raise HTTPException(status_code=400, detail="Некорректный формат клетки")

        promo = _promotion_piece(req.promotion)

        move_plain = chess.Move(from_sq, to_sq)
        move_promo = chess.Move(from_sq, to_sq, promotion=promo) if promo is not None else None

        if move_plain in board.legal_moves:
            user_move = move_plain
        elif move_promo is not None and move_promo in board.legal_moves:
            user_move = move_promo
        else:
            raise HTTPException(status_code=400, detail="Невозможный ход")

    _save_move(game, board, user_move)

    with game.lock:
        if board.is_game_over(claim_draw=True):
            game_over, result, termination = _update_game_status(game, board)
            return MoveResponse(
                game_id=game.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=list(game.moves_san),
                game_over=game_over,
                result=result,
                termination=termination,
            )

    engine_move, _policy = _engine_pick_move(game)
    if engine_move is None:
        raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")

    _save_move(game, board, engine_move)

    game_over, result, termination = _update_game_status(game, board)

    with game.lock:
        fen = board.fen()
        moves_san = list(game.moves_san)

    return MoveResponse(
        game_id=game.id,
        board_fen=fen,
        engine_move=engine_move.uci(),
        moves_san=moves_san,
        game_over=game_over,
        result=result,
        termination=termination,
    )


@app.post("/engine_move", response_model=MoveResponse)
def engine_move(req: EngineMoveRequest):
    game = _get_game(req.game_id)
    board = game.board

    with game.lock:
        if board.is_game_over(claim_draw=True):
            game_over, result, termination = _update_game_status(game, board)
            return MoveResponse(
                game_id=game.id,
                board_fen=board.fen(),
                engine_move=None,
                moves_san=list(game.moves_san),
                game_over=game_over,
                result=result,
                termination=termination,
            )

        human_bool = _human_color_bool(game.human_color)
        if board.turn == human_bool:
            raise HTTPException(status_code=400, detail="Сейчас ход человека, движок ходить не должен")

    mv, _policy = _engine_pick_move(game)
    if mv is None:
        raise HTTPException(status_code=500, detail="Движок не смог выбрать ход (MCTS вернул None)")

    _save_move(game, board, mv)

    game_over, result, termination = _update_game_status(game, board)

    with game.lock:
        fen = board.fen()
        moves_san = list(game.moves_san)

    return MoveResponse(
        game_id=game.id,
        board_fen=fen,
        engine_move=mv.uci(),
        moves_san=moves_san,
        game_over=game_over,
        result=result,
        termination=termination,
    )


@app.get("/games/{game_id}")
def get_game(game_id: str):
    game = _get_game(game_id)
    with game.lock:
        return {
            "game_id": game.id,
            "status": game.status,
            "human_color": game.human_color,
            "result": game.result,
            "termination": game.termination,
            "fen": game.board.fen(),
            "created_at": game.created_at.isoformat(),
        }


@app.get("/games/{game_id}/moves")
def get_moves(game_id: str):
    game = _get_game(game_id)
    with game.lock:
        out = []
        for idx, (uci, san) in enumerate(zip(game.moves_uci, game.moves_san), start=1):
            out.append({"ply": idx, "uci": uci, "san": san})
        return out


# -------- Web UI --------
@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Шахматы с агентом</title>
    <link rel="stylesheet" href="/static/css/chessboard-1.0.0.min.css">
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .top-bar { display:flex; gap:10px; align-items:center; }
        .layout { display:flex; gap:20px; align-items:flex-start; margin-top:20px; }
        #board { width: 480px; }
        #status { margin-top: 10px; min-height: 20px; }
        #fen { margin-top: 10px; background:#f5f5f5; padding:5px; font-family: monospace; }
        .moves-panel { min-width: 240px; font-size: 14px; }
        .moves-panel table { border-collapse: collapse; width: 100%; }
        .moves-panel th, .moves-panel td { padding: 2px 4px; border-bottom: 1px solid #ddd; text-align:left; }
        .moves-panel th { font-weight: 600; }
    </style>
</head>
<body>
    <h1>Шахматы с агентом</h1>

    <div class="top-bar">
        <button onclick="newGame()">Новая партия</button>
        <label>
            Цвет:
            <select id="colorSelect">
                <option value="w">Белыми</option>
                <option value="b">Чёрными</option>
            </select>
        </label>
        <span id="status"></span>
    </div>

    <div class="layout">
        <div>
            <div id="board"></div>
            <pre id="fen"></pre>
        </div>
        <div id="moves" class="moves-panel">
            <h3>Ходы</h3>
            <p>Пока ходов нет.</p>
        </div>
    </div>

    <script src="/static/js/jquery-3.7.1.min.js"></script>
    <script src="/static/js/chess.min.js"></script>
    <script src="/static/js/chessboard-1.0.0.min.js"></script>

    <script>
        let boardUI = null;
        let game = null;
        let gameId = null;

        let isMyTurn = true;
        let moves = [];
        let gameResult = null;
        let termination = null;
        let humanColor = "w";

        function setStatus(msg, isError) {
            const el = document.getElementById('status');
            el.textContent = msg || "";
            el.style.color = isError ? "red" : "black";
        }

        function updateFen() {
            const fenEl = document.getElementById('fen');
            fenEl.textContent = game ? game.fen() : "";
        }

        function renderMoveList() {
            const container = document.getElementById('moves');
            if (!container) return;

            if (!moves || moves.length === 0) {
                container.innerHTML = "<h3>Ходы</h3><p>Пока ходов нет.</p>";
                return;
            }

            let html = "<h3>Ходы</h3><table>";
            html += "<tr><th>#</th><th>Белые</th><th>Чёрные</th></tr>";

            for (let i = 0; i < moves.length; i += 2) {
                const moveNum = (i / 2) + 1;
                const white = moves[i] || "";
                const black = moves[i + 1] || "";
                html += "<tr><td>" + moveNum + "</td><td>" + white + "</td><td>" + black + "</td></tr>";
            }

            html += "</table>";
            if (gameResult) {
                const extra = termination ? (" (" + termination + ")") : "";
                html += "<div style='margin-top:6px;font-weight:700;'>Результат: " + gameResult + extra + "</div>";
            }

            container.innerHTML = html;
        }

        function onDragStart(source, piece) {
            if (!game) return false;
            if (game.game_over()) return false;
            if (!isMyTurn) return false;

            const isWhitePiece = piece[0] === "w";
            const isBlackPiece = piece[0] === "b";
            if (humanColor === "w" && isBlackPiece) return false;
            if (humanColor === "b" && isWhitePiece) return false;
        }

        function onDrop(source, target) {
          if (!game || !gameId) return "snapback";
          if (source === target) return "snapback";

          // базовый ход БЕЗ promotion
          let moveConfig = { from: source, to: target };

          // promotion prompt
          const piece = game.get(source);
          let promo = null;

          if (piece && piece.type === "p") {
            const fromRank = source[1], toRank = target[1];
            const isWhitePromotion = (piece.color === "w" && fromRank === "7" && toRank === "8");
            const isBlackPromotion = (piece.color === "b" && fromRank === "2" && toRank === "1");
            if (isWhitePromotion || isBlackPromotion) {
              promo = window.prompt("Превращение (q, r, b, n):", "q");
              if (!promo) promo = "q";
              promo = promo.toLowerCase();
              if (!["q","r","b","n"].includes(promo)) promo = "q";
              moveConfig.promotion = promo; // добавляем ТОЛЬКО здесь
            }
          }

          // локальная проверка chess.js
          const mv = game.move(moveConfig);
          if (mv === null) return "snapback";

          boardUI.position(game.fen());
          updateFen();

          isMyTurn = false;
          sendMoveToServer(source, target, promo); // promo может быть null
        }

        async function sendMoveToServer(from, to, promotion) {
            try {
                const res = await fetch("/make_move", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        game_id: gameId,
                        from_square: from,
                        to_square: to,
                        promotion: promotion || null
                    })
                });

                const data = await res.json();

                if (!res.ok) {
                    const msg = data.detail || JSON.stringify(data);
                    setStatus("Ошибка сервера: " + msg, true);
                    // откат локального хода
                    game.undo();
                    boardUI.position(game.fen());
                    updateFen();
                    isMyTurn = true;
                    return;
                }

                gameId = data.game_id;
                game.load(data.board_fen);
                boardUI.position(game.fen());
                updateFen();

                moves = data.moves_san || [];
                gameResult = data.result || null;
                termination = data.termination || null;
                renderMoveList();

                if (data.game_over) {
                    setStatus("Игра окончена. Результат: " + data.result + (data.termination ? (" ("+data.termination+")") : ""), false);
                    isMyTurn = false;
                } else {
                    isMyTurn = true;
                    setStatus("Ход принят. Движок ответил: " + (data.engine_move || "?"), false);
                }
            } catch (err) {
                setStatus("Ошибка при запросе к серверу: " + err.message, true);
                game.undo();
                boardUI.position(game.fen());
                updateFen();
                isMyTurn = true;
            }
        }

        async function newGame() {
            setStatus("", false);
            try {
                const select = document.getElementById("colorSelect");
                humanColor = select ? select.value : "w";

                const res = await fetch("/new_game", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ human_color: humanColor })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || ("HTTP " + res.status));

                if (!boardUI) {
                    boardUI = Chessboard("board", {
                        draggable: true,
                        position: "start",
                        orientation: "white",
                        onDragStart: onDragStart,
                        onDrop: onDrop,
                        pieceTheme: "/static/img/chesspieces/wikipedia/{piece}.png"
                    });
                }

                gameId = data.game_id;
                boardUI.orientation(humanColor === "w" ? "white" : "black");

                game = new Chess(data.board_fen);
                boardUI.position(game.fen());
                updateFen();

                moves = data.moves_san || [];
                gameResult = data.result || null;
                termination = data.termination || null;
                renderMoveList();

                // после /new_game, если человек чёрными — движок уже сделал первый ход
                isMyTurn = true;

                if (humanColor === "w") {
                    setStatus("Новая партия начата. Ты играешь белыми.", false);
                } else {
                    setStatus("Новая партия начата. Ты играешь чёрными. Движок уже сделал первый ход.", false);
                }
            } catch (err) {
                setStatus("Ошибка при создании новой партии: " + err.message, true);
            }
        }
    </script>
</body>
</html>"""
