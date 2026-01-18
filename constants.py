# constants.py
# -*- coding: utf-8 -*-

"""
Central config for the chess self-play + training loop.

NOTE (hardcoded config):
- This file intentionally DOES NOT read environment variables.
- To change hyperparameters / debug flags, edit the constants below.

Key design decisions (current "WD L" setup):
- Network outputs:
    * policy_logits: (B, TOTAL_MOVES)
    * wdl_logits:    (B, 3)  classes [WIN, DRAW, LOSS] from side-to-move perspective
    * value:         scalar derived from wdl (used in MCTS only)
- Training uses:
    * policy loss: cross-entropy with target visit-count distribution π
    * value loss:  cross-entropy on WDL classes
- "Contempt" is implemented in search-time value mapping:
    v = P(win) - P(loss) + CONTEMPT_DRAW * P(draw)
  This keeps WDL training intact, but biases MCTS away from drawish lines.
"""

import chess

# -----------------------------
# Observation encoding
# -----------------------------
BOARD_LAYERS = 12                  # piece planes: 6 for white + 6 for black
EN_PASSANT_LAYERS = 1              # file of en-passant square (or all zeros)
CASTLES_LAYERS = 4                 # WK, WQ, BK, BQ
TURN_LAYERS = 1                    # side to move

LAST_POSITIONS = 8                 # how many previous positions are stacked

META_LAYERS = EN_PASSANT_LAYERS + CASTLES_LAYERS + TURN_LAYERS
TOTAL_LAYERS = META_LAYERS + LAST_POSITIONS * BOARD_LAYERS   # 6 + 8*12 = 102

# -----------------------------
# Action encoding
# -----------------------------
TOTAL_MOVES = 4672

# -----------------------------
# Paths
# -----------------------------
CHECKPOINT_PATH = "checkpoints/alphazero_like.pth"     # candidate
BEST_CHECKPOINT_PATH = "checkpoints/best.pth"          # baseline

REPLAY_PATH = "replay_buffer/replay_buffer.npz"
DEFAULT_REPLAY_PATH = REPLAY_PATH

# -----------------------------
# Self-play
# -----------------------------
MAX_GAME_LENGTH = 2048

# MCTS
MCTS_SIMULATIONS = 256
MCTS_C_PUCT = 1.5

# Leaf-parallelism inside one worker's MCTS (micro-batching NN evals).
# Used only when InferenceClient is present. 1 disables leaf-parallelism.
MCTS_LEAF_PARALLELISM = 64

DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS = 0.25

TEMPERATURE_MOVES = 10
TEMPERATURE_TAU_START = 1.25
TEMPERATURE_TAU_END = 0.25

# Contempt (search-time only)
CONTEMPT_DRAW = -0.05

# -----------------------------
# Replay buffer
# -----------------------------
REPLAY_CAPACITY = 700_000
REPLAY_TARGET_MATE_FRACTION = 0.8  # target fraction of mate samples stored (rest are draws)
P_MATE_IN_BATCH = 0.8

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 1024
TRAIN_STEPS_PER_GAME = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# -----------------------------
# IPC queue sizing (perf knobs)
# -----------------------------
# NOTE: these queues can hold large payloads (obs tensors, logits arrays, game samples).
# Keep GAME_Q_MAXSIZE moderate; request/response can be larger.
REQUEST_Q_MAXSIZE = 4096
RESPONSE_Q_MAXSIZE = 128
GAME_Q_MAXSIZE = 64

# IPC payload minimization (reduces Python multiprocessing overhead).
IPC_OBS_UINT8 = True                # send obs planes as uint8 (0/1) instead of float32
IPC_SEND_ONLY_LEGAL = True          # send only logits for legal moves (requires main.py support)

# -----------------------------
# Central inference server (main process)
# -----------------------------
NUM_SELFPLAY_WORKERS = 16
SERVER_MAX_INFERENCE_POSITIONS = 1024      # max batch size
SERVER_BATCH_TIMEOUT_S = 0.002             # batching latency
SERVER_TICK_EVERY_S = 5.0                  # console tick
SERVER_DRAIN_GAMES_PER_TICK = 4            # how many finished games we consume per tick
SERVER_IDLE_SLEEP_S = 0.001                # main loop sleep if nothing to do
SERVER_RESPONSE_PUT_TIMEOUT_S = 0.5        # timeout for putting responses to workers

# -----------------------------
# Saving
# -----------------------------
SAVE_MODEL_PER_GAMES = 100
SAVE_BUFFER_PER_GAMES = 1000

ASYNC_REPLAY_SAVE = True  # save replay buffer in background thread
REPLAY_SAVE_COMPRESSED = False  # faster saves; larger file
TRAIN_TIME_BUDGET_S = 0.25  # 0.0 to disable time budget

# -----------------------------
# Evaluation + gating
# -----------------------------
EVAL_EVERY_GAMES = 3001
EVAL_NUM_GAMES = 50
EVAL_MAX_GAME_LENGTH = 768


# Exclusive evaluation mode:
# - When eval is running, pause self-play and (optionally) pause training,
#   so all compute (GPU inference + CPU MCTS) goes to evaluation.
EVAL_EXCLUSIVE_MODE = True

# If True, evaluation uses the same IPC inference server as self-play (GPU + batching).
# This requires two extra response queues (candidate+best) and model routing by model_id.
EVAL_USE_IPC_INFERENCE = True

# If True, skip training steps while eval is running (recommended to keep candidate fixed).
EVAL_PAUSE_TRAINING = True

# Sleep used by self-play workers while paused (exclusive eval).
SELFPLAY_PAUSE_SLEEP_S = 0.01

# Promote if lower bound of score CI >= threshold.
# Score uses chess scoring: win=1, draw=0.5, loss=0.
EVAL_PROMOTE_SCORE = 0.55
EVAL_SCORE_Z = 1.96

# MCTS for eval (keep smaller than self-play by default)
EVAL_MCTS_SIMULATIONS = 128

# -----------------------------
# Web UI / API (web_api.py)
# -----------------------------
WEB_MCTS_SIMULATIONS = 1024

# Mixed precision for inference (only affects forward pass under CUDA).
USE_FP16_INFERENCE = True

# If True, try `channels_last` for model + input tensors (may improve throughput on some GPUs).
USE_CHANNELS_LAST = False

# -----------------------------
# Debug flags (grouped)
# -----------------------------
DEBUG_PREFIX_TIME = True

# 1) games stats
DEBUG_GAMES = True

# 2) infer_pos/5s, gpm
DEBUG_INFER = True

# 3) evaluation + gating
DEBUG_EVALUATION = True

# 4) replay_buffer sizes
DEBUG_REPLAY = True

# 5) losses (policy/value/total)
DEBUG_LOSSES = True

# 6) draw reason stats
DEBUG_DRAW_REASON = True

# Extra one-shot worker prints
DEBUG_PRINT_WORKER_FIRST_INFER = True
DEBUG_PRINT_WORKER_FIRST_GAME = True
DEBUG_PRINT_EVAL_TERMINATIONS = True

# -----------------------------
# Debug flags expected by main.py (compat)
# -----------------------------
# main.py historically used names like DEBUG_EVAL/DEBUG_LOSS/DEBUG_DRAW; keep aliases.
DEBUG_EVAL = DEBUG_EVALUATION
DEBUG_LOSS = DEBUG_LOSSES
DEBUG_DRAW = DEBUG_DRAW_REASON

# IPC / queue diagnostics
DEBUG_IPC = True
DEBUG_IPC_QUEUES = True

# Extra CUDA timing instrumentation for the inference server.
# NOTE: enabling this may add CUDA synchronization overhead; keep it off unless debugging.
DEBUG_SERVER_TIMINGS = True

# Backward-compat for other potential debug gates
DEBUG_PROFILE = True

# IPC client timeouts
INFER_CLIENT_PUT_TIMEOUT_S = 0.1
INFER_CLIENT_GET_TIMEOUT_S = 0.1

# Worker-side diagnostics / timeouts (used by main.py)
DEBUG_WORKER_INFER_LATENCY = False

# Warn if a single inference round-trip (submit->wait) exceeds this threshold.
WORKER_INFER_WARN_AFTER_S = 1.0

# When game_q.put blocks, print warning at most once per this interval.
WORKER_GAME_PUT_WARN_EVERY_S = 2.0

# Timeout for pushing finished games from worker to the main process.
WORKER_GAME_PUT_TIMEOUT_S = 2.0

# Self-play profiling
DEBUG_SELFPLAY_PROFILE = True
SELFPLAY_PROFILE_EVERY_N_GAMES = 20
