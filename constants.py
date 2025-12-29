# constants.py
# -*- coding: utf-8 -*-

"""
Central config for the chess self-play + training loop.

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

# -----------------------------
# Performance toggles
# -----------------------------
# Use FP16 autocast for inference on CUDA (обычно быстрее на RTX 40xx)
USE_FP16_INFERENCE = True

# Use channels_last memory format for CNN (может ускорять cuDNN conv)
USE_CHANNELS_LAST = True


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
MCTS_SIMULATIONS = 200
MCTS_C_PUCT = 1.5

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
REPLAY_CAPACITY = 300_000
P_MATE_IN_BATCH = 0.5

# -----------------------------
# Training
# -----------------------------
BATCH_SIZE = 1024
TRAIN_STEPS_PER_GAME = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

# -----------------------------
# Central inference server (main process)
# -----------------------------
NUM_SELFPLAY_WORKERS = 64

SERVER_MAX_INFERENCE_POSITIONS = 256      # max batch size
SERVER_BATCH_TIMEOUT_S = 0.010            # batching latency
SERVER_TICK_EVERY_S = 5.0                 # console tick
SERVER_DRAIN_GAMES_PER_TICK = 4           # how many finished games we consume per tick
SERVER_IDLE_SLEEP_S = 0.001               # main loop sleep if nothing to do

# -----------------------------
# Saving
# -----------------------------
SAVE_MODEL_PER_GAMES = 50
SAVE_BUFFER_PER_GAMES = 50

# -----------------------------
# Evaluation + gating
# -----------------------------
EVAL_EVERY_GAMES = 1
EVAL_NUM_GAMES = 50
EVAL_MAX_GAME_LENGTH = 768

# Promote if lower bound of score CI >= threshold.
# Score uses chess scoring: win=1, draw=0.5, loss=0.
EVAL_PROMOTE_SCORE = 0.55
EVAL_SCORE_Z = 1.96


# Threads for Torch CPU inference during evaluation (best vs cand)
EVAL_TORCH_THREADS = 8
# Backwards-typo alias
ELAV_TORCH_THREADS = EVAL_TORCH_THREADS
# MCTS for eval (keep smaller than self-play by default)
EVAL_MCTS_SIMULATIONS = 128
WEB_MCTS_SIMULATIONS = 1600

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
