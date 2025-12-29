# constants.py
# -*- coding: utf-8 -*-
"""Project-wide constants.

Key invariants (please keep these consistent with checkpoints / replay buffer):

Observation tensor shape
------------------------
We use `TOTAL_LAYERS = 102` planes of size 8x8:

History (LAST_POSITIONS * 12):
  - 12 piece planes per position: [P,N,B,R,Q,K] x [white, black]
  - We stack the last `LAST_POSITIONS` positions (including the current one).
    => 8 * 12 = 96 planes

Meta (6):
  - 1 plane: side-to-move (all 1.0 if white to move else 0.0)
  - 4 planes: castling rights (WK, WQ, BK, BQ)
  - 1 plane: en-passant square (one-hot on the ep square, else all zeros)

TOTAL: 96 + 6 = 102.

Move space
----------
AlphaZero-style fixed action space: 8x8x73 = 4672.
"""

# -----------------------------
# Paths
# -----------------------------
CHECKPOINT_PATH = "checkpoints/alphazero_like.pth"     # current training weights (candidate)
BEST_CHECKPOINT_PATH = "checkpoints/best.pth"          # frozen baseline for gating/eval

REPLAY_PATH = "replay_buffer/replay_buffer.npz"
DEFAULT_REPLAY_PATH = REPLAY_PATH  # compat: replay_buffer.py uses this name

EVAL_CANDIDATE_PATH = "tmp_eval/candidate_eval.pth"
EVAL_TMP_DIR = "tmp_eval"

# -----------------------------
# Observation encoding
# -----------------------------
BOARD_LAYERS = 12
LAST_POSITIONS = 8

TURN_LAYERS = 1
CASTLES_LAYERS = 4
EN_PASSANT_LAYERS = 1

TOTAL_LAYERS = LAST_POSITIONS * BOARD_LAYERS + TURN_LAYERS + CASTLES_LAYERS + EN_PASSANT_LAYERS  # 102

# -----------------------------
# Action space
# -----------------------------
TOTAL_MOVES = 4672  # 8*8*73

# -----------------------------
# Self-play game length
# -----------------------------
MAX_GAME_LENGTH = 2048

# -----------------------------
# MCTS (Monte Carlo Tree Search)
# -----------------------------
MCTS_SIMULATIONS = 128
WEB_MCTS_SIMULATIONS = 1600
EVAL_MCTS_SIMULATIONS = 256
MCTS_C_PUCT = 1.5

# Dirichlet noise (root exploration during training self-play)
DIRICHLET_ALPHA = 0.30
DIRICHLET_EPS = 0.25

# MCTS batching (within one worker/game). The central inference server
# will also batch across workers, but this helps avoid per-leaf round-trips.
INFERENCE_BATCH_SIZE = 16

# Temperature schedule (sampling from visit-count policy in early plies)
TEMPERATURE_MOVES = 30
TEMPERATURE_TAU_START = 1.25
TEMPERATURE_TAU_END = 0.25

# If True: claim draw by threefold repetition when checking termination.
# If False: repetitions won't immediately terminate; games rely on MAX_GAME_LENGTH.
THREEFOLD = True

# -----------------------------
# Optimization / training
# -----------------------------
# Train only after this many positions are present (avoids overfitting on tiny replay).
MIN_REPLAY_SIZE = 4096

# How many gradient steps to run per newly finished self-play game.
TRAIN_STEPS_PER_GAME = 4

BATCH_SIZE = 1024
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# -----------------------------
# Replay buffer
# -----------------------------
REPLAY_CAPACITY = 300_000

# Stratified replay: keep checkmates in a separate buffer.
MATE_BUFFER_FRACTION = 0.5
P_MATE_IN_BATCH = 0.5

# -----------------------------
# Multi-process self-play
# -----------------------------
NUM_SELFPLAY_WORKERS = 16

# -----------------------------
# Central inference server (main process)
# -----------------------------
SERVER_MAX_INFERENCE_POSITIONS = 128
SERVER_BATCH_TIMEOUT_S = 0.005

SERVER_TICK_EVERY_S = 5.0
SERVER_DRAIN_GAMES_PER_TICK = 16

# -----------------------------
# Checkpoint saving
# -----------------------------
SAVE_MODEL_PER_GAMES = 100
SAVE_BUFFER_PER_GAMES = 500

# -----------------------------
# Evaluation (arena + gating)
# -----------------------------
EVAL_EVERY_GAMES = 500
EVAL_NUM_GAMES = 50

# A secondary, human-readable metric.
ELO_FROM_SCORE_EPS = 1e-6
EVAL_PROMOTE_ELO = 70.0  # kept for logging / optional use

# Primary metric: score = (W + 0.5*D) / N (draw counts as half-win).
EVAL_PROMOTE_SCORE = 0.55   # promote if lower CI bound of score >= this
EVAL_SCORE_Z = 1.96         # ~95% normal-approx for score CI lower bound

EVAL_MAX_GAME_LENGTH = 768

# -----------------------------
# Debug
# -----------------------------
DEBUG_PRINT_WORKER_FIRST_INFER = True
DEBUG_PRINT_EVAL_TERMINATIONS = True
