import chess

BOARD_LAYERS = 12
EN_PASSANT_LAYERS = 1
CASTLES_LAYERS = 4
TURN_LAYERS = 1
LAST_POSITIONS = 8
META_LAYERS = EN_PASSANT_LAYERS + CASTLES_LAYERS + TURN_LAYERS
TOTAL_LAYERS = META_LAYERS + LAST_POSITIONS * BOARD_LAYERS
CHECKPOINT_PATH = "checkpoints/alphazero_like.pth"

TOTAL_MOVES = 4672

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,  # короля обычно не считаем в материале
}

TRAINING_MCTS_SIMULATIONS=256
TRAINING_MAX_MOVES=200_000
INFERENCE_BATCH_SIZE = 256
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.35
CONTEMPT_AGAINST_DRAW = 0
REPETITION_PENALTY = -0.5
THREEFOLD = True

# temperature schedule (self-play)
TEMPERATURE_MOVES = 512
TEMPERATURE_TAU_START = 2     # tau на самом старте партии
TEMPERATURE_TAU_END   = 0.2    # минимальный tau перед переходом в greedy
TEMPERATURE_DECAY_PLY = 384      # за сколько полуходов (ply) опускаем tau до end


# ================== Гиперпараметры буфера ==================

REPLAY_CAPACITY = 700_000       # максимум позиций в буфере
MIN_REPLAY_SIZE = 3_000        # с какого размера буфера начинаем full-обучение
BATCH_SIZE = 256               # размер минибатча
TRAIN_STEPS_PER_ITER = 32      # сколько SGD-шагов на одну итерацию
DEFAULT_REPLAY_PATH = "replay_buffer/replay_buffer.npz"
# ===========================================================

WEB_MCTS_SIMULATIONS = 800