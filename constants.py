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

SAVE_MODEL_PER_GAMES = 100
TRAINING_MCTS_SIMULATIONS=128
THINK_LONGER_AS_GAME_GOES = False
THINK_COEFF = 1.2
TRAINING_MAX_MOVES=512
INFERENCE_BATCH_SIZE = 256
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
CONTEMPT_AGAINST_DRAW = -0.5
REPETITION_PENALTY = -1.0
THREEFOLD = True

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0

# temperature schedule (self-play)
TEMPERATURE_MOVES = 384
TEMPERATURE_TAU_START = 3.0     # tau на самом старте партии
TEMPERATURE_TAU_END   = 0.15    # минимальный tau перед переходом в greedy
TEMPERATURE_DECAY_PLY = 256     # за сколько полуходов (ply) опускаем tau до end


# ================== Гиперпараметры буфера ==================
REPLAY_CAPACITY = 300_000       # максимум позиций в буфере
MIN_REPLAY_SIZE = 30_000        # с какого размера буфера начинаем full-обучение
BATCH_SIZE = 256               # размер минибатча
TRAIN_STEPS_PER_ITER = 2      # сколько SGD-шагов на одну итерацию
DEFAULT_REPLAY_PATH = "replay_buffer/replay_buffer.npz"
SAVE_BUFFER_PER_GAMES = 100
# ===== Stratified replay (mate vs draw) =====
MATE_BUFFER_FRACTION = 0.5   # доля REPLAY_CAPACITY под mate_buffer
P_MATE_IN_BATCH = 0.5        # доля mate-позиций в каждом batch при sample
# ==========================================
WEB_MCTS_SIMULATIONS = 1600