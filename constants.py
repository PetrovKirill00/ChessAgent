import chess

# ================== Представление состояния ==================
BOARD_LAYERS = 12
EN_PASSANT_LAYERS = 1
CASTLES_LAYERS = 4
TURN_LAYERS = 1
LAST_POSITIONS = 8

META_LAYERS = EN_PASSANT_LAYERS + CASTLES_LAYERS + TURN_LAYERS
TOTAL_LAYERS = META_LAYERS + LAST_POSITIONS * BOARD_LAYERS

TOTAL_MOVES = 4672

CHECKPOINT_PATH = "checkpoints/alphazero_like.pth"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# ================== Self-play / MCTS ==================
SAVE_MODEL_PER_GAMES = 100

# DEBUG: уменьшаем, чтобы быстро увидеть первые партии
TRAINING_MCTS_SIMULATIONS = 128
THINK_LONGER_AS_GAME_GOES = False
THINK_COEFF = 0.8

# DEBUG: чтобы партия закончилась быстро и дошла до main
TRAINING_MAX_MOVES = 2000

# DEBUG: ключевой параметр — маленький batch, чтобы первый inference запрос прилетал быстро
INFERENCE_BATCH_SIZE = 16

DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

CONTEMPT_AGAINST_DRAW = -0.5
REPETITION_PENALTY = -1.0
THREEFOLD = False

# ================== Оптимизация ==================
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0

TEMPERATURE_MOVES = 128
TEMPERATURE_TAU_START = 3.0
TEMPERATURE_TAU_END = 0.3
TEMPERATURE_DECAY_PLY = 128

# ================== Replay buffer ==================
REPLAY_CAPACITY = 300_000
MIN_REPLAY_SIZE = 30_000
BATCH_SIZE = 256
TRAIN_STEPS_PER_ITER = 8
DEFAULT_REPLAY_PATH = "replay_buffer/replay_buffer.npz"
SAVE_BUFFER_PER_GAMES = 100

MATE_BUFFER_FRACTION = 0.5
P_MATE_IN_BATCH = 0.5

WEB_MCTS_SIMULATIONS = 1600

# ================== Multi-process + Central inference server ==================
NUM_SELFPLAY_WORKERS = 12

# DEBUG: маленькие батчи на сервере
SERVER_MAX_INFERENCE_POSITIONS = 128
SERVER_INFERENCE_WAIT_MS = 5

SERVER_DRAIN_GAMES_PER_TICK = 4
SERVER_TRAIN_STEPS_PER_GAME = 2
SERVER_IDLE_SLEEP_MS = 1
