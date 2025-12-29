import numpy as np
import chess
from collections import deque
from constants import TOTAL_LAYERS, TOTAL_MOVES, BOARD_LAYERS


def board_to_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((BOARD_LAYERS, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    mapping = {
        (chess.PAWN,   True): 0,
        (chess.KNIGHT, True): 1,
        (chess.BISHOP, True): 2,
        (chess.ROOK,   True): 3,
        (chess.QUEEN,  True): 4,
        (chess.KING,   True): 5,
        (chess.PAWN,   False): 6,
        (chess.KNIGHT, False): 7,
        (chess.BISHOP, False): 8,
        (chess.ROOK,   False): 9,
        (chess.QUEEN,  False): 10,
        (chess.KING,   False): 11,
    }

    for square, piece in piece_map.items():
        p = mapping[(piece.piece_type, piece.color)]
        r = chess.square_rank(square)
        c = chess.square_file(square)
        planes[p, r, c] = 1.0
    return planes

def board_to_obs(board: chess.Board, position_deque: deque):
    obs = np.zeros((TOTAL_LAYERS, 8, 8), dtype=np.float32)

    idx = 0

    if board.turn == chess.WHITE:
        obs[idx, :, :] = 1.0
    idx += 1

    # рокировки: WK, WQ, BK, BQ
    obs[idx, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    idx += 1
    obs[idx, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    idx += 1
    obs[idx, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    idx += 1
    obs[idx, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    idx += 1

    # эн-пассант
    if board.ep_square is not None:
        r = chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        obs[idx, r, c] = 1.0
    idx += 1

    for b in position_deque:
        obs[idx: idx + BOARD_LAYERS] = b
        idx += BOARD_LAYERS

    return obs

def action_id_to_move(board: chess.Board, action_id: int) -> chess.Move:
    """
    Декодируем индекс из [0, 4671] в ход python-chess (chess.Move).

    Схема:
    - 64 from-клетки * 73 плоскости.
    - 0..55 : 8 направлений * 7 расстояний (слайдинги).
    - 56..63: 8 ходов коня.
    - 64..72: 9 underpromotion (N/B/R) для пешек.
    """

    if action_id < 0 or action_id >= TOTAL_MOVES:
        return chess.Move.null()

    from_sq = action_id // 73              # 0..63
    plane = action_id % 73                # 0..72

    from_file = chess.square_file(from_sq)  # 0..7
    from_rank = chess.square_rank(from_sq)  # 0..7

    # ---------- 0..55: слайдинги ----------
    if plane < 56:
        dir_id = plane // 7               # 0..7
        k = plane % 7 + 1                 # 1..7

        # (df, dr): file, rank
        directions = [
            (1, 0),   # вправо
            (-1, 0),  # влево
            (0, 1),   # вверх
            (0, -1),  # вниз
            (1, 1),   # вверх-вправо
            (-1, 1),  # вверх-влево
            (1, -1),  # вниз-вправо
            (-1, -1), # вниз-влево
        ]
        df, dr = directions[dir_id]
        to_file = from_file + df * k
        to_rank = from_rank + dr * k

        # вышли за доску — считаем как null, потом отфильтруешь по legal_moves
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return chess.Move.null()

        to_sq = chess.square(to_file, to_rank)

        # особый случай: если это пешка, которая дошла до конца — авто-промо в ферзя
        piece = board.piece_at(from_sq)
        promotion = None
        if piece is not None and piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE and to_rank == 7:
                promotion = chess.QUEEN
            elif piece.color == chess.BLACK and to_rank == 0:
                promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    # ---------- 56..63: конь ----------
    if plane < 64:
        knight_id = plane - 56
        knight_moves = [
            (1, 2), (2, 1),
            (2, -1), (1, -2),
            (-1, -2), (-2, -1),
            (-2, 1), (-1, 2),
        ]
        df, dr = knight_moves[knight_id]
        to_file = from_file + df
        to_rank = from_rank + dr

        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return chess.Move.null()

        to_sq = chess.square(to_file, to_rank)
        return chess.Move(from_sq, to_sq)

    # ---------- 64..72: underpromotion N/B/R ----------
    promo_id = plane - 64                 # 0..8
    dir_id = promo_id // 3                # 0..2
    piece_id = promo_id % 3               # 0..2

    # под что промоутим (без ферзя: он покрыт "обычными" ходами выше)
    promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    promo_piece = promo_pieces[piece_id]

    # три направления для пешки
    # для белых: вперёд, взятие влево, взятие вправо
    # для чёрных: зеркально
    if board.turn == chess.WHITE:
        directions = [
            (0, 1),   # вперёд
            (-1, 1),  # взятие влево
            (1, 1),   # взятие вправо
        ]
        final_rank = 7
    else:
        directions = [
            (0, -1),
            (1, -1),
            (-1, -1),
        ]
        final_rank = 0

    df, dr = directions[dir_id]
    to_file = from_file + df
    to_rank = from_rank + dr

    if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
        return chess.Move.null()

    # underpromotion имеет смысл только с предпоследней горизонтали
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.PAWN:
        return chess.Move.null()

    if board.turn == chess.WHITE and from_rank != 6:
        return chess.Move.null()
    if board.turn == chess.BLACK and from_rank != 1:
        return chess.Move.null()

    if to_rank != final_rank:
        return chess.Move.null()

    to_sq = chess.square(to_file, to_rank)
    return chess.Move(from_sq, to_sq, promotion=promo_piece)

def move_to_id(board: chess.Board, move: chess.Move) -> int:
    """
    Обратное преобразование chess.Move -> action_id в [0, TOTAL_MOVES).

    Схема такая же, как в _decode_action:
    - 0..55  : слайдинги (8 направлений * 7 расстояний)
    - 56..63 : конь
    - 64..72 : underpromotion (N/B/R) пешек
    """

    # null-ход кодировать не будем
    if move is None or move == chess.Move.null():
        return -1

    from_sq = move.from_square
    to_sq = move.to_square

    if from_sq is None or to_sq is None:
        return -1

    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = chess.square_file(to_sq)
    to_rank = chess.square_rank(to_sq)

    df = to_file - from_file
    dr = to_rank - from_rank

    # ---------- 1) Underpromotion N/B/R (плоскости 64..72) ----------
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        # направления должны совпадать с _decode_action
        if board.turn == chess.WHITE:
            directions = [
                (0, 1),  # вперёд
                (-1, 1),  # взятие влево
                (1, 1),  # взятие вправо
            ]
            final_rank = 7
            start_rank = 6
        else:
            directions = [
                (0, -1),
                (1, -1),
                (-1, -1),
            ]
            final_rank = 0
            start_rank = 1

        # базовая валидация, как в _decode_action
        if chess.square_rank(from_sq) != start_rank or chess.square_rank(to_sq) != final_rank:
            return -1

        step = (df, dr)
        if step not in directions:
            return -1

        dir_id = directions.index(step)
        piece_id = promo_pieces.index(move.promotion)

        promo_id = dir_id * 3 + piece_id  # 0..8
        plane = 64 + promo_id  # 64..72

        action_id = from_sq * 73 + plane
        return action_id if 0 <= action_id < TOTAL_MOVES else -1

    # ---------- 2) Конь (плоскости 56..63) ----------
    knight_moves = [
        (1, 2), (2, 1),
        (2, -1), (1, -2),
        (-1, -2), (-2, -1),
        (-2, 1), (-1, 2),
    ]

    step = (df, dr)
    if step in knight_moves:
        knight_id = knight_moves.index(step)  # 0..7
        plane = 56 + knight_id  # 56..63
        action_id = from_sq * 73 + plane
        return action_id if 0 <= action_id < TOTAL_MOVES else -1

    # ---------- 3) Слайдинги (плоскости 0..55) ----------
    # те же directions, что в _decode_action
    directions = [
        (1, 0),  # вправо
        (-1, 0),  # влево
        (0, 1),  # вверх
        (0, -1),  # вниз
        (1, 1),  # вверх-вправо
        (-1, 1),  # вверх-влево
        (1, -1),  # вниз-вправо
        (-1, -1),  # вниз-влево
    ]

    # проверяем, что ход действительно "линейный"
    if df == 0 and dr == 0:
        return -1

    # шаг по направлению
    def sign(x: int) -> int:
        return int(x > 0) - int(x < 0)

    step = (sign(df), sign(dr))

    if step not in directions:
        # не слайдинг (и не конь, и не underpromotion) — не кодируем
        return -1

    # расстояние
    k_file = abs(df) if df != 0 else 0
    k_rank = abs(dr) if dr != 0 else 0
    k = max(k_file, k_rank)

    if k < 1 or k > 7:
        return -1

    dir_id = directions.index(step)  # 0..7
    plane = dir_id * 7 + (k - 1)  # 0..55

    action_id = from_sq * 73 + plane
    return action_id if 0 <= action_id < TOTAL_MOVES else -1
