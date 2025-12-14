import random
import time
from sys import setrecursionlimit

import torch
from nw import CNNActorCritic
import os
from agent import train_one_iteration
import chess
from constants import (
    CHECKPOINT_PATH,
    TRAINING_MCTS_SIMULATIONS,
    TRAINING_MAX_MOVES,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from replay_buffer import load_replay_buffer, save_replay_buffer


def main(device: str="cuda"):
    os.makedirs("checkpoints", exist_ok=True)

    model = CNNActorCritic().to(device)

    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {CHECKPOINT_PATH}")
    else:
        print("Model not loaded. Creating new one...")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    # Загрузка replay buffer, если файл существует
    print("Trying to load replay buffer...")
    if load_replay_buffer():
        print("Replay buffer loaded from disk.")
    else:
        print("No replay buffer found, starting with an empty buffer.")

    print("Starting infinite self-play training loop. Press Ctrl+C to stop.")

    i = 0
    total_elapsed = 0
    exceeded_move_limit_games = 0
    try:
        while True:
            i += 1
            time_start = time.perf_counter()
            stats = train_one_iteration(
                model,
                optimizer,
                num_simulations=TRAINING_MCTS_SIMULATIONS,
                max_moves=TRAINING_MAX_MOVES,
                device=device,
            )
            time_finish = time.perf_counter()
            elapsed = time_finish - time_start
            total_elapsed += elapsed

            if stats is None:
                continue

            exceeded_move_limit_games += int(stats.get("exceeded_move_limit", 0))

            print(
                f"Iter {i}: "
                f"loss={stats['loss']:.8f} "
                f"policy_loss={stats['policy_loss']:.8f} "
                f"value_loss={stats['value_loss']:.8f}\n"
                f"positions={stats['num_positions']} "
                f"buffer={stats.get('buffer_size', 0)} "
                f"steps={stats.get('train_steps', 0)} "
                f"train_pos_used={stats.get('positions_used_for_training', 0)}\n"
                f"elapsed={elapsed:.3f} seconds, "
                f"average per game={total_elapsed/i:.3f} seconds"
            )
            outcome = stats.get("outcome")
            if outcome is None:
                print("Exceeded move limit")
            elif outcome.winner is not None:
                if outcome.winner is chess.WHITE:
                    print("White won by checkmate")
                else:
                    print("Black won by checkmate")
            elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                print("Threefold repetition")
            elif outcome.termination == chess.Termination.STALEMATE:
                print("Stalemate occurred")
            elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                print("Draw called due to insufficient material")
            elif outcome.termination == chess.Termination.FIFTY_MOVES:
                print("Draw by 50 moves")
            elif outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
                print("Fivefold repetition")
            elif outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
                print("Draw by 75 moves")
            else:
                print("Something weird happened... Called draw")
            print()

            # периодически сохраняем модель и replay buffer
            if i % 10 == 0:
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"Checkpoint saved at iter {i}")
            if i % 100 == 0:
                save_replay_buffer()
                print(f"Replay buffer saved at iter {i}")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught. Saving final checkpoint and replay buffer...")
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        save_replay_buffer()
        print(f"Final model saved to {CHECKPOINT_PATH}")
        print("Replay buffer saved to replay_buffer.npz")

if __name__ == "__main__":
    # play_game_mcts_vs_random()
    # debug_once()
    # debug_self_play()
    # train_loop(num_iters=10, device="cuda")
    # play_game_mcts_nn_vs_random(device="cuda")
    main(device="cuda")

