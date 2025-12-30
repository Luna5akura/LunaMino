# test_tetris_engine.py
# Run this script in your project directory: python test_tetris_engine.py

from ai.utils import TetrisGame
from ai.reward import calculate_heuristics
import numpy as np

def print_board(board):
    """Pretty print the board."""
    for row in board:
        print(' '.join(['#' if cell > 0 else '.' for cell in row]))

game = TetrisGame(seed=42)
print("=== Initial State ===")
board, ctx = game.get_state()
print("Board:")
print_board(board)
print("Context:", ctx)
metrics = calculate_heuristics(board)
print("Heuristics:", metrics)
legal = game.get_legal_moves()
print("Legal moves count:", len(legal))
if len(legal) > 0:
    print("Sample move:", legal[0])

# Add garbage
game.receive_garbage(2)
print("\n=== After Adding 2 Garbage Lines ===")
board, ctx = game.get_state()
print("Board:")
print_board(board)
print("Context:", ctx)
metrics = calculate_heuristics(board)
print("Heuristics:", metrics)
legal = game.get_legal_moves()
print("Legal moves count:", len(legal))

if len(legal) == 0:
    print("No legal moves after adding garbage - possible engine issue.")
else:
    # Execute first legal move
    move = legal[0]
    print("\n=== Executing First Legal Move ===")
    print("Move:", move)
    res = game.step(move[0], move[1], move[2], move[4])
    print("Step Result:", res)
    board, ctx = game.get_state()
    print("New Board:")
    print_board(board)
    metrics = calculate_heuristics(board)
    print("New Heuristics:", metrics)
    print("Game Over:", res[3])

    if res[3]:
        print("Game over after first step - check if placement caused overflow or engine bug.")

# Optional: Enable render to visualize
# game.enable_render()
# while True:
#     game.render()
#     # Add logic to continue or break