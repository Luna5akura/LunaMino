# ai/test_clear_reward.py

import numpy as np
from .utils import TetrisGame
from .reward import get_reward, calculate_heuristics

def run_test(seed=42):
    game = TetrisGame(seed=seed)
    game.receive_garbage(1)
    
    # Dummy step to flush garbage (use first legal move, assume no clear on empty board)
    legal_dummy = game.get_legal_moves()
    if legal_dummy:
        dummy_move = legal_dummy[0]  # Arbitrary, should not clear
        res_dummy = game.step(dummy_move['x'], dummy_move['y'], dummy_move['rotation'], dummy_move['use_hold'])
        print("Dummy res lines:", res_dummy['lines_cleared'])  # Expect 0, flushes garbage
    
    # Now board has garbage; get prev
    prev_board, _, _ = game.get_state()
    prev_metrics = calculate_heuristics(prev_board)
    hole_pos = np.where(prev_board[-1] == 0)[0]
    print("Bottom row after dummy:", prev_board[-1])
    print("Hole position(s):", hole_pos if len(hole_pos) > 0 else "No hole - bug")
    
    _, _, p_type = game.get_state()
    print(f"Seed {seed}: Piece type: {p_type} (0=I)")
    
    legal = game.get_legal_moves()
    print(f"Legal count: {len(legal)}")
    
    clearing_moves = []
    for move in legal:
        clone = game.clone()
        res = clone.step(move['x'], move['y'], move['rotation'], move['use_hold'])
        next_board, _, _ = clone.get_state()
        cur_metrics = calculate_heuristics(next_board)
        reward, _ = get_reward(res, cur_metrics, prev_metrics, is_training=True)
        
        if res['lines_cleared'] > 0:
            clearing_moves.append((move, res['lines_cleared'], reward))
            print(f"Clearing move: {move}, lines={res['lines_cleared']}, reward={reward:.2f}")
        del clone
    
    print(f"Clearing moves found: {len(clearing_moves)}")
    if clearing_moves:
        _, lines, reward = clearing_moves[0]
        if lines > 0:
            print("Test PASS: Clear detected.")
            if reward > 40:
                print("Reward PASS.")
    else:
        print("WARNING: No clear. Adjust seed or check BFS/step.")

# Test seeds, focus on I-piece (p_type=0)
for seed in range(0, 100):  # More to find I
    print("\nTest seed:", seed)
    run_test(seed)