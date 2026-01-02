# ai/test/test_greedy.py

import numpy as np
from ai.utils import TetrisGame
from ai.reward import calculate_heuristics

def test_greedy_performance(num_games=10, max_steps=500, seed=42):
    """
    Runs greedy simulations to evaluate baseline performance.
    - Tracks average lines, steps survived, holes, and max height.
    - If averages are low (e.g., <50 steps), greedy is ineffective.
    Returns aggregated stats.
    """
    np.random.seed(seed)
    stats = {"lines": [], "steps": [], "avg_holes": [], "avg_max_height": []}
    
    for _ in range(num_games):
        game = TetrisGame(seed=np.random.randint(0, 1000000))
        total_lines = 0
        total_holes = 0
        total_max_h = 0
        steps = 0
        
        while steps < max_steps:
            legal_moves = game.get_legal_moves()
            if len(legal_moves) == 0:
                break
            
            best_score = -np.inf
            best_move = None
            for move in legal_moves:
                with game.clone() as sim:
                    res = sim.step(move[0], move[1], move[2], move[4])
                    board, _ = sim.get_state()
                    metrics = calculate_heuristics(board.copy())
                    score = res[0] * 10 - metrics['holes'] * 5 - metrics['max_height'] * 2 - metrics['bumpiness'] * 1
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            if best_move.any():
                res = game.step(best_move[0], best_move[1], best_move[2], best_move[4])
                total_lines += res[0]
                board, _ = game.get_state()
                metrics = calculate_heuristics(board.copy())
                total_holes += metrics['holes']
                total_max_h += metrics['max_height']
                steps += 1
            else:
                break
        
        stats["lines"].append(total_lines)
        stats["steps"].append(steps)
        stats["avg_holes"].append(total_holes / steps if steps > 0 else 0)
        stats["avg_max_height"].append(total_max_h / steps if steps > 0 else 0)
    
    agg_stats = {k: np.mean(v) for k, v in stats.items()}
    interpretation = ""
    if agg_stats["steps"] < 50:
        interpretation += "Issue: Short survival—greedy creates too many holes.\n"
    if agg_stats["lines"] < 10:
        interpretation += "Issue: Few lines cleared—reward heuristics misaligned.\n"
    
    agg_stats["interpretation"] = interpretation or "Greedy performs adequately."
    return agg_stats

# Usage: results = test_greedy_performance(); print(results)

if __name__ == "__main__":
    results = test_greedy_performance()
    print(results)