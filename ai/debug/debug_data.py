# ai/debug_data.py
import pickle
import numpy as np
from ai import config

def inspect_warmup():
    print(f"Loading {config.MEMORY_FILE}...")
    try:
        with open(config.MEMORY_FILE, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error: Memory file not found!")
        return

    boards = data['boards']
    values = data['values'] # Value 应该对应分数
    probs = data['probs']
    size = data['size']
    
    print(f"Total Samples: {size}")
    
    # 1. 检查 Value 分布
    # 如果大部分 Value 都是 -1 (或极低)，说明贪心算法本身就没玩好，或者 Reward 计算有误
    avg_val = np.mean(values[:size])
    max_val = np.max(values[:size])
    min_val = np.min(values[:size])
    
    print(f"Value Stats -> Avg: {avg_val:.4f}, Max: {max_val:.4f}, Min: {min_val:.4f}")
    
    if max_val < 0:
        print("[CRITICAL] Warmup data has NO positive outcomes. The Greedy algorithm failed.")
        return

    # 2. 检查 Probs (Policy Target)
    # 看看是否真的有确定的动作（one-hot），还是全是0
    sample_idx = np.random.randint(0, size)
    sample_prob = probs[sample_idx]
    max_p = np.max(sample_prob)
    
    print(f"Sample {sample_idx} Max Prob: {max_p}")
    if max_p < 0.9:
        print("[WARNING] Warmup targets are not One-Hot/Greedy? (Expected 1.0 for best move)")

    # 3. 检查 Board 是否全是 0
    sample_board = boards[sample_idx]
    if np.sum(sample_board) == 0:
        print("[WARNING] Sample board is completely empty. Is data collection working?")
    else:
        print("[OK] Board data looks non-empty.")

if __name__ == "__main__":
    inspect_warmup()