# ai/reward.py

import numpy as np
from numba import njit
from . import config

# ----------------------------------------------------------------------
# 优化点 1 & 2: 
# 使用 Numba 加速启发式计算，移除无用的变量，使用 fastmath。
# ----------------------------------------------------------------------
@njit(cache=True, fastmath=True, nogil=True)
def _fast_heuristics(board):
    rows, cols = board.shape
    # 使用 int32 避免溢出
    heights = np.zeros(cols, dtype=np.int32)
    
    holes = 0
    agg_height = 0
    
    # 单次遍历计算 Heights, Aggregate Height 和 Holes
    for c in range(cols):
        found = False
        for r in range(rows):
            if board[r, c] > 0:
                if not found:
                    h = rows - r
                    heights[c] = h
                    agg_height += h
                    found = True
            elif found:
                holes += 1
                
    # 独立的 Bumpiness 循环
    bumpiness = 0
    for c in range(cols - 1):
        bumpiness += abs(heights[c] - heights[c+1])
        
    max_height = 0
    for h in heights:
        if h > max_height: max_height = h
        
    # 直接返回标量 Tuple
    return max_height, holes, bumpiness, agg_height

def calculate_heuristics(board_state):
    """
    输入: board_state (numpy array, uint8)
    输出: tuple (max_height, holes, bumpiness, agg_height)
    """
    # 优化点 3: 移除不必要的 astype 拷贝，除非内存不连续
    if not board_state.flags['C_CONTIGUOUS']:
        board_state = np.ascontiguousarray(board_state)
    
    return _fast_heuristics(board_state)

def get_reward(step_result, current_stats, prev_stats, steps_survived, episode_stats=None, is_training=True):
    """
    参数:
      step_result: tuple (lines, damage, type, game_over, b2b, combo)
      current_stats: tuple (max_h, holes, bump, agg_h)
      prev_stats: tuple (max_h, holes, bump, agg_h)
    """
    reward = 0.0
    
    # ------------------------------------------------------------------
    # 修复点: 使用索引解包 tuple，对应 utils.py 中 step() 的返回值顺序
    # 0: lines, 1: damage, 2: type, 3: game_over, 4: b2b, 5: combo
    # ------------------------------------------------------------------
    lines = step_result[0]
    game_over = step_result[3]
    
    # Tuple 解包
    cur_max_h, cur_holes, cur_bump, cur_agg = current_stats
    prev_max_h, prev_holes, prev_bump, prev_agg = prev_stats
    
    # 1. 高度奖励/惩罚
    if cur_agg < prev_agg:
        reward += 0.02 * (prev_agg - cur_agg)
    
    # 2. 空洞惩罚 (加大惩罚力度)
    if cur_holes > prev_holes:
        reward -= 0.5 * (cur_holes - prev_holes)
    elif cur_holes < prev_holes:
        reward += 0.5 * (prev_holes - cur_holes)
    
    # 3. 崎岖度惩罚
    if cur_bump < prev_bump:
        reward += 0.05
    elif cur_bump > prev_bump:
        reward -= 0.05
    
    # 4. 存活奖励
    reward += 0.01
    
    # 5. 消行奖励 (查表优化)
    if lines > 0:
        if lines == 1: reward += 2.0
        elif lines == 2: reward += 5.0
        elif lines == 3: reward += 10.0
        elif lines == 4: reward += 20.0
    
    # 6. 游戏结束惩罚
    if game_over:
        reward -= 2.0
    
    # 7. 训练时的强制结束惩罚 (防止在高处无限苟活)
    force_game_over = False
    if is_training and cur_max_h >= 18:
        reward -= 5.0
        force_game_over = True
        
    reward = (reward + config.SCORE_OFFSET) / config.SCORE_SCALER
    return float(reward), force_game_over