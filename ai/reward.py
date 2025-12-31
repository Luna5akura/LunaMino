# ai/get_reward.py

import numpy as np
from numba import njit

@njit(cache=True, fastmath=True, nogil=True)
def _fast_heuristics(board_state):
    rows, cols = board_state.shape
    heights = np.zeros(cols, dtype=np.int32)
    row_blocks = np.zeros(rows, dtype=np.int32)
    holes = 0
    
    for c in range(cols):
        found_top = False
        for r in range(rows):
            val = board_state[r, c]
            if val > 0:
                row_blocks[r] += 1
                if not found_top:
                    heights[c] = rows - r
                    found_top = True
            elif found_top:
                holes += 1
                
    max_height = 0
    agg_height = 0
    for h in heights:
        if h > max_height: max_height = h
        agg_height += h
        
    bumpiness = 0
    for c in range(cols - 1):
        diff = heights[c] - heights[c+1]
        if diff < 0: diff = -diff
        bumpiness += diff
        
    near_clears = 0
    for r in range(rows):
        if row_blocks[r] >= 8: near_clears += 1
        
    return max_height, holes, bumpiness, agg_height, heights, near_clears

def calculate_heuristics(board_state):
    # 保持原样，处理 contiguous
    if not board_state.flags['C_CONTIGUOUS']:
        board = np.ascontiguousarray(board_state, dtype=np.int32)
    else:
        if board_state.dtype != np.int32:
            board = board_state.astype(np.int32)
        else:
            board = board_state
    
    max_h, holes, bump, agg_h, heights, near_c = _fast_heuristics(board)
    return {
        'max_height': int(max_h),
        'holes': int(holes),
        'bumpiness': float(bump),
        'agg_height': int(agg_h),
        'heights': heights,
        'near_clears': int(near_c)
    }

# ==========================================
# 优化后的 Reward 函数
# ==========================================
def get_reward(step_result, current_metrics, prev_metrics, steps_survived,
               context=None, prev_context=None, episode_stats=None, is_training=True):
    reward = 0.0
    
    lines = int(step_result.get('lines_cleared', 0))
    game_over = step_result.get('game_over', False)
    
    # 1. 消除奖励 (保持，甚至可以适当提高比重)
    if lines > 0:
        if lines == 1: reward += 10       # 降低绝对值，让数值范围可控
        elif lines == 2: reward += 30
        elif lines == 3: reward += 60
        elif lines == 4: reward += 120    # 鼓励 Tetris
        
        combo = step_result.get('combo', 0)
        if combo > 0: reward += combo * 5

    # 2. 状态惩罚 (关键修改！！！)
    cur_holes = current_metrics['holes']
    prev_holes = prev_metrics.get('holes', 0)
    max_h = current_metrics['max_height']
    
    # [修改点 1] 只有极其微小的存活奖励，防止它为了刷分而故意不消除
    reward += 0.01 

    # [修改点 2] 极其温和的空洞惩罚
    # 现在的 AI 还没学会走路，不要打断它的腿
    # 只要有空洞，每步微扣
    reward += 0.05 
    
    reward -= 0.1 * cur_holes  # Or fixed: if cur_holes > 0: reward -= 0.05
    # 新增空洞惩罚：从 300 降到 2.0
    # 只有当它真的产生新空洞时才扣分，但不要扣死
    if cur_holes > prev_holes:
        reward -= (cur_holes - prev_holes) * 30 
    
    # 填洞奖励：鼓励它填坑
    elif cur_holes < prev_holes:
        reward += (prev_holes - cur_holes) * 20

    # 高度惩罚
    if max_h > 10:
        reward -= (max_h - 10) * 0.1
    
    # 崎岖度惩罚
    reward -= current_metrics['bumpiness'] * 0.01

    # 3. 游戏结束
    if game_over:
        reward -= 5.0 # 只要比消除 Tetris (12.0) 小，它就会为了消除而冒险，但比一步乱走大
    
    # 4. 训练截断 (Force Game Over)
    # 如果你也想训练它不自杀，这里不要扣太多，或者不算入 value
    if is_training:
        if max_h > 18 or cur_holes > 20: # 放宽一点条件
            reward -= 2.0
            force_game_over = True
        else:
            force_game_over = False
    else:
        force_game_over = False
            
    # 现在的 reward 范围大概在 [-5, +15] 之间，而不是 [-6000, +1000]
    return float(reward), bool(force_game_over)