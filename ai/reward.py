# ai/get_reward.py


import numpy as np
from numba import njit
from . import config

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
    
    # -------------------------------------------------------------------------
    # [新增逻辑] 计数器初始化与更新
    # -------------------------------------------------------------------------
    if episode_stats is not None:
        # 1. 连续空洞增加计数
        cur_holes = current_metrics['holes']
        prev_holes_val = prev_metrics.get('holes', 0)
        
        if cur_holes > prev_holes_val:
            episode_stats['consecutive_hole_inc'] = episode_stats.get('consecutive_hole_inc', 0) + 1
        else:
            # 只要有一步没增加，就重置（如果想改为"净增加"逻辑可以调整这里，但"连续"通常指中断即重置）
            episode_stats['consecutive_hole_inc'] = 0
            
        # 2. 连续最高高度增加计数
        cur_max_h = current_metrics['max_height']
        prev_max_h_val = prev_metrics.get('max_height', 0)
        
        if cur_max_h > prev_max_h_val:
            episode_stats['consecutive_height_inc'] = episode_stats.get('consecutive_height_inc', 0) + 1
        else:
            episode_stats['consecutive_height_inc'] = 0
    # -------------------------------------------------------------------------

    # 1. 消除奖励
    if lines > 0:
        if lines == 1: reward += 100       
        elif lines == 2: reward += 300
        elif lines == 3: reward += 600
        elif lines == 4: reward += 1200    
        
        combo = step_result.get('combo', 0)
        if combo > 0: reward += combo * 5

    # 2. 状态惩罚
    # 只有极其微小的存活奖励，防止它为了刷分而故意不消除
    reward += 0.01 

    # 极其温和的空洞惩罚
    reward += 0.05 
    
    # 基础空洞扣分
    cur_holes = current_metrics['holes']
    prev_holes = prev_metrics.get('holes', 0)
    reward -= 0.1 * cur_holes 

    # 动态空洞惩罚/奖励
    if cur_holes > prev_holes:
        reward -= (cur_holes - prev_holes) * 30 
    else:
        reward += (prev_holes - cur_holes) * 200

    # 高度惩罚
    max_h = current_metrics['max_height']
    if max_h > 10:
        reward -= (max_h - 10) * 0.1
    
    # 崎岖度惩罚
    reward -= current_metrics['bumpiness'] * 0.01

    # 3. 游戏结束
    if game_over:
        reward -= 50
    
    # 4. 训练截断 (Force Game Over)
    force_game_over = False
    
    if is_training:
        # [修改点] 检查连续恶化条件
        # 注意：这里给了较重的惩罚 (-5.0)，因为这是非常糟糕的行为
        consecutive_holes = episode_stats.get('consecutive_hole_inc', 0)
        consecutive_height = episode_stats.get('consecutive_height_inc', 0)
        
        if consecutive_holes >= 3:
            reward -= 50
            force_game_over = True
            # print(f"Force Over: Consecutive Holes ({consecutive_holes})")
            
        elif consecutive_height >= 3:
            reward -= 50
            force_game_over = True
            # print(f"Force Over: Consecutive Height ({consecutive_height})")
            
        # 原有的截断逻辑 (防止死循环或无效探索)
        elif max_h > 18 or cur_holes > 10: 
            reward -= 20
            force_game_over = True
            

    reward = (reward + config.SCORE_OFFSET)/config.SCORE_SCALER
    return float(reward), bool(force_game_over)