# ai/reward.py
import numpy as np
def calculate_heuristics(board_state):
    board = (board_state > 0).astype(int)
    mask = board != 0
    first_non_zero = np.argmax(mask, axis=0)
    has_blocks = mask[first_non_zero, np.arange(10)]
    heights = np.where(has_blocks, 20 - first_non_zero, 0)
    max_height = np.max(heights)
    agg_height = np.sum(heights)
    holes = 0
    for x in range(10):
        if heights[x] == 0: continue
        top = 20 - heights[x]
        holes += np.sum(board[top:, x] == 0)
    bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
    near_clears = sum(1 for row in board if np.sum(row) >= 8)
    return {
        'max_height': max_height, 'holes': holes, 'bumpiness': bumpiness,
        'agg_height': agg_height, 'heights': heights, 'near_clears': near_clears
    }
def get_reward(step_result, current_metrics, prev_metrics, steps_survived, is_training=True):
    reward = 0.0
    lines = step_result['lines_cleared']
    attack_type = step_result['attack_type']
    # --- 1. 基础消行奖励 (弱化，焦点在Special) ---
    if lines > 0:
        base_rewards = [0, 40, 120, 250, 700] # Tetris高奖但整体下调
        reward += base_rewards[lines]
        reward += step_result.get('combo', 0) * 20 # Combo低权重
    # --- 2. Special 清行奖励 (重奖Tetris/T-Spin) ---
    type_rewards = {
        4: 300, # Tetris extra
        8: 800, # TSD 高奖
        9: 1200, # TST 最高奖
        5: 40, 6: 150, 7: 350 # 其他T-Spin
    }
    reward += type_rewards.get(attack_type, 0)
    # --- 4. 空洞变化 (生存核心) ---
    hole_delta = current_metrics['holes'] - prev_metrics.get('holes', 0)
    if hole_delta < 0:
        reward += abs(hole_delta) * 120.0 # 重奖清理
    elif hole_delta > 0:
        reward -= hole_delta * 30.0 # 重罚新洞
    # --- 5. 静态惩罚 (轻柔但持续) ---
    reward -= current_metrics['holes'] * 0.3
    reward -= current_metrics['bumpiness'] * 0.08
    reward -= current_metrics['max_height'] * 0.4
    # --- 6. 生存 bonus (累积鼓励长存) ---
    reward += steps_survived * 0.1
    reward += current_metrics['near_clears'] * 20 # 激励近满行清理
    # --- 7. 游戏结束 ---
    if step_result['game_over']:
        print('\nWow Game Over')
        reward -= 150 # 中等罚，允许学习从高堆恢复
    # --- 8. 强制结束 (阈值18，罚高堆) ---
    force_game_over = False
    if is_training and current_metrics['max_height'] > 18:
        force_game_over = True
        reward -= 50
    return reward, force_game_over