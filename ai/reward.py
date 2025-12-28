
# ai/reward.py (加强清行 bonus + 放宽 force_over)
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
    near_clears = sum(1 for row in board if np.sum(row) >= 8) # ≥8/10
    return {
        'max_height': max_height, 'holes': holes, 'bumpiness': bumpiness,
        'agg_height': agg_height, 'heights': heights, 'near_clears': near_clears
    }
def get_reward(step_result, current_metrics, prev_metrics, is_training=True):
    reward = 1.5 # ↓1.5 (减生存，逼探索)
    lines = step_result['lines_cleared']
    if lines > 0:

        print(f'\n {lines=} \n') # 保持

        if lines == 1: reward += 40 # ↑x2
        elif lines == 2: reward += 80
        elif lines == 3: reward += 150
        elif lines == 4: reward += 300
        reward += 100 * lines  # 新: 清行额外 bonus，放大信号
        if step_result.get('combo', 0) > 0: reward += step_result['combo'] * 15
    # reward += current_metrics['near_clears'] * 2.0 # +2/near (强推设局)
    reward -= current_metrics['holes'] * 0.2 # ↑x2
    reward -= current_metrics['bumpiness'] * 0.05 # 新: 不平罚
    cur_holes, prev_holes = current_metrics['holes'], prev_metrics.get('holes', 0)
    hole_delta = cur_holes - prev_holes
    if hole_delta > 0: reward -= hole_delta * 1.5 # ↓温和
    else: reward += abs(hole_delta) * 6.0
    mh = current_metrics['max_height']
    reward -= (mh / 20.0)**2 * 2.0 # 平滑高度罚 (取代阈值)
    if step_result['game_over']: reward -= 40 # ↓温和
    force_game_over = False
    if is_training and (current_metrics['holes'] > 35 or mh > 18): # 放宽 holes>35
        force_game_over = True
        reward -= 15
    return reward, force_game_over  # 恢复 force_over

