# ai/reward.py (保持不变)
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
    # print(f"{board=}")
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
    reward = 0.0
    lines = step_result['lines_cleared']
    if lines > 0:
        print(f'\n\nLINESLINESLINESLINESLINES: {lines=}\n\n')
        base = [0,40,100,300,1200][lines]  # NES
        reward += base + 300 * lines
        reward += step_result.get('combo', 0) * 100
    reward += current_metrics['near_clears'] * 10  # 新: shaping近清奖
    reward -= current_metrics['holes'] * 0.05
    reward -= current_metrics['bumpiness'] * 0.01
    hole_delta = current_metrics['holes'] - prev_metrics.get('holes', 0)
    if hole_delta > 0: reward -= hole_delta * 1  # 弱罚
    else: reward += abs(hole_delta) * 30  # 强奖减holes
    mh = current_metrics['max_height']
    reward -= (mh / 20.0) * 0.5  # 弱
    if step_result['game_over']: reward -= 300
    force_game_over = False
    if is_training and (current_metrics['holes'] > 80 or mh > 19):  # 放宽
        force_game_over = True
        reward -= 30
    return reward, force_game_over
