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
    
    # 1. 基础消行奖励（可以适当调低，让位给消洞）
    lines = step_result['lines_cleared']
    line_rewards = [0, 10, 30, 80, 200] # 调低基础分，让AI不只是为了消行而消行
    reward += line_rewards[lines]
    
    # 2. 空洞惩罚（核心）
    curr_holes = current_metrics['holes']
    prev_holes = prev_metrics.get('holes', 0)
    
    # 静态惩罚：只要有洞，每一步都扣分（促使AI尽快消洞）
    reward -= curr_holes * 2.0 
    
    # 动态奖励：消灭一个洞给高分
    hole_delta = curr_holes - prev_holes
    if hole_delta < 0:
        # 消洞了！给予重奖
        reward += abs(hole_delta) * 50.0 
    elif hole_delta > 0:
        # 造洞了！严厉处罚
        reward -= hole_delta * 80.0
        
    # 3. 覆盖空洞惩罚 (防止在空洞上方堆叠)
    # 如果空洞上方的格子被填满了，惩罚增加
    reward -= current_metrics['agg_height'] * 0.1
    
    # 4. 死亡惩罚
    if step_result['game_over']:
        reward -= 500
        
    return reward, False