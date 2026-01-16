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
    
    # --- 基础统计 ---
    max_height = 0
    holes = 0
    cumulative_height = 0
    bumpiness = 0
    
    # 记录每一列的高度
    col_heights = np.zeros(cols, dtype=np.int32)
    
    for c in range(cols):
        found = False
        for r in range(rows):
            if board[r, c] > 0:
                if not found:
                    h = rows - r
                    col_heights[c] = h
                    if h > max_height: max_height = h
                    found = True
                
                # 这一格有方块，累计高度
                cumulative_height += 1 
            else:
                # 这一格没方块
                if found:
                    # 如果上面已经发现过方块，那么这里就是空洞
                    holes += 1
    
    # 计算平整度 (Bumpiness)
    for c in range(cols - 1):
        diff = col_heights[c] - col_heights[c+1]
        if diff < 0: diff = -diff
        bumpiness += diff

    # --- [关键修改] 归一化评分计算 ---
    # 我们不要用 if-else，而是用线性加权
    # 目标：让分数大概在 -1.0 到 1.0 之间
    
    # 权重设定 (经典经验值简化版)
    # 1. 高度越高，扣分越多 (权重: -0.5)
    # 2. 空洞越多，扣分越多 (权重: -4.0) -> 空洞依然很坏，但不是直接判死刑
    # 3. 表面越不平整，扣分越多 (权重: -1.0)
    
    # 归一化因子 (Normalize)
    # 假设最大高度20，最大空洞数可能20，最大bumpiness可能20
    
    score = 0.0
    
    # 1. 高度惩罚 (Aggregate Height)
    # 鼓励低高度。即使有空洞，只要高度低，也比堆到顶要好。
    # 这里的 cumulative_height 是所有列高度之和
    score -= (cumulative_height / 200.0) * 1.0
    
    # 2. 空洞惩罚 (Holes)
    # 空洞是坏的，但每个空洞扣 0.1 分，而不是直接变成负数
    score -= (holes * 0.15)
    
    # 3. 平整度惩罚 (Bumpiness)
    # 鼓励表面平整
    score -= (bumpiness * 0.05)
    
    # 4. 最大高度惩罚 (防止单列冲顶)
    if max_height > 15:
        score -= 0.5 # 危险高度额外扣分
    
    # --- 调整到 tanh 舒适区 [-1, 1] ---
    # 基础分给 0.5，做得好加分，做得差扣分
    final_score = 0.5 + score
    
    # 截断，防止溢出
    if final_score < -1.0: final_score = -1.0
    if final_score > 1.0: final_score = 1.0
    
    return max_height, holes, bumpiness, cumulative_height, final_score

# 必须修改 calculate_heuristics 以适应新的返回值
def calculate_heuristics(board_state):
    if not board_state.flags['C_CONTIGUOUS']:
        board_state = np.ascontiguousarray(board_state)
    
    # 注意：现在返回 5 个值，最后一个是 score
    return _fast_heuristics(board_state)

def get_reward(step_result, current_stats, prev_stats, steps_survived, episode_stats=None, is_training=True):
    # 解包 tuple
    lines = step_result[0]
    game_over = step_result[3]
  
    # 当前虽然不怎么用这些惩罚项，但先解包出来备用
    cur_max_h, cur_holes, cur_bump, cur_agg = current_stats
    prev_max_h, prev_holes, prev_bump, prev_agg = prev_stats
  
    reward = 0.0
    
    # -----------------------------------------------------------
    # 1. 存活奖励 (大幅提升，先让它学会不死)
    # -----------------------------------------------------------
    reward += 1.0 
  
    # -----------------------------------------------------------
    # 2. 消行奖励 (简单线性奖励，诱导它去消行)
    # -----------------------------------------------------------
    if lines > 0:
        # 1行给50，4行给200，这比活着(1分)爽多了
        reward += lines * 50.0 

    # -----------------------------------------------------------
    # 3. 状态惩罚
    # -----------------------------------------------------------
    # 建议：初期不要给太重的空洞惩罚，甚至可以先去掉。
    # 让 MCTS 的启发式逻辑去管空洞，Reward 函数只管"活着"和"消行"。
    # 如果一定要留，必须确保它远小于生存奖励。
    
    # 修改建议：暂时注释掉空洞惩罚，或者减小权重

    # if cur_holes - prev_holes < 0:
    #    reward -= (cur_holes - prev_holes) * 10 
 
    # if cur_holes - prev_holes > 0:
    #    reward -= (cur_holes - prev_holes) * 0.5  # 稍微罚一点就行
    
    # -----------------------------------------------------------
    # 4. 死亡惩罚 (必须放开！)
    # -----------------------------------------------------------
    if game_over:
        # 必须给负分！而且要大！
        reward -= 100.0
    # -----------------------------------------------------------
    # 5. 归一化
    # -----------------------------------------------------------
    # 简单的缩放，让数值不要太大
    reward = reward / config.SCORE_SCALER
   
    return reward, False