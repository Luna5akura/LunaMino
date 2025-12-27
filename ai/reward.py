# ai/reward.py
import numpy as np

def calculate_heuristics(board_state):
    """
    计算启发式特征
    假设 board_state 是 (20, 10) 的 0/1 矩阵
    根据 C 代码逻辑：y=0 是底部，y=19 是顶部
    """
    # 确保是 int 类型
    board = (board_state > 0).astype(int)
    
    heights = []
    holes = 0
    
    # 1. 计算每一列的高度 和 空洞
    for x in range(10):
        # 这一列的数据
        col = board[:, x]
        
        # 找高度: 从顶部(19)向下找第一个1
        h = 0
        for y in range(19, -1, -1):
            if col[y] == 1:
                h = y + 1
                break
        heights.append(h)
        
        # 找空洞: 在当前高度以下，如果有0，就是空洞
        # 比如高度是5，那么索引 0~4 应该是实心的。如果 board[y][x] == 0 且 y < h-1，则是空洞
        if h > 0:
            # 取出地面到最高点之间的部分
            col_valid = col[:h]
            # 统计里面的 0 的个数
            n_zeros = np.sum(col_valid == 0)
            holes += n_zeros

    heights = np.array(heights)
    max_height = np.max(heights)
    aggregate_height = np.sum(heights)
    
    # 2. 不平整度 (相邻列高度差)
    bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
    
    return {
        'max_height': max_height,
        'holes': holes,
        'bumpiness': bumpiness,
        'agg_height': aggregate_height
    }

def get_reward(step_result, board_state, current_steps, is_training=True):
    """
    统一的奖励函数
    
    :param step_result: game.step() 返回的字典
    :param board_state: game.get_state() 返回的 board (20, 10)
    :param current_steps: 当前局已走的步数
    :param is_training: 如果为 True，会启用启发式强制结束(force_over)以加速训练
                        如果为 False，则只计算分数，不强制结束游戏
                        默认值为 True，保持向后兼容
    :return: (reward, should_force_game_over)
    """
    # 处理向后兼容：如果只传三个参数，则默认为训练模式
    if len(locals()) <= 3:  # step_result, board_state, current_steps
        is_training = True
    
    metrics = calculate_heuristics(board_state)
    
    reward = 0.0
    force_game_over = False

    # --- 1. 基础生存与消除奖励 ---
    if step_result['lines_cleared'] > 0:
        # 消除行是最高优先级
        reward += step_result['lines_cleared'] * 1.0
        # 鼓励多消 (Tetris)
        if step_result['lines_cleared'] >= 4:
            reward += 2.0
    else:
        # 没消行时，给极小的生存分
        reward += 0.01

    # --- 2. 攻击奖励 (可选) ---
    if step_result['damage_sent'] > 0:
        reward += step_result['damage_sent'] * 0.5

    # --- 3. 启发式惩罚 ---
    # 惩罚空洞：空洞是万恶之源
    reward -= metrics['holes'] * 0.15
    
    # 惩罚不平整：鼓励地形平整
    reward -= metrics['bumpiness'] * 0.02
    
    # 惩罚过高：防止突然死亡
    if metrics['max_height'] > 12:
        reward -= (metrics['max_height'] - 12) * 0.1
    if metrics['max_height'] > 17:
        reward -= 1.0 # 极度危险

    # --- 4. 游戏结束判定 ---
    if step_result['game_over']:
        reward -= 2.0 # 死亡重罚
    
    # --- 5. 强制结束逻辑 ---
    if is_training:
        # 训练时：如果玩得太烂（空洞太多），直接重开，不要浪费时间
        if metrics['holes'] > 15:
            force_game_over = True
            reward -= 1.0
    else:
        # 评估/观看时：即使玩得烂，也让它挣扎到死，不人为干预
        pass

    return reward, force_game_over

# 兼容旧代码的包装函数
def get_reward_compat(step_result, board_state, current_steps):
    """向后兼容的包装函数，用于旧代码调用"""
    return get_reward(step_result, board_state, current_steps, is_training=True)