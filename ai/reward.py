# ai/reward.py

import numpy as np

def calculate_heuristics(board_state):
    """
    计算启发式特征
    board_state: (20, 10) 0/1 矩阵, 0是顶, 19是底
    """
    board = (board_state > 0).astype(int)
    
    # 获取每一列的最高点高度
    # argmax 如果全是0会返回0，所以需要特殊处理
    # 我们构造一个辅助判断：如果一列全是0，top_y 应该是 20 (height=0)
    # 这里用一种向量化写法加速
    mask = board != 0
    # 找到每列第一个非零元素的索引
    # argmax 在 bool 数组上会返回第一个 True 的索引
    first_non_zero = np.argmax(mask, axis=0)
    
    # 检查该列是否真的有方块（全0列 argmax 也是0）
    has_blocks = mask[first_non_zero, np.arange(10)]
    
    # y=0是顶，y=19是底。高度 = 20 - y
    heights = np.where(has_blocks, 20 - first_non_zero, 0)
    
    max_height = np.max(heights)
    agg_height = np.sum(heights)
    
    # 计算空洞
    # 定义：任何在“方块下方”的空白格都是空洞
    # 我们可以通过累积和来掩盖掉“表面”
    # 这是一个比较快速的技巧：
    # 对每一列，从顶到底，一旦遇到1，后续所有的0都是空洞
    
    holes = 0
    for x in range(10):
        if heights[x] == 0:
            continue
        # 取出这一列
        col = board[:, x]
        # 找到最高点位置
        top = 20 - heights[x]
        # 统计 top 下方 0 的数量
        holes += np.sum(col[top:] == 0)

    # 不平整度 (相邻列高度差绝对值之和)
    bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
    
    return {
        'max_height': max_height,
        'holes': holes,
        'bumpiness': bumpiness,
        'agg_height': agg_height,
        'heights': heights # 供调试用
    }
def get_reward(step_result, current_metrics, prev_metrics, is_training=True):
    reward = 0.0
    force_game_over = False
    
    # ==========================================================
    # 1. 生存奖励 (Living Wage)
    # ==========================================================
    # 提高一点生存奖励，确保即使走了一步烂棋，只要没死，总分尽量非负
    reward += 2.0 

    # ==========================================================
    # 2. 消除奖励 (Incentive)
    # ==========================================================
    lines = step_result['lines_cleared']
    if lines > 0:
        print(f'\n {lines=} \n')
        # 即使只消1行，也给它巨大的甜头，引导它去消行
        if lines == 1: reward += 20.0  # 原来是10
        elif lines == 2: reward += 50.0
        elif lines == 3: reward += 100.0
        elif lines == 4: reward += 200.0 # Tetris 是终极目标
        
        if step_result.get('combo', 0) > 0:
            reward += step_result['combo'] * 10.0

    # ==========================================================
    # 3. 惩罚 (Penalties) - 更加温和
    # ==========================================================
    
    # A. 静态空洞惩罚 (Static Holes)
    # 只要有空洞就扣分，但扣得很少，作为长期压力
    reward -= current_metrics['holes'] * 0.1 

    # B. 动态空洞惩罚 (Delta Holes) - 关键修改点
    # 原来是 * 5.0，太重了。现在改为 * 2.0
    # 并且，如果这一步既消行了又产生了空洞，稍微宽容一点
    cur_holes = current_metrics['holes']
    prev_holes = prev_metrics['holes'] if prev_metrics else 0
    hole_change = cur_holes - prev_holes
    
    if hole_change > 0:
        # 制造了新空洞
        penalty = hole_change * 2.0 
        reward -= penalty
    elif hole_change < 0:
        # 填补了空洞，给奖励！这是让它学会修补的关键
        print('fill hole')
        reward += abs(hole_change) * 5.0  # 填坑奖励要比挖坑惩罚大

    # C. 高度惩罚
    # 鼓励在这个阶段把方块放低
    # 增加一个“落点高度”的微小惩罚，让它倾向于放在低处
    # (注意：utils.py 里需要确保 landing_height 正确，如果暂不可用，用 max_height 代替)
    mh = current_metrics['max_height']
    if mh > 10: reward -= 0.5
    if mh > 16: reward -= 3.0 # 危险区重罚

    # ==========================================================
    # 4. 游戏结束
    # ==========================================================
    if step_result['game_over']:
        reward -= 50.0 

    # 5. 训练强制结束 (烂局重开)
    if is_training:
        # 稍微放宽一点限制，别让它刚开始学就一直重开
        if cur_holes > 20 or mh > 19:
            force_game_over = True
            reward -= 20.0

    return reward, force_game_over