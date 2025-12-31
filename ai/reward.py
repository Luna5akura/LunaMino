# ai/reward.py

import numpy as np

def calculate_heuristics(board_state):
    board = (board_state > 0).astype(int)
    mask = board != 0
    # 找每列第一个非零行（top-most filled cell index in original orientation）
    first_non_zero = np.argmax(mask, axis=0)
    has_blocks = mask[first_non_zero, np.arange(10)]
    heights = np.where(has_blocks, 20 - first_non_zero, 0)
    max_height = np.max(heights)
    agg_height = np.sum(heights)

    holes = 0
    for x in range(10):
        if heights[x] == 0:
            continue
        top = 20 - heights[x]
        holes += np.sum(board[top:, x] == 0)

    bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
    near_clears = int(sum(1 for row in board if np.sum(row) >= 8))

    return {
        'max_height': int(max_height),
        'holes': int(holes),
        'bumpiness': float(bumpiness),
        'agg_height': int(agg_height),
        'heights': heights,
        'near_clears': near_clears
    }

def get_reward(step_result, current_metrics, prev_metrics, steps_survived,
               context=None, prev_context=None, episode_stats=None, is_training=True):
    """
    返回 (reward, force_game_over)
    - context / prev_context: numpy array from game.get_state() (optional) — 我们用它读取 pending_garbage & b2b/combo 信息
    - episode_stats: optional dict (mutable) will be updated to accumulate digging stats:
        keys used: 'pieces_since_last_garbage', 'total_garbage_cleared', 'garbage_clear_events', 'pieces_used_for_garbage'
    """

    reward = 0.0
    lines = int(step_result.get('lines_cleared', 0))
    attack_type = int(step_result.get('attack_type', 0))
    combo = int(step_result.get('combo', 0))

    # ---------- 1) 行清奖励（放大并使用平方以鼓励大清） ----------
    # 基础缩放因子（可调）
    BASE_LINE_SCALE = 80.0
    # squared reward: 1->1,2->4,3->9,4->16 之后乘以 BASE_LINE_SCALE
    if lines > 0:
        reward += BASE_LINE_SCALE * (lines ** 2)
        # combo bonus (鼓励连续连击)
        reward += 40.0 * (combo ** 1.2)

    # ---------- 2) 特殊类型奖励 (T-Spin/Tetris/B2B) ----------
    # 这里保留已有的 mapping，但整体规模更大（与 BASE_LINE_SCALE 同阶）
    type_rewards = {4: 12.0, 8: 24.0, 9: 36.0, 5: 4.0, 6: 12.0, 7: 18.0}
    # attack_type 值在 C side 定义，乘以 BASE_LINE_SCALE 做最终贡献
    if attack_type in type_rewards:
        reward += BASE_LINE_SCALE * type_rewards[attack_type] / 8.0

    # B2B 增益：如果前一帧 context 表示处于 B2B（假设 prev_context 有 b2b at index 0）
    try:
        if prev_context is not None:
            prev_b2b = int(prev_context[0])
        else:
            prev_b2b = 0
    except Exception:
        prev_b2b = 0

    if prev_b2b > 0 and lines > 0:
        # 如果在 B2B 且能打出清行，额外加成
        reward += 0.5 * BASE_LINE_SCALE * lines

    # ---------- 3) 挖掘（Garbage digging）效率奖励 ----------
    # 我们假设 context 的最后一项是 pending_garbage （在 utils.get_state 中按你的实现应是第11项）
    prev_pending = 0
    if prev_context is not None and len(prev_context) >= 11:
        prev_pending = int(prev_context[-1])
    cur_pending = 0
    if context is not None and len(context) >= 11:
        cur_pending = int(context[-1])

    # episode_stats 用于统计 pieces-per-dig 的效率
    if episode_stats is not None:
        episode_stats.setdefault('pieces_since_last_garbage', 0)
        episode_stats.setdefault('total_garbage_cleared', 0)
        episode_stats.setdefault('garbage_clear_events', 0)
        episode_stats.setdefault('pieces_used_for_garbage', 0)
        # 每放一块时，调用方应在调用 get_reward 之前或之后把 pieces_since_last_garbage += 1
        # 这里我们只在清行时使用它

    # 如果上一步存在 pending 且本步清行 -> 视为挖掘成功
    if prev_pending > 0 and lines > 0:
        garbage_cleared = min(lines, prev_pending)  # 保守估计
        # 记录统计数据（caller 需确保传入 episode_stats）
        if episode_stats is not None:
            pieces = max(1, episode_stats.get('pieces_since_last_garbage', 1))
            episode_stats['total_garbage_cleared'] += garbage_cleared
            episode_stats['garbage_clear_events'] += 1
            episode_stats['pieces_used_for_garbage'] += pieces
            # reset counter after a successful clear
            episode_stats['pieces_since_last_garbage'] = 0

            # 奖励：高效挖掘（小于等于3块/次视为高效）
            if pieces <= 3:
                # 极高奖励鼓励 1-3 块就清垃圾
                reward += 200.0 * (4 - pieces)
            else:
                # 仍然奖励，但递减
                reward += 40.0 * max(0, 3.0 / pieces)

    # ---------- 4) 洞 & bumpiness & height 的 shaping（弱化惩罚以免过度保守） ----------
    hole_delta = int(current_metrics.get('holes', 0)) - int(prev_metrics.get('holes', 0))
    if hole_delta < 0:
        # 清洞奖励（鼓励挖洞和清理）
        reward += 60.0 * abs(hole_delta)
    elif hole_delta > 0:
        # 新洞惩罚（减少，但不能太大）
        reward -= 20.0 * hole_delta

    # 静态惩罚，系数较温和
    reward -= 1.0 * current_metrics.get('holes', 0)    # 每个洞 -1
    reward -= 0.3 * current_metrics.get('bumpiness', 0) # bumpiness 轻罚
    reward -= 0.8 * current_metrics.get('max_height', 0) # 高度惩罚，但不过分

    # near clears 奖励（鼓励搭建）——保持适中
    reward += 30.0 * current_metrics.get('near_clears', 0)

    # 生存奖励（每步小正奖励，鼓励长存活）
    reward += 0.6

    # ---------- 5) 游戏结束与强制结束 ----------
    force_game_over = False
    if step_result.get('game_over', False):
        # 缩小 game over 惩罚（-100）避免过强惩罚使策略收敛坏方向
        reward -= 100.0
    # 强制结束条件（训练用，允许恢复）：当高度极高时短暂惩罚并设 force_over True
    if is_training and current_metrics.get('max_height', 0) > 19:
        force_game_over = True
        reward -= 40.0  # 轻罚，鼓励回退但不毁训练信号

    # 返回 reward 和是否要强制结束
    return float(reward), bool(force_game_over)