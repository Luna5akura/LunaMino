# ai/config.py
import numpy as np

# Centralized hyperparameters and configuration
# Environment settings
MAX_LEGAL_MOVES = 256
BOARD_HEIGHT = 20
BOARD_WIDTH = 10
CTX_DIM = 11
ACTION_DIM = 4 # [x, y, rotation, use_hold]

# Normalization ranges (assumed based on interface; adjust if exact ranges differ)
X_MIN = -2
X_MAX = 10
Y_MIN = 0 
Y_MAX = 23 
ROT_MAX = 3
HOLD_MAX = 1

# Model architecture
CNN_CHANNELS = [32, 64]
HIDDEN_DIM = 128
EMBED_DIM = 64

# Training hyperparameters
GAMMA = 0.99 # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9996
BUFFER_SIZE = 100000
BATCH_SIZE = 256        # 从 64 提升到 256。给 GPU 充足的矩阵让它去乘！
LR = 0.001
TARGET_UPDATE_FREQ = 2500  # 因为 Batch 变大了，目标网络的更新频率也适当拉长
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 50000

# Multi-threading configuration
NUM_THREADS = 8  # Number of threads for parallel episode execution; adjust as needed

# =========================================================================
# Reward shaping coefficients & Logic (强化学习奖励调整区)
# =========================================================================

REWARD_HOLES = -10.0       # 从-30回调到-10。依然讨厌空洞，但不至于吓死
REWARD_BUMPINESS = -0.2    # 【关键】从-2.0降到-0.2。极大地宽容表面崎岖，允许它挖“深井”等长条！
REWARD_HEIGHT = -0.1       # 略微惩罚高度，防止它无意义地往上堆
REWARD_LANDING = 0.0       
REWARD_GAME_OVER = -100.0  # 从-500降到-100。防止巨额负分导致 Q网络崩溃（Loss爆炸）

def compute_reward(step_res, prev_board, post_board, landing_height):
    lines_cleared, _, _, is_game_over, _, _ = step_res
    
    # 1. 基础得分：消4行是终极目标
    line_rewards = {0: 0, 1: 0, 2: 30, 3: 200, 4: 500}
    game_reward = line_rewards.get(lines_cleared, 0)
    
    # 如果死亡，直接给一个固定惩罚并结束（不计算势能）
    if is_game_over:
        return -100.0

    # 2. 核心魔法：计算棋盘的“势能 (Potential)”
    def calc_potential(board):
        heights = np.zeros(BOARD_WIDTH, dtype=int)
        holes = 0
        
        # 计算每列高度和空洞
        for col in range(BOARD_WIDTH):
            col_data = board[:, col]
            occ = np.where(col_data > 0)[0]
            if len(occ) > 0:
                heights[col] = BOARD_HEIGHT - occ[0]
                holes += np.sum(col_data[occ[0]:] == 0)
                
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(BOARD_WIDTH - 1))
        max_height = np.max(heights)
        
        # 寻找“井 (Well)”：某列比相邻的列都要低
        well_depths = np.zeros(BOARD_WIDTH, dtype=int)
        for i in range(BOARD_WIDTH):
            left = heights[i-1] if i > 0 else BOARD_HEIGHT
            right = heights[i+1] if i < BOARD_WIDTH - 1 else BOARD_HEIGHT
            depth = min(left, right) - heights[i]
            if depth > 0:
                well_depths[i] = depth
                
        max_well = np.max(well_depths)
        # 俄罗斯方块最多消4行，所以井深大于4的部分不给额外奖励
        capped_well = min(12, max_well) 
        
        # 【关键诱惑】：如果棋盘上没有空洞，我们就重赏这个“井”！
        # 每1格井深奖励 15 分。这会刺激 AI 去主动建一堵缺了一个口的墙。
        well_bonus = (capped_well * 15) if holes == 0 else 0
        
        # 势能公式：
        # 极度厌恶空洞 (-50)
        # 轻微厌恶崎岖 (-2，因为有了 well_bonus，它不会因为建井的崎岖而害怕)
        # 轻微厌恶太高 (-1)
        potential = -(holes * 50) - (bumpiness * 2) - (max_height * 1) + well_bonus
        return potential

    # 3. 势能差计算
    prev_phi = calc_potential(prev_board)
    post_phi = calc_potential(post_board)
    potential_diff = post_phi - prev_phi
    
    # 4. 存活保底分：只要走这一步没死，且没造大孽，保底给一点分，打破“寻死”循环
    survival_bonus = 1.0
    
    # 最终奖励 = 实际得分 + 势能变化 + 存活奖励
    reward = game_reward + potential_diff + survival_bonus
    return reward