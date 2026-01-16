# ai/config.py

from torch import cuda
import os

# --- Board / Action Config ---
ACTION_DIM = 2304
GRID_WIDTH_X = 12
GRID_HEIGHT_Y = 24
OFFSET_X = 2

# --- Inference Flags ---
FLAG_IDLE = 0
FLAG_REQ_READY = 1
FLAG_RES_READY = 2

# --- Training Hyperparams ---
# 建议: 初期用小 Batch (512/1024) 加快迭代，后期可加到 2048
BATCH_SIZE = 1024

# 建议: 稍微调大一点 LR 以便快速脱离随机初始化状态
LR = 1e-4  

# 建议: 减小 Memory Size，让初期产生的"笨蛋数据"更快被淘汰
MEMORY_SIZE = 20000  

GAMMA = 0.99

# 建议: 调小 Scaler，放大奖励/惩罚信号
# 生存(+1) -> 0.02, 死亡(-50) -> -1.0, 消行(+50) -> +1.0
# 这让"死亡"和"消行"对模型来说变得非常重要
SCORE_SCALER = 100

TANH_OFFSET = 0 # 建议设为 0，保持 tanh 自然的 -1 到 1 范围
QUEUE_MAX_SIZE = 32
ATTEMPTS_PER_SEED = 100

# 贪婪引导概率 (40% 概率使用贪婪算法带路，只用于生成数据，不用于 Label)
GREEDY_EPSILON = 0.11

# --- MCTS Config ---
MCTS_BATCH_SIZE = 128

# 这样能减少随机噪声，让"好动作"和"坏动作"的访问次数拉开差距
MCTS_SIMS_TRAIN = 500  
MCTS_SIMS_EVAL = 500

MCTS_DIRICHLET = 0.2
MCTS_EPSILON = 0.15
NUM_WORKERS = 8       # 根据你的 CPU 核心数调整
MCTS_C_PUCT = 1.4
TAU_INIT = 1.0
TAU_DECAY = 0.995

# --- Game Limits ---
# 单局最大步数，防止训练后期一局跑太久卡住 Worker
MAX_STEPS_TRAIN = 5000 
MAX_STEPS_EVAL = 10000


# --- [新增] MCTS Value 混合权重 ---
# 解释: 决定 MCTS 相信 "直觉(NN)" 还是 "规则(Heuristics)"
# 现状: NN 处于习得性无助状态，需要大幅依靠 Heuristics 强制引导
MCTS_VAL_WEIGHT_NN = 0.3        # 杀掉 NN 权重
MCTS_VAL_WEIGHT_HEURISTIC = 0.7 # 规则独裁

# --- [极度严格修正] 温度控制策略 ---
# 解释: 禁止任何随机探索，强迫模型从第一步开始就必须走最优解
TEMP_START_STEPS = 9        # 0 步随机！
TEMP_HIGH = 0.8             # 即使进入探索逻辑，也保持极低温度
TEMP_LOW = 0.3              # 几乎就是 Argmax (只选概率最高的)
TEMP_EMERGENCY_HEIGHT = 10  # 稍微降低危险阈值
TEMP_EMERGENCY = 0.01       # 绝对贪婪

# --- Paths ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.npz" 
BACKUP_DIR = "backups"

# --- Device ---
DEVICE = 'cuda' if cuda.is_available() else 'cpu'

DEBUG_MODE = False  
DEBUG_FREQ = 10000

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)