# ai/config.py (小改：加 MCTS_C_PUCT=2.0 增强探索)
import torch
import os
# --- 训练超参数 ---
BATCH_SIZE = 128
LR = 0.005
MEMORY_SIZE = 100000 # 经验池大小
gamma = 0.99 # 折扣因子 (如未来引入)
# --- MCTS 配置 ---
MCTS_SIMS_TRAIN = 1200 # 训练时的搜索次数 (追求速度)
MCTS_SIMS_EVAL = 800 # 观看/评估时的搜索次数 (追求质量) ↑ 更好 eval
MCTS_DIRICHLET = 0.3
MCTS_EPSILON = 0.25

NUM_WORKERS = 10 # mp_train 的进程数 (建议 CPU核心数 - 2)
MCTS_C_PUCT = 3.5 # 新: UCB 常数，更大=更探索 (原1.0)
# --- 游戏限制 ---
MAX_STEPS_TRAIN = 1000 # 训练时每局最大步数 (防止死循环)
MAX_STEPS_EVAL = 50000 # 观看时几乎不限制步数
# --- 路径配置 ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"
BACKUP_DIR = "backups" # 备份文件夹路径
# --- 设备 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --- 确保备份文件夹存在 ---
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)

