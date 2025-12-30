# ai/config.py
import torch
import os
# --- 训练超参数 ---
BATCH_SIZE = 64  # 初始稳定
LR = 0.003
MEMORY_SIZE = 20000  # 增大以积累多样经验
GAMMA = 0.99  # 略高，鼓励长远生存和B2B链
# --- MCTS 配置 ---
MCTS_SIMS_TRAIN = 100  # 起始低，加速初始迭代；后期增
MCTS_SIMS_EVAL = 800
MCTS_DIRICHLET = 0.4
MCTS_EPSILON = 0.3
NUM_WORKERS = 10
MCTS_C_PUCT = 3.5
TAU_INIT = 1.0  # 初始温度高，鼓励探索
TAU_DECAY = 0.995
# --- 游戏限制 ---
MAX_STEPS_TRAIN = 3000  # 增大鼓励生存训练
MAX_STEPS_EVAL = 50000
# --- 路径配置 ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"
BACKUP_DIR = "backups"
# --- 设备 ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)