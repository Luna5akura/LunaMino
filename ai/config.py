# ai/config.py

import torch
import os


ACTION_DIM = 2304 

# 定义偏移量常量，方便 Utils 使用
GRID_WIDTH_X = 12  # 覆盖 x=[-2, 9]
GRID_HEIGHT_Y = 24 # 覆盖 y=[0, 23]
OFFSET_X = 2       # 将 x=-2 映射到索引 0

# --- Training Hyperparams ---
BATCH_SIZE = 128  # Larger for stability
LR = 0.0001  # Lower to prevent overshooting
MEMORY_SIZE = 300000  # Increase for more diverse replays
GAMMA = 0.9  # Slight increase for longer horizons
SCORE_SCALER = 800
SCORE_OFFSET = 200

# --- MCTS Config ---
MCTS_SIMS_TRAIN = 2000  # Increase for better exploration
MCTS_SIMS_EVAL = 2000
MCTS_DIRICHLET = 0.5  # Slightly higher noise
MCTS_EPSILON = 0.35
NUM_WORKERS = 10  # Scale if hardware allows
MCTS_C_PUCT = 4.0  # More exploration
TAU_INIT = 1.0
TAU_DECAY = 0.99  # Slower decay for prolonged stochasticity

# --- Game Limits ---
MAX_STEPS_TRAIN = 5000  # Encourage even longer survival
MAX_STEPS_EVAL = 200000

# --- Paths ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"
BACKUP_DIR = "backups"

# --- Device ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)