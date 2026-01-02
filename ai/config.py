# ai/config.py

import torch
import os

ACTION_DIM = 2304
GRID_WIDTH_X = 12
GRID_HEIGHT_Y = 24
OFFSET_X = 2

# --- Training Hyperparams ---
BATCH_SIZE = 1 
LR = 0.0003
MEMORY_SIZE = 300000
GAMMA = 0.99
SCORE_SCALER = 20
SCORE_OFFSET = 2
ATTEMPTS_PER_SEED = 100

# --- MCTS Config ---
MCTS_SIMS_TRAIN = 300  # 适当调整
MCTS_SIMS_EVAL = 200
MCTS_DIRICHLET = 0.3
MCTS_EPSILON = 0.25
NUM_WORKERS = 10   # 默认进程数
MCTS_C_PUCT = 4.0
TAU_INIT = 1.0
TAU_DECAY = 0.995

# --- Game Limits ---
MAX_STEPS_TRAIN = 5000
MAX_STEPS_EVAL = 200000

# --- Paths ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"
BACKUP_DIR = "backups"

# --- Device ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)