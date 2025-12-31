# ai/config.py

import torch
import os

# --- Training Hyperparams ---
BATCH_SIZE = 512  # Larger for stability
LR = 0.0005  # Lower to prevent overshooting
MEMORY_SIZE = 200000  # Double for more diverse replays
GAMMA = 0.995  # Slight increase for longer horizons

# --- MCTS Config ---
MCTS_SIMS_TRAIN = 600      # 可以在困难阶段临时增大
MCTS_SIMS_EVAL = 2000
MCTS_DIRICHLET = 0.3  # Slightly higher noise
MCTS_EPSILON = 0.25
NUM_WORKERS = 10  # Scale if hardware allows
MCTS_C_PUCT = 2.0  # More exploration
TAU_INIT = 1.0
TAU_DECAY = 0.995  # Slower decay for prolonged stochasticity

# --- Game Limits ---

MAX_STEPS_TRAIN = 20000  # Encourage even longer survival
MAX_STEPS_EVAL = 200000

# --- Paths ---
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"
BACKUP_DIR = "backups"

# --- Device ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if not os.path.exists(BACKUP_DIR):
    os.makedirs(BACKUP_DIR)