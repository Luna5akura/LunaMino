# ai/config.py
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
Y_MIN = 0 # Assumed; may need adjustment
Y_MAX = 23 # Assumed to cover possible heights; may need adjustment
ROT_MAX = 3
HOLD_MAX = 1
# Model architecture
CNN_CHANNELS = [32, 64]
HIDDEN_DIM = 128
EMBED_DIM = 64
# Training hyperparameters
GAMMA = 0.99 # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999
BUFFER_SIZE = 100000
BATCH_SIZE = 64
LR = 0.001
TARGET_UPDATE_FREQ = 1000 # Steps between target network updates
MAX_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000 # Prevent infinite games
# Reward shaping coefficients
REWARD_LINES = 100.0 # Bonus per line cleared
REWARD_HEIGHT = -0.05 # Penalty per unit increase in aggregate height
REWARD_HOLES = -4.0 # Penalty per new hole
REWARD_BUMPINESS = -0.1 # Penalty per unit increase in bumpiness
REWARD_LANDING = -0.01 # Penalty per unit landing height
REWARD_GAME_OVER = -500.0 # Terminal penalty
# Multi-threading configuration
NUM_THREADS = 8  # Number of threads for parallel episode execution; adjust as needed