# ai/test_model.py
import ctypes
import numpy as np
import time

# Constants from config
ACTION_DIM = 2304
GRID_WIDTH_X = 12
GRID_HEIGHT_Y = 24
OFFSET_X = 2

# Load library (assume compiled libtetris.so is in current dir)
lib = ctypes.CDLL('./libtetris.so')

# Structures
class MacroAction(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_int8),
        ('y', ctypes.c_int8),
        ('rotation', ctypes.c_int8),
        ('landing_height', ctypes.c_int8),
        ('use_hold', ctypes.c_bool)
    ]

class LegalMoves(ctypes.Structure):
    _fields_ = [
        ('count', ctypes.c_int),
        ('moves', MacroAction * 256)
    ]

class StepResult(ctypes.Structure):
    _fields_ = [
        ('lines_cleared', ctypes.c_int),
        ('damage_sent', ctypes.c_int),
        ('attack_type', ctypes.c_int),
        ('is_game_over', ctypes.c_bool),
        ('b2b_count', ctypes.c_int),
        ('combo_count', ctypes.c_int)
    ]

# Function prototypes
lib.create_tetris.argtypes = [ctypes.c_int]
lib.create_tetris.restype = ctypes.c_void_p

lib.destroy_tetris.argtypes = [ctypes.c_void_p]

lib.ai_get_legal_moves.argtypes = [ctypes.c_void_p, ctypes.POINTER(LegalMoves)]

lib.ai_get_state.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),  # board (200)
    ctypes.POINTER(ctypes.c_int),  # queue (5)
    ctypes.POINTER(ctypes.c_int),  # hold (1)
    ctypes.POINTER(ctypes.c_int)   # meta (5): [b2b, combo, can_hold, piece_type, pending_garbage]
]

lib.ai_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ai_step.restype = StepResult

# Helper to get legal moves as np array
def get_legal_moves(ptr):
    moves_struct = LegalMoves()
    lib.ai_get_legal_moves(ptr, ctypes.byref(moves_struct))
    count = moves_struct.count
    if count == 0:
        return np.empty((0, 5), dtype=np.int8)
    moves_np = np.frombuffer(moves_struct.moves, dtype=np.int8).reshape(256, ctypes.sizeof(MacroAction))[:count, :5]
    return moves_np.copy()

# Helper to compute action_ids (from your get_legal_moves_with_ids)
def compute_action_ids(moves):
    if len(moves) == 0:
        return np.array([], dtype=np.int64)
    xs = moves[:, 0].astype(np.int64)
    rots = moves[:, 2].astype(np.int64)
    land_ys = moves[:, 3].astype(np.int64)
    holds = moves[:, 4].astype(np.int64)
    xs_idx = np.clip(xs + OFFSET_X, 0, GRID_WIDTH_X - 1)
    ys_idx = np.clip(land_ys, 0, GRID_HEIGHT_Y - 1)
    stride_x = GRID_HEIGHT_Y
    stride_rot = GRID_WIDTH_X * GRID_HEIGHT_Y
    stride_hold = GRID_WIDTH_X * GRID_HEIGHT_Y * 4
    action_ids = (holds * stride_hold) + (rots * stride_rot) + (xs_idx * stride_x) + ys_idx
    return action_ids

# Main experiment
def run_experiment():
    # Create game
    seed = int(time.time())
    ptr = lib.create_tetris(seed)
    
    try:
        # Buffers for state
        board_buf = (ctypes.c_int * 200)()
        queue_buf = (ctypes.c_int * 5)()
        hold_buf = (ctypes.c_int * 1)()
        meta_buf = (ctypes.c_int * 5)()
        board_ptr = ctypes.cast(board_buf, ctypes.POINTER(ctypes.c_int))
        queue_ptr = ctypes.cast(queue_buf, ctypes.POINTER(ctypes.c_int))
        hold_ptr = ctypes.cast(hold_buf, ctypes.POINTER(ctypes.c_int))
        meta_ptr = ctypes.cast(meta_buf, ctypes.POINTER(ctypes.c_int))
        
        # Advance until I-piece (type 1)
        steps = 0
        max_steps = 1000  # Safety limit
        while steps < max_steps:
            lib.ai_get_state(ptr, board_ptr, queue_ptr, hold_ptr, meta_ptr)
            current_piece = meta_buf[3]
            if current_piece == 1:  # I-piece
                print(f"Found I-piece at step {steps}")
                break
            
            # Take a random legal move to advance
            moves = get_legal_moves(ptr)
            if len(moves) == 0:
                print("No legal moves, game over early")
                return
            rand_idx = np.random.randint(len(moves))
            move = moves[rand_idx]
            lib.ai_step(ptr, move[0], move[1], move[2], move[4])
            steps += 1
        
        if steps >= max_steps:
            print("Max steps reached without I-piece")
            return
        
        # Now get legal moves and action_ids for this state
        moves = get_legal_moves(ptr)
        action_ids = compute_action_ids(moves)
        
        # Mock model logits (sequential for easy verification: logits[i] == i)
        mock_logits = np.arange(ACTION_DIM, dtype=np.float32)
        
        # Extract valid_logits using ids
        valid_logits = mock_logits[action_ids]
        
        # Print for verification
        print("\nLegal Moves (x, y, rot, land_h, hold):")
        print(moves)
        
        print("\nAction IDs:")
        print(action_ids)
        
        print("\nExtracted Valid Logits (should match action_ids since logits == arange):")
        print(valid_logits)
        
        # Manual verification check
        if np.all(valid_logits == action_ids):
            print("\nVerification PASSED: valid_logits correspond to correct action_ids")
        else:
            print("\nVerification FAILED: Mismatch detected")
    
    finally:
        lib.destroy_tetris(ptr)

if __name__ == '__main__':
    run_experiment()