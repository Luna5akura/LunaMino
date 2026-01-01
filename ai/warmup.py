# ai/warmup.py

import numpy as np
import pickle
import os
import gc
import ctypes
from rich.console import Console
from rich.progress import track

# 引入 lib 和常量
from ai.utils import TetrisGame, lib, MacroAction, LegalMoves
from ai.reward import calculate_heuristics, get_reward
from ai import config

# ==========================================
# 1. 定义复用缓冲区 (Global Reusable Buffers)
# ==========================================
# 我们不需要每次模拟都 new 一个 array，全程复用这一组
_sim_board_buf = (ctypes.c_int * 200)()
_sim_queue_buf = (ctypes.c_int * 5)()
_sim_hold_buf  = (ctypes.c_int * 1)()
_sim_meta_buf  = (ctypes.c_int * 5)()

# 获取指针
_p_board = ctypes.cast(_sim_board_buf, ctypes.POINTER(ctypes.c_int))
_p_queue = ctypes.cast(_sim_queue_buf, ctypes.POINTER(ctypes.c_int))
_p_hold  = ctypes.cast(_sim_hold_buf, ctypes.POINTER(ctypes.c_int))
_p_meta  = ctypes.cast(_sim_meta_buf, ctypes.POINTER(ctypes.c_int))

# 预先创建 Numpy 视图 (Zero-copy)
# 注意：Tetris 底层是 1D array，我们 reshape 成 (20, 10)，并且倒序
_np_board_view = np.ctypeslib.as_array(_sim_board_buf).reshape(20, 10)[::-1]

def fast_simulate_move(game_ptr, move, w_holes, w_height, w_bump, w_lines):
    """
    极速模拟：不创建 Python 对象，直接调用 C 接口
    """
    # 1. Clone (C Level)
    sim_ptr = lib.clone_tetris(game_ptr)
    if not sim_ptr:
        return -float('inf')

    try:
        # 2. Step (C Level)
        # move: [x, y, rotation, landing_height, use_hold]
        # ai_step 返回的是结构体值，不是指针
        res = lib.ai_step(sim_ptr, move[0], move[1], move[2], move[4])
        
        # 如果死局，直接返回极低分
        if res.is_game_over:
            return -1e9

        # 3. Get State (写入全局复用 buffer)
        lib.ai_get_state(sim_ptr, _p_board, _p_queue, _p_hold, _p_meta)
        
        # 4. Heuristics
        # _np_board_view 的内容已经被上面的 ai_get_state 更新了
        # 注意：calculate_heuristics 内部可能会 copy，但这已经是最小开销了
        metrics = calculate_heuristics(_np_board_view)
        
        # 5. Score
        score = 0.0
        score += metrics['holes'] * w_holes
        score += metrics['max_height'] * w_height
        score += metrics['bumpiness'] * w_bump
        score += res.lines_cleared * w_lines
        
        return score

    finally:
        # 6. Destroy (C Level)
        lib.destroy_tetris(sim_ptr)

class WarmupBuffer:
    def __init__(self, capacity, board_shape=(20, 10), ctx_dim=11, action_dim=config.ACTION_DIM):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        print(f"[Buffer] Allocating memory for {capacity} samples...")
        # 尝试分配内存，如果不足则自动减半
        try:
            self.boards = np.zeros((capacity, *board_shape), dtype=np.int8)
            self.probs = np.zeros((capacity, action_dim), dtype=np.float32)
            self.ctxs = np.zeros((capacity, ctx_dim), dtype=np.float32)
            self.values = np.zeros((capacity, 1), dtype=np.float32)
        except MemoryError:
            new_cap = 100000
            print(f"[Warning] RAM insufficient. Reducing capacity to {new_cap}")
            self.capacity = new_cap
            self.boards = np.zeros((new_cap, *board_shape), dtype=np.int8)
            self.probs = np.zeros((new_cap, action_dim), dtype=np.float32)
            self.ctxs = np.zeros((new_cap, ctx_dim), dtype=np.float32)
            self.values = np.zeros((new_cap, 1), dtype=np.float32)

    def add_batch(self, boards, ctxs, probs, values):
        n = len(boards)
        if n == 0: return
        indices = np.arange(self.ptr, self.ptr + n) % self.capacity
        self.boards[indices] = boards
        self.ctxs[indices] = ctxs
        self.probs[indices] = probs
        self.values[indices] = values
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def save(self, path):
        print(f"Saving {self.size} samples to {path}...")
        with open(path, 'wb') as f:
            pickle.dump({
                'ptr': self.ptr, 'size': self.size,
                'boards': self.boards[:self.size],
                'ctxs': self.ctxs[:self.size],
                'probs': self.probs[:self.size],
                'values': self.values[:self.size]
            }, f)

def get_greedy_move_optimized(game):
    """
    优化后的入口函数
    """
    legal_moves = game.get_legal_moves()
    if len(legal_moves) == 0:
        return None, None

    best_move_idx = -1
    best_score = -float('inf')
    
    # 贪心参数
    W_HOLES = -5.0
    W_HEIGHT = -1.0
    W_BUMP = -1.0
    W_LINES = 5.0 

    # 遍历所有动作
    for i, move in enumerate(legal_moves):
        # 调用无对象开销的模拟函数
        score = fast_simulate_move(game.ptr, move, W_HOLES, W_HEIGHT, W_BUMP, W_LINES)
        
        if score > best_score:
            best_score = score
            best_move_idx = i

    return best_move_idx, legal_moves

def run_warmup():
    console = Console()
    console.rule("[bold green]Generating Warmup Data (Low Memory Mode)[/bold green]")
    
    # 稍微减少局数，保证质量即可
    NUM_GAMES = 300
    MAX_STEPS_PER_GAME = 1000
    
    buffer = WarmupBuffer(config.MEMORY_SIZE)
    game = TetrisGame()
    
    # 预分配临时存储，避免循环内 malloc
    tmp_boards = np.zeros((MAX_STEPS_PER_GAME, 20, 10), dtype=np.int8)
    tmp_ctxs = np.zeros((MAX_STEPS_PER_GAME, 11), dtype=np.float32)
    tmp_probs = np.zeros((MAX_STEPS_PER_GAME, config.ACTION_DIM), dtype=np.float32)
    
    total_score_sum = 0
    
    for i in track(range(NUM_GAMES), description="Playing Greedy Games..."):
        seed = i + int(time.time())
        game.reset(seed=seed)
        
        steps = 0
        game_score = 0
        
        board_view, ctx_view = game.get_state()
        curr_board = board_view.copy()
        curr_ctx = ctx_view.copy()
        prev_metrics = calculate_heuristics(curr_board)
        
        while steps < MAX_STEPS_PER_GAME:
            # 使用优化后的决策函数
            best_idx, legal_moves = get_greedy_move_optimized(game)
            
            if best_idx is None:
                break
                
            # Store State
            tmp_boards[steps] = curr_board.astype(np.int8)
            tmp_ctxs[steps] = curr_ctx
            
            # One-Hot Action
            tmp_probs[steps].fill(0)
            tmp_probs[steps, best_idx] = 1.0
            
            # Execute
            move = legal_moves[best_idx]
            res = game.step(move[0], move[1], move[2], move[4])
            
            # Reward
            next_board_view, next_ctx_view = game.get_state()
            next_board = next_board_view.copy()
            next_ctx = next_ctx_view.copy()
            cur_metrics = calculate_heuristics(next_board)
            
            step_result = {
                'lines_cleared': res[0], 'game_over': res[3], 'b2b_count': res[4], 'combo': res[5]
            }
            # print(res,step_result)
            r, _ = get_reward(step_result, cur_metrics, prev_metrics, steps)
            game_score += r
            
            curr_board = next_board
            curr_ctx = next_ctx
            prev_metrics = cur_metrics
            steps += 1
            if steps % 5 == 0: game.receive_garbage(1) 
            
            if res[3]:
                break
        # Save to buffer
        if steps > 0:
            final_value = np.tanh(game_score / config.TANH_SCALER)
            batch_values = np.full((steps, 1), final_value, dtype=np.float32)
            
            buffer.add_batch(
                tmp_boards[:steps],
                tmp_ctxs[:steps],
                tmp_probs[:steps],
                batch_values
            )
            total_score_sum += game_score
            
        if i % 10 == 0:
            # 手动 GC，双重保险
            gc.collect()
            print(f" Game {i} | Steps: {steps} | Score: {game_score:.1f} | Buffer: {buffer.size}")

    console.print(f"[bold blue]Warmup Finished![/bold blue]")
    if NUM_GAMES > 0:
        console.print(f"Avg Score: {total_score_sum / NUM_GAMES:.2f}")
    
    buffer.save(config.MEMORY_FILE)
    console.print(f"[green]Saved to {config.MEMORY_FILE}[/green]")

import time # 补上 import

if __name__ == "__main__":
    run_warmup()