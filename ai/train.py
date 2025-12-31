# ai/train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import time
import gc
from torch.amp import autocast, GradScaler
from .utils import TetrisGame
from .model import TetrisPolicyValue
from .mcts import MCTS
from .reward import get_reward, calculate_heuristics
from . import config

# ==========================================
# 1. 高效的 Numpy Replay Buffer
# ==========================================
class NumpyReplayBuffer:
    def __init__(self, capacity, board_shape=(20, 10), ctx_dim=11, action_dim=256):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 使用 int8 存储 Board，节省 75% 内存 (Tetris Board 只有 0 或 1-N，int8 足够)
        self.boards = np.zeros((capacity, *board_shape), dtype=np.int8)
        self.ctxs = np.zeros((capacity, ctx_dim), dtype=np.float32)
        self.probs = np.zeros((capacity, action_dim), dtype=np.float32)
        # Value 是一维的
        self.values = np.zeros((capacity, 1), dtype=np.float32)

    def add_batch(self, data):
        """
        一次性添加一局游戏的数据。
        data: list of [board, ctx, probs, value]
        """
        if not data:
            return
            
        # 解包数据 (利用 zip(*data) 快速转置)
        # boards: tuple of (20, 10) arrays
        boards, ctxs, probs, values = zip(*data)
        
        n = len(boards)
        
        # 计算写入索引（支持循环覆盖）
        indices = np.arange(self.ptr, self.ptr + n) % self.capacity
        
        # 批量写入 Numpy 数组 (比 list append 快得多)
        # 注意：这里会自动将 int32/float 的 board 转换为 int8 存储
        self.boards[indices] = np.array(boards, dtype=np.int8)
        self.ctxs[indices] = np.array(ctxs, dtype=np.float32)
        self.probs[indices] = np.array(probs, dtype=np.float32)
        self.values[indices] = np.array(values, dtype=np.float32).reshape(-1, 1)
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        # O(1) 随机采样
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # 返回数据 (board 转回 float32 供 Tensor 使用)
        return (
            self.boards[idxs].astype(np.float32), 
            self.ctxs[idxs], 
            self.probs[idxs], 
            self.values[idxs]
        )

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            # 只保存有效数据部分，减少磁盘占用
            pickle.dump({
                'boards': self.boards[:self.size],
                'ctxs': self.ctxs[:self.size],
                'probs': self.probs[:self.size],
                'values': self.values[:self.size],
                'ptr': self.ptr,
                'size': self.size
            }, f)

    def load(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # 如果容量不匹配，需要截断或填充，这里假设匹配
                sz = data['size']
                self.boards[:sz] = data['boards']
                self.ctxs[:sz] = data['ctxs']
                self.probs[:sz] = data['probs']
                self.values[:sz] = data['values']
                self.ptr = data['ptr']
                self.size = sz
                print(f"[Buffer] Loaded {sz} samples.")
        except Exception as e:
            print(f"[Buffer] Load failed: {e}")

# ==========================================
# 2. 优化后的 Battle Simulation
# ==========================================
def normalize_score(score):
    return np.tanh((score + config.TANH_OFFSET) / config.TANH_SCALER)

def battle_simulation(game, mcts, render=False):
    """
    game: 复用的 TetrisGame 实例
    mcts: 复用的 MCTS 实例
    """
    # 重置游戏状态 (C++ 内部重置，不涉及 Python 对象销毁)
    game.reset(random.randint(0, 1000000))
    
    # 如果需要渲染
    if render:
        game.enable_render()
    
    training_data = []
    steps = 0
    pieces_placed = 0
    total_score = 0.0
    total_lines = 0
    tetris_count = 0
    
    # 统计数据复用
    episode_stats = {
        'pieces_since_last_garbage': 0,
        'total_garbage_cleared': 0,
        'garbage_clear_events': 0,
        'pieces_used_for_garbage': 0
    }
    
    # 获取初始状态 (Views)
    board_view, prev_ctx_view = game.get_state()
    # 必须 Copy，因为我们要把这个状态存入 training_data，且后续 step 会修改底层内存
    board = board_view.copy()
    prev_ctx = prev_ctx_view.copy()
    
    prev_metrics = calculate_heuristics(board)
    
    while True:
        if render:
            game.render()
            time.sleep(0.05) # 稍微快一点
        
        # MCTS 运行 (复用 Buffer)
        root = mcts.run(game)
        
        # 获取策略
        temp = 1.0 if steps < 30 else 0.5
        action_probs = mcts.get_action_probs(root, temp=temp)
        
        # 收集数据 [Board, Ctx, Probs, Value_Placeholder]
        # 注意：这里存入的是已经 copy 过的 board
        training_data.append([board, prev_ctx, action_probs, None])
        
        # 采样动作
        legal = game.get_legal_moves()
        if len(legal) == 0:
            break
            
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        move = root.legal_moves[action_idx]
        
        # 执行
        episode_stats['pieces_since_last_garbage'] += 1
        res = game.step(move[0], move[1], move[2], move[4])
        
        pieces_placed += 1
        # if pieces_placed % 5 == 0 and not res[3]:
        #     # 随机添加垃圾行
        #     garbage = random.randint(1, 4)
        #     game.receive_garbage(garbage)
            
        # 获取新状态
        next_board_view, next_ctx_view = game.get_state()
        next_board = next_board_view.copy()
        next_ctx = next_ctx_view.copy()
        
        cur_metrics = calculate_heuristics(next_board)
        
        step_result = {
            'lines_cleared': res[0],
            'damage_sent': res[1],
            'attack_type': res[2],
            'game_over': res[3],
            'combo': res[4]
        }
        
        step_reward, force_over = get_reward(
            step_result, cur_metrics, prev_metrics, steps,
            context=next_ctx, prev_context=prev_ctx,
            episode_stats=episode_stats, is_training=True
        )
        
        total_score += step_reward
        total_lines += step_result['lines_cleared']
        if step_result['lines_cleared'] == 4:
            tetris_count += 1
            
        steps += 1
        
        # 滚动状态
        board = next_board
        prev_ctx = next_ctx
        prev_metrics = cur_metrics
        
        if step_result['game_over'] or force_over or steps > config.MAX_STEPS_TRAIN:
            break
            
    # 回溯 Value
    final_value = normalize_score(total_score)
    for item in training_data:
        item[3] = final_value
        
    stats = {
        "score": total_score,
        "steps": steps,
        "lines": total_lines,
        "tetrises": tetris_count
    }
    
    return training_data, stats

# ==========================================
# 3. 优化后的主训练循环
# ==========================================
def save_checkpoint(net, optimizer, buffer, game_idx):
    print(f"[Saving] Checkpoint at game {game_idx}")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_idx
    }, config.CHECKPOINT_FILE)
    
    # Buffer 单独保存
    if buffer.size > 0:
        buffer.save(config.MEMORY_FILE)
    print("[Saving] Done.")

def load_checkpoint(net, optimizer, buffer):
    start_idx = 0
    if os.path.exists(config.CHECKPOINT_FILE):
        print("[Loading] Checkpoint found.")
        checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_idx = checkpoint['game_idx']
        
    if os.path.exists(config.MEMORY_FILE):
        buffer.load(config.MEMORY_FILE)
            
    return start_idx

def train():
    torch.backends.cudnn.benchmark = True
    
    net = TetrisPolicyValue().to(config.DEVICE)
    # 启用 channels_last 内存布局加速卷积
    net = net.to(memory_format=torch.channels_last)
    
    optimizer = optim.Adam(net.parameters(), lr=config.LR)
    scaler = GradScaler()
    
    # 初始化高效 Buffer
    buffer = NumpyReplayBuffer(config.MEMORY_SIZE)
    
    game_idx = load_checkpoint(net, optimizer, buffer)
    
    print(f"[Train] Optimized Single-process training on {config.DEVICE}.")
    
    # --- 关键优化：复用对象 ---
    # 1. 复用 TetrisGame 实例 (避免 ctypes 重复 init)
    game_instance = TetrisGame()
    
    # 2. 复用 MCTS 实例 (避免 CUDA tensor 重复 alloc)
    # MCTS 内部会持有 model 的引用
    mcts_instance = MCTS(net, device=config.DEVICE, num_simulations=config.MCTS_SIMS_TRAIN)
    
    RENDER_INTERVAL = 20 
    
    try:
        while True:
            render = (game_idx % RENDER_INTERVAL == 0)
            
            # 动态调整模拟次数
            current_sims = config.MCTS_SIMS_EVAL if render else config.MCTS_SIMS_TRAIN
            mcts_instance.num_simulations = current_sims
            
            # 1. 生成数据 (传入复用的对象)
            # mcts.run() 会自动调用 model.eval()
            new_data, stats = battle_simulation(game_instance, mcts_instance, render=render)
            
            # 批量存入 Buffer
            buffer.add_batch(new_data)
            game_idx += 1
            
            if buffer.size < config.BATCH_SIZE:
                continue
            
            # 2. 训练阶段
            new_samples = len(new_data)
            # 动态训练步数：每生成 BATCH_SIZE 数据，训练 2-3 次
            train_steps = max(1, int(new_samples / config.BATCH_SIZE * 5.0))
            train_steps = min(train_steps, 20)
            
            if buffer.size < 2000:
                train_steps = 1

            total_loss = 0
            
            # 显式切换到训练模式 (启用 BN update / Dropout)
            net.train()
            
            for _ in range(train_steps):
                # 极速采样：直接从 Numpy Array 切片，无需 stack
                b_board_np, b_ctx_np, b_policy_np, b_value_np = buffer.sample(config.BATCH_SIZE)
                
                # 转换为 Tensor
                b_board = torch.from_numpy(b_board_np).to(config.DEVICE, non_blocking=True)
                b_ctx = torch.from_numpy(b_ctx_np).to(config.DEVICE, non_blocking=True)
                b_policy = torch.from_numpy(b_policy_np).to(config.DEVICE, non_blocking=True)
                b_value = torch.from_numpy(b_value_np).to(config.DEVICE, non_blocking=True)

                # 确保输入是 channels_last 格式
                if b_board.dim() == 3:
                    b_board = b_board.unsqueeze(1)
                b_board = b_board.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad()
                
                with autocast(device_type='cuda'):
                    p, v = net(b_board, b_ctx)
                    
                    policy_loss = -torch.sum(b_policy * F.log_softmax(p, dim=1), dim=1).mean()
                    value_loss = F.mse_loss(v, b_value)
                    entropy = -torch.mean(torch.sum(F.softmax(p, dim=1) * F.log_softmax(p, dim=1), dim=1))
                    
                    loss = policy_loss + value_loss - 0.01 * entropy
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                
                total_loss += loss.item()
            
            # 训练结束，打印日志
            if game_idx % 10 == 0:
                avg_loss = total_loss / train_steps if train_steps > 0 else 0
                print(f"[Train] Game {game_idx} | Score {stats['score']:.0f} | Steps {stats['steps']} | Loss {avg_loss:.4f} | Buf {buffer.size}")
            
            # 定期保存
            if game_idx % 50 == 0:
                save_checkpoint(net, optimizer, buffer, game_idx)
                gc.collect() 
                
    except KeyboardInterrupt:
        print("[Interrupt] Saving final checkpoint.")
        save_checkpoint(net, optimizer, buffer, game_idx)

if __name__ == "__main__":
    train()