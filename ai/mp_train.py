# ai/mp_train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import random
import os
import pickle
import time
import gc
import queue
import shutil  # <--- [新增 1] 引入 shutil 用于复制文件
from torch.amp import autocast, GradScaler

# 引入优化后的模块
from .utils import TetrisGame
from .model import TetrisPolicyValue
from .mcts import MCTS
from . import config
from .reward import get_reward, calculate_heuristics

# UI Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich import box
from datetime import datetime

# ==========================================
# 1. 优化配置
# ==========================================
# 强制 Worker 仅使用 CPU，防止 CUDA 初始化冲突
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Trainer 看到的设备
WORKER_DEVICE = 'cpu'
TRAINER_DEVICE = config.DEVICE

# 限制 NumPy 线程数，防止 Worker 抢占 CPU 核心
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# ==========================================
# 2. 高效 Numpy Replay Buffer (内嵌定义以保持独立性)
# ==========================================
class NumpyReplayBuffer:
    def __init__(self, capacity, board_shape=(20, 10), ctx_dim=11, action_dim=config.ACTION_DIM):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 优化：使用 int8 存储 Board，节省 75% 内存与传输带宽
        self.boards = np.zeros((capacity, *board_shape), dtype=np.int8)
        self.ctxs = np.zeros((capacity, ctx_dim), dtype=np.float32)
        self.probs = np.zeros((capacity, action_dim), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

    def add_batch(self, boards, ctxs, probs, values):
        """一次性写入一批数据 (Numpy Arrays)"""
        n = len(boards)
        if n == 0: return
        
        indices = np.arange(self.ptr, self.ptr + n) % self.capacity
        
        self.boards[indices] = boards
        self.ctxs[indices] = ctxs
        self.probs[indices] = probs
        self.values[indices] = values
        
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.boards[idxs].astype(np.float32), # 转回 float32 给 Tensor
            self.ctxs[idxs],
            self.probs[idxs],
            self.values[idxs]
        )
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'ptr': self.ptr, 'size': self.size,
                'boards': self.boards[:self.size],
                'ctxs': self.ctxs[:self.size],
                'probs': self.probs[:self.size],
                'values': self.values[:self.size]
            }, f)

    def load(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                sz = data['size']
                self.boards[:sz] = data['boards']
                self.ctxs[:sz] = data['ctxs']
                self.probs[:sz] = data['probs']
                self.values[:sz] = data['values']
                self.ptr = data['ptr']
                self.size = sz
        except Exception as e:
            print(f"Failed to load buffer: {e}")

# ==========================================
# 3. 优化后的 Worker Process
# ==========================================
def worker_process(rank, shared_model, data_queue, device):
    # 再次确保单线程
    torch.set_num_threads(1)
    
    seed = rank + int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 本地模型 (CPU)
    local_model = TetrisPolicyValue().to(device)
    local_model.eval()
    
    # 复用 MCTS 和 Game 实例
    mcts = MCTS(local_model, device=device, num_simulations=config.MCTS_SIMS_TRAIN)
    game = TetrisGame()
    
    episode_stats = {
        'pieces_since_last_garbage': 0, 'total_garbage_cleared': 0,
        'garbage_clear_events': 0, 'pieces_used_for_garbage': 0
    }
    
    while True:
        # --- 优化 1: 仅在游戏开始前同步一次权重 ---
        # 避免在 MCTS 搜索过程中频繁从 GPU 拉取权重导致 IO 阻塞
        # 注意：shared_model 在 GPU，local 在 CPU，PyTorch 自动处理跨设备 copy
        try:
            local_model.load_state_dict(shared_model.state_dict())
        except Exception as e:
            # 极少数情况下可能遇到文件锁或传输错误
            print(f"Worker {rank} sync failed: {e}")

        # 重置游戏
        game_seed = random.randint(0, 1000000)
        game.reset(game_seed)
        game.receive_garbage(2)
        
        # 重置统计
        steps = 0
        score_sum = 0.0
        lines_sum = 0
        holes_sum = 0
        tetris_count = 0
        pieces_placed = 0
        for k in episode_stats: episode_stats[k] = 0
        
        # 本地数据缓存 (List)
        data_boards = []
        data_ctxs = []
        data_probs = []
        
        board_view, prev_ctx_view = game.get_state()
        board = board_view.copy()
        prev_ctx = prev_ctx_view.copy()
        prev_metrics = calculate_heuristics(board)
        
        while True:
            # MCTS 搜索
            root = mcts.run(game)
            
            # 获取策略
            temp = 1.0 if steps < 30 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)

            action_probs = np.nan_to_num(action_probs, nan=0.0)

            # 2. 重新归一化 (Renormalize)
            s = np.sum(action_probs)
            if s < 1e-9:
                # 如果全是 0，随机选一个合法动作
                # 但这里我们只有 2304 维的向量，不知道哪些是合法的 mask
                # 如果发生这种情况，说明 MCTS 彻底失败了，通常不会走到这里
                # 为了防止 Crash，我们均匀分布
                action_probs = np.ones_like(action_probs) / len(action_probs)
            else:
                action_probs /= s
            
            # --- 优化 2: 立即转换为 int8 存储 ---
            # 大幅减少 Queue 传输负荷
            data_boards.append(board.astype(np.int8))
            data_ctxs.append(prev_ctx)
            data_probs.append(action_probs)
            

            # 采样动作
            legal, ids = game.get_legal_moves_with_ids() # 确保这里用的是新接口
            if len(legal) == 0:
                break
    
            # === 修复：仅在合法动作范围内采样 ===
            # action_probs 是 2304 维的，直接 choice(2304) 可能会选到非法动作
            # 应该从 legal moves 中选
            
            # 提取当前合法动作对应的概率
            valid_probs = action_probs[ids]
            
            # 再次归一化 Valid Probs (防止精度误差)
            s_valid = np.sum(valid_probs)
            if s_valid < 1e-9:
                valid_probs = np.ones_like(valid_probs) / len(valid_probs)
            else:
                valid_probs /= s_valid
                
            # 在合法动作的索引 (0, 1, 2...) 中采样
            action_idx_in_legal = np.random.choice(len(legal), p=valid_probs)
            
            # 获取对应的 Move
            move = legal[action_idx_in_legal]
            
            episode_stats['pieces_since_last_garbage'] += 1
            res = game.step(move[0], move[1], move[2], move[4])
            
            pieces_placed += 1
            if pieces_placed % 5 == 0 and not res[3]:
                 game.receive_garbage(random.randint(1, 2))
            
            # 状态更新
            next_board_view, next_ctx_view = game.get_state()
            next_board = next_board_view.copy()
            next_ctx = next_ctx_view.copy()
            cur_metrics = calculate_heuristics(next_board)
            
            step_result = {
                'lines_cleared': res[0], 'damage_sent': res[1],
                'attack_type': res[2], 'game_over': res[3], 'b2b_count': res[4] ,'combo': res[5]
            }
            
            reward, force_over = get_reward(
                step_result, cur_metrics, prev_metrics, steps,
                context=next_ctx, prev_context=prev_ctx,
                episode_stats=episode_stats, is_training=True
            )
            
            score_sum += reward
            lines_sum += step_result['lines_cleared']
            holes_sum += cur_metrics['holes']
            if step_result['lines_cleared'] == 4:
                tetris_count += 1
                
            steps += 1
            board = next_board
            prev_ctx = next_ctx
            prev_metrics = cur_metrics
            
            if step_result['game_over'] or force_over or steps > config.MAX_STEPS_TRAIN:
                break
        
        # --- 打包数据 ---
        final_value = np.tanh(score_sum)
        # 转换为 Numpy Array 以便快速传输
        np_boards = np.array(data_boards, dtype=np.int8)
        np_ctxs = np.array(data_ctxs, dtype=np.float32)
        np_probs = np.array(data_probs, dtype=np.float32)
        np_values = np.full((len(data_boards), 1), final_value, dtype=np.float32)
        stats = {
            'score': score_sum, 
            'lines': lines_sum, 
            'steps': steps,
            'holes_step': holes_sum / steps if steps > 0 else 0,
            'tetrises': tetris_count,
            'avg_holes': holes_sum / steps if steps > 0 else 0,  # Explicitly add for clarity (redundant but illustrative)
            # Add more if desired, e.g., 'max_height': cur_metrics['max_height'] (from final state)
        }
       
        # 发送数据
        try:
            # 传输的是紧凑的 Numpy 数组，而非包含大量小对象的 List
            data_queue.put((np_boards, np_ctxs, np_probs, np_values, stats))
        except queue.Full:
            # 如果队列满了，丢弃这局数据，避免 Worker 死锁
            pass
        
        # 显式 GC
        if rank == 0 and random.random() < 0.1:
            gc.collect()

# ==========================================
# 4. Dashboard 类 (保持原样，略微精简)
# ==========================================
class TrainingDashboard:
    def __init__(self):
        self.console = Console()
        self.log_buffer = []
        self.stats = {
            "game_count": 0, "memory_size": 0, "avg_score": 0.0,
            "loss": 0.0, "pps": 0.0, "tetrises": 0.0, "train_steps": 0,
            "avg_holes": 0.0,  # New: Average holes per step
            "avg_lines": 0.0,  # New: Average lines cleared per game
        }
        # ... (existi
        self.start_time = time.time()

    def log(self, message):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_buffer.append(f"[{ts}] {message}")
        if len(self.log_buffer) > 10: self.log_buffer.pop(0)

    def update_stats(self, **kwargs):
        self.stats.update(kwargs)

    def make_layout(self):
        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="body", ratio=1), Layout(name="footer", size=8))
        elapsed = int(time.time() - self.start_time)
        header = f"Tetris RL (LunaMino) | Time: {elapsed//3600:02}:{(elapsed%3600)//60:02}:{elapsed%60:02}"
        layout["header"].update(Panel(header, style="white on blue"))
        
        grid = Table.grid(expand=True); grid.add_column(); grid.add_column()
        t1 = Table(box=box.ROUNDED)
        t1.add_column("Metric"); t1.add_column("Value")
        t1.add_row("Total Games", str(self.stats["game_count"]))
        t1.add_row("Memory", f"{self.stats['memory_size']:,}")
        t1.add_row("Loss", f"{self.stats['loss']:.4f}")
        
        t2 = Table(box=box.ROUNDED)
        t2.add_column("Performance"); t2.add_column("Value")
        t2.add_row("Avg Score", f"{self.stats['avg_score']:.1f}")
        t2.add_row("Tetris Rate", f"{self.stats['tetrises']:.2f}")
        t2.add_row("Avg Holes/Step", f"{self.stats['avg_holes']:.2f}")  # New row
        t2.add_row("Avg Lines/Game", f"{self.stats['avg_lines']:.1f}")  # New row

        grid.add_row(Panel(t1, title="Training"), Panel(t2, title="Eval (Avg 50)"))
        layout["body"].update(grid)
        layout["footer"].update(Panel("\n".join(self.log_buffer), title="Logs"))
        return layout

# ==========================================
# 5. 优化后的 Trainer Manager
# ==========================================
def train_manager():
    mp.set_start_method('spawn', force=True)
    if not os.path.exists(config.BACKUP_DIR): os.makedirs(config.BACKUP_DIR)
    
    dashboard = TrainingDashboard()
    
    with Live(dashboard.make_layout(), refresh_per_second=4, screen=True) as live:
        dashboard.log(f"Starting {config.NUM_WORKERS} workers on CPU...")
        
        # 1. 共享模型 (GPU)
        shared_model = TetrisPolicyValue().to(TRAINER_DEVICE)
        # 开启 channels_last 加速
        shared_model = shared_model.to(memory_format=torch.channels_last)
        shared_model.share_memory()
        
        optimizer = optim.Adam(shared_model.parameters(), lr=config.LR)
        scaler = GradScaler()
        
        # 2. 高效 Buffer
        buffer = NumpyReplayBuffer(config.MEMORY_SIZE)
        
        # 3. 加载 Checkpoint
        start_idx = 0
        if os.path.exists(config.CHECKPOINT_FILE):
            ckpt = torch.load(config.CHECKPOINT_FILE, map_location=TRAINER_DEVICE)
            shared_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_idx = ckpt['game_idx']
            dashboard.log(f"Loaded checkpoint: Game {start_idx}")
            
        if os.path.exists(config.MEMORY_FILE):
            buffer.load(config.MEMORY_FILE)
            dashboard.log(f"Loaded buffer: {buffer.size} samples")

        # 4. 启动 Workers
        data_queue = mp.Queue(maxsize=100) # 减小 Queue 大小，因为每个包变大了 (Whole Game)
        processes = []
        for rank in range(config.NUM_WORKERS):
            p = mp.Process(target=worker_process, args=(rank, shared_model, data_queue, WORKER_DEVICE))
            p.start()
            processes.append(p)
            
        # 统计滑动窗口
        from collections import deque
        scores_hist = deque(maxlen=50)
        tetris_hist = deque(maxlen=50)
        holes_hist = deque(maxlen=50)  # New: For average holes per step
        lines_hist = deque(maxlen=50)  # New: For lines cleared per game
        
        game_cnt = start_idx
        total_train_steps = 0
        
        try:
            while True:
                # 1. 接收数据 (Batch Receive)
                # 每次处理最多 20 个游戏的数据，避免 Trainer 过于繁忙不训练
                processed_games = 0
                while not data_queue.empty() and processed_games < 20:
                    try:
                        # Unpack numpy arrays
                        np_boards, np_ctxs, np_probs, np_values, stats = data_queue.get_nowait()
                        
                        buffer.add_batch(np_boards, np_ctxs, np_probs, np_values)
                        
                        scores_hist.append(stats['score'])
                        tetris_hist.append(stats['tetrises'])
                        holes_hist.append(stats['holes_step'])  # New: Track holes per step
                        lines_hist.append(stats['lines'])  # New: Track lines per game
                        
                        game_cnt += 1
                        processed_games += 1
                        
                        if game_cnt % 10 == 0:
                            dashboard.log(f"Game {game_cnt}: Score {stats['score']:.3f}")
                            
                    except queue.Empty:
                        break
                
                # 更新 UI 统计
                if processed_games > 0:
                    dashboard.update_stats(
                        game_count=game_cnt,
                        memory_size=buffer.size,
                        avg_score=np.mean(scores_hist) if scores_hist else 0,
                        tetrises=np.mean(tetris_hist) if tetris_hist else 0
                    )

                # 2. 训练循环 (如果有新数据或 Buffer 足够)
                if buffer.size >= config.BATCH_SIZE and processed_games > 0:
                    # 动态调整训练步数：Worker 越多，数据生成越快，需要更多步数来消耗
                    # 比率：每个新采样点被训练 4-8 次
                    # 每个游戏平均 200-500 步 -> 20个游戏约 6000 样本 -> Batch 512 -> 约 12 个 Batch
                    # Target: Train Ratio 4 -> 48 updates
                    
                    new_samples = processed_games * 300 # 估算
                    train_steps = max(2, int(new_samples / config.BATCH_SIZE * 4.0))
                    train_steps = min(train_steps, 50) # 上限防止阻塞太久

                    shared_model.train() # Enable BN updates
                    
                    avg_loss = 0
                    for _ in range(train_steps):
                        # 极速采样
                        b_board, b_ctx, b_probs, b_val = buffer.sample(config.BATCH_SIZE)
                        
                        # 转 Tensor (Non-blocking)
                        t_board = torch.from_numpy(b_board).to(TRAINER_DEVICE, non_blocking=True)
                        t_ctx = torch.from_numpy(b_ctx).to(TRAINER_DEVICE, non_blocking=True)
                        t_probs = torch.from_numpy(b_probs).to(TRAINER_DEVICE, non_blocking=True)
                        t_val = torch.from_numpy(b_val).to(TRAINER_DEVICE, non_blocking=True)
                        
                        # 确保输入维度 (B, 1, 20, 10)
                        if t_board.dim() == 3: t_board = t_board.unsqueeze(1)
                        t_board = t_board.contiguous(memory_format=torch.channels_last)
                        
                        optimizer.zero_grad()
                        with autocast(device_type='cuda'):
                            p, v = shared_model(t_board, t_ctx)
                            
                            p_loss = -torch.sum(t_probs * F.log_softmax(p, dim=1), dim=1).mean()
                            v_loss = F.mse_loss(v, t_val)
                            entropy = -torch.mean(torch.sum(F.softmax(p, dim=1) * F.log_softmax(p, dim=1), dim=1))
                            
                            loss = p_loss + v_loss - 0.01 * entropy
                            
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        avg_loss += loss.item()
                        total_train_steps += 1
                        
                    dashboard.update_stats(loss=avg_loss/train_steps, train_steps=total_train_steps)
                
                live.update(dashboard.make_layout())
                
                if game_cnt % 200 == 0 and processed_games > 0:
                    # 1. 保存最新的 checkpoint (覆盖写)
                    torch.save({
                        'model_state_dict': shared_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'game_idx': game_cnt
                    }, config.CHECKPOINT_FILE)
                    
                    # 2. [新增] 备份逻辑：每 1000 局游戏另存一份到 backups 文件夹
                    if game_cnt % 1000 == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # 文件名格式: model_game_12000_20231027_1030.pth
                        backup_name = f"model_game_{game_cnt}_{timestamp}.pth"
                        backup_path = os.path.join(config.BACKUP_DIR, backup_name)
                        
                        try:
                            # 复制刚才保存的 checkpoint 文件
                            shutil.copy(config.CHECKPOINT_FILE, backup_path)
                            dashboard.log(f"[Backup] Saved to {backup_name}")
                        except Exception as e:
                            dashboard.log(f"[Backup] Failed: {e}")

                # 3. 定期保存
                if game_cnt % 200 == 0 and processed_games > 0:
                    torch.save({
                        'model_state_dict': shared_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'game_idx': game_cnt
                    }, config.CHECKPOINT_FILE)
                
                if game_cnt % 1000 == 0 and processed_games > 0:
                    buffer.save(config.MEMORY_FILE)
                    
                if processed_games == 0:
                    time.sleep(0.05) # 避免空转

        except KeyboardInterrupt:
            dashboard.log("Stopping...")
            for p in processes: p.terminate()
            torch.save({'model_state_dict': shared_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'game_idx': game_cnt}, config.CHECKPOINT_FILE)
            buffer.save(config.MEMORY_FILE)

if __name__ == "__main__":
    train_manager()