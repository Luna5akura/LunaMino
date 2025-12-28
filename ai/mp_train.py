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
import shutil
from collections import deque, defaultdict

# 引入模块 (确保你在项目根目录下运行，或者设置了 PYTHONPATH)
from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS
from . import config
from .reward import get_reward, calculate_heuristics

# ==========================================
# 监控辅助类 (在 Worker 中运行，分散计算压力)
# ==========================================
class StatsMonitor:
    def __init__(self):
        self.data = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, key, value, mode='avg'):
        """
        :param mode: 
            'avg' = 累加后求平均 (如: 平均空洞数)
            'sum' = 累加 (如: 总分, 总行数)
            'max' = 取最大值 (如: 最大高度)
            'set' = 直接覆盖 (如: 步数)
        """
        if mode == 'avg':
            self.data[key] += value
            self.counts[key] += 1
        elif mode == 'sum':
            self.data[key] += value
        elif mode == 'max':
            if value > self.data[key]:
                self.data[key] = value
        elif mode == 'set':
            self.data[key] = value

    def get_summary(self):
        """返回计算好的最终统计字典"""
        result = {}
        for k, v in self.data.items():
            if k in self.counts and self.counts[k] > 0:
                result[k] = v / self.counts[k]
            else:
                result[k] = v
        return result

# ==========================================
# Worker Process
# ==========================================
def worker_process(rank, shared_model, data_queue, device):
    # 不同的进程设置不同的随机种子
    seed = rank + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        game = TetrisGame()
    except Exception as e:
        print(f"[Worker {rank}] Init failed: {e}")
        return

    mcts = MCTS(shared_model, device=device, num_simulations=config.MCTS_SIMS_TRAIN)
    # print(f"[Worker {rank}] Ready on {device}")
    

    total_worker_steps = 0
    start_time = time.time()
    while True:
        # 重置游戏
        from .utils import lib
        game_seed = random.randint(0, 1000000)
        lib.ai_reset_game(game.ptr, game_seed)
        
        # 初始化本局状态
        monitor = StatsMonitor()
        steps = 0
        game_data = []
        
        # 获取初始状态指标 (用于计算 Delta 奖励)
        board, _, _ = game.get_state()
        prev_metrics = calculate_heuristics(board)

        while True:
            # --- 1. MCTS 决策 ---
            root = mcts.run(game)
            
            # 前几步增加随机性 (Temperature)，防止过早收敛
            temp = 1.0 if steps < 20 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
            
            # 收集训练数据
            board, ctx, p_type = game.get_state()
            game_data.append([board, ctx, p_type, action_probs, None])
            
            # --- 2. 选择并执行动作 ---
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            use_hold = 1 if action_idx >= 40 else 0
            if use_hold: action_idx -= 40
            
            res = game.step(action_idx // 4, action_idx % 4, use_hold)

            total_worker_steps += 1
            
            if total_worker_steps % 100 == 0:
                elapsed = time.time() - start_time
                pps = 100 / elapsed
                # 可以把这个 pps 放到 monitor 里，或者直接打印 debug
                monitor.update("pps", pps, 'set') 
                start_time = time.time()
            
            # --- 3. 获取新状态 & 计算奖励 ---
            next_board, _, _ = game.get_state()
            cur_metrics = calculate_heuristics(next_board)
            
            # 关键：传入 current 和 prev 以计算变化量
            step_reward, force_over = get_reward(res, cur_metrics, prev_metrics, is_training=True)
            
            if force_over:
                res['game_over'] = True
            
            # --- 4. 记录监控数据 (根据需要添加) ---
            monitor.update("score", step_reward, 'sum')
            monitor.update("lines", res['lines_cleared'], 'sum')
            monitor.update("holes_step", cur_metrics['holes'], 'avg') 
            monitor.update("max_height", cur_metrics['max_height'], 'max')
            monitor.update("bumpiness", cur_metrics['bumpiness'], 'avg')
            
            if res['lines_cleared'] > 0:
                monitor.update("reward_clear", step_reward, 'avg')
            else:
                monitor.update("reward_normal", step_reward, 'avg')
            
            if res['lines_cleared'] == 4:
                monitor.update("tetrises", 1, 'sum')

            # 更新循环变量
            prev_metrics = cur_metrics
            steps += 1
            
            if res['game_over'] or steps > config.MAX_STEPS_TRAIN:
                break
        
        # --- 5. 局结束处理 ---
        monitor.update("steps", steps, 'set')
        stats = monitor.get_summary()
        
        # 回填 Value (Game Outcome)
        # 使用 tanh 归一化总分，分母可以根据实际分数范围调整 (比如 100.0)
        final_val = np.tanh(stats['score'] / 100.0) 
        for item in game_data:
            item[4] = final_val
            
        data_queue.put((game_data, stats))

# ==========================================
# Manager Process
# ==========================================
# ... (前面的 imports 和 worker_process 保持不变) ...

# ==========================================
# 辅助函数：安全保存 Memory
# ==========================================
def save_memory_safe(memory, filename):
    """
    安全保存内存：先写入临时文件，再重命名，防止写入中途崩溃导致文件损坏
    """
    temp_name = filename + ".tmp"
    try:
        # 将 deque 转为 list 再保存，兼容性更好
        data_to_save = list(memory)
        print(f"[System] Saving memory ({len(data_to_save)} items) to disk...")
        
        with open(temp_name, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        # 原子操作：重命名
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_name, filename)
        print("[System] Memory saved successfully.")
    except Exception as e:
        print(f"[Error] Failed to save memory: {e}")
        if os.path.exists(temp_name):
            os.remove(temp_name)

# ==========================================
# Manager Process
# ==========================================

# ai/mp_train.py 中的 train_manager 函数

def train_manager():
    # Windows/MacOS 需要 spawn
    mp.set_start_method('spawn', force=True)
    
    # --- [新增] 确保备份文件夹存在 ---
    backup_dir = "backups"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"[Manager] Created backup directory: {backup_dir}")

    print(f"[Manager] Initializing {config.NUM_WORKERS} workers on {config.DEVICE}...")
    
    # ... (中间的初始化代码 shared_model, optimizer, data_queue 等保持不变) ...
    shared_model = TetrisNet().to(config.DEVICE)
    shared_model.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=config.LR)
    
    start_idx = 0 
    if os.path.exists(config.CHECKPOINT_FILE):
        try:
            ckpt = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
            shared_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_idx = ckpt['game_idx']
            print(f"[Manager] Resuming from Game {start_idx}")
        except Exception as e:
            print(f"[Manager] Checkpoint load failed: {e}")

    data_queue = mp.Queue(maxsize=100)
    processes = []
    for rank in range(config.NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(rank, shared_model, data_queue, config.DEVICE))
        p.start()
        processes.append(p)
    
    # 加载 Memory
    memory = deque(maxlen=config.MEMORY_SIZE)
    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, 'rb') as f:
                memory.extend(pickle.load(f))
            print(f"[Manager] Memory loaded: {len(memory)} items")
        except:
            print("[Manager] Memory load failed, starting fresh.")

    game_cnt = start_idx
    stats_keys = ["score", "lines", "steps", "holes_step", "max_height", "reward_normal", "reward_clear", "tetrises"]
    history = {k: deque(maxlen=100) for k in stats_keys}

    print("[Manager] Training started...")

    try:
        while True:
            # 1. 接收数据
            received_packets = 0
            while not data_queue.empty() and received_packets < 20:
                try:
                    packet = data_queue.get_nowait()
                    memory.extend(packet[0])
                    stats = packet[1]
                    for k in stats_keys:
                        history[k].append(stats.get(k, 0))
                    game_cnt += 1
                    received_packets += 1
                except:
                    break
            
            # 2. 打印日志 (每20局)
            if game_cnt > start_idx and game_cnt % 20 == 0 and received_packets > 0:
                avg = {k: np.mean(v) if len(v) > 0 else 0 for k, v in history.items()}
                print("-" * 75)
                print(f"[Epoch {game_cnt}] Mem: {len(memory)} | Dev: {config.DEVICE}")
                print(f"  Score    : {avg['score']:6.2f}  | Steps : {avg['steps']:5.1f} | Lines : {avg['lines']:.2f}")
                print(f"  Holes/Stp: {avg['holes_step']:6.2f}  | MaxHt : {avg['max_height']:5.1f} | Tetris: {avg['tetrises']:.2f}")
                print("-" * 75)
                
            # 3. 训练网络
            if len(memory) >= config.BATCH_SIZE:
                # ... (Batch采样和训练代码保持不变) ...
                batch = random.sample(memory, config.BATCH_SIZE)
                b_board = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                b_ctx = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                b_ptype = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.long).to(config.DEVICE)
                b_policy = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                b_value = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
                
                optimizer.zero_grad()
                p, v = shared_model(b_board, b_ctx, b_ptype)
                loss = -torch.sum(b_policy * F.log_softmax(p, dim=1), dim=1).mean() + F.mse_loss(v, b_value)
                loss.backward()
                optimizer.step()
                
                # ==========================================
                # --- [重点修改] 保存与备份逻辑 ---
                # ==========================================
                if game_cnt % 100 == 0:
                    # 1. 保存最新版 (用于断点续传)
                    print(f"[Saving] Updating latest checkpoint...")
                    torch.save({
                        'model_state_dict': shared_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'game_idx': game_cnt
                    }, config.CHECKPOINT_FILE)
                    
                    save_memory_safe(memory, config.MEMORY_FILE)

                    # 2. 创建 Checkpoint 副本 (每500局)
                    if game_cnt % 500 == 0:
                        ckpt_backup_name = f"ckpt_{game_cnt}.pth"
                        ckpt_backup_path = os.path.join(backup_dir, ckpt_backup_name)
                        try:
                            shutil.copy(config.CHECKPOINT_FILE, ckpt_backup_path)
                            print(f"[Backup] Copied checkpoint to {ckpt_backup_path}")
                        except Exception as e:
                            print(f"[Backup Error] Failed to copy checkpoint: {e}")

                    # 3. 创建 Memory 副本 (每2000局)
                    # Memory文件很大，不要太频繁备份，否则会卡顿
                    if game_cnt % 2000 == 0:
                        mem_backup_name = f"mem_{game_cnt}.pkl"
                        mem_backup_path = os.path.join(backup_dir, mem_backup_name)
                        try:
                            shutil.copy(config.MEMORY_FILE, mem_backup_path)
                            print(f"[Backup] Copied memory to {mem_backup_path}")
                        except Exception as e:
                            print(f"[Backup Error] Failed to copy memory: {e}")

            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Manager] Stopping...")
        for p in processes:
            p.terminate()
            p.join()
        
        # 退出时保存最后一次
        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'game_idx': game_cnt
        }, config.CHECKPOINT_FILE)
        save_memory_safe(memory, config.MEMORY_FILE)
        print("[Manager] Final save done.")

if __name__ == "__main__":
    train_manager()