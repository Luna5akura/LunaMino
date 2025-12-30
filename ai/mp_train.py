# ai/mp_train.py (modified for new structure)

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
from .model import TetrisPolicyValue
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
    print(f"[Worker {rank}] Starting on {device}")
    seed = rank + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        game = TetrisGame()
        print(f"[Worker {rank}] Game initialized")
    except Exception as e:
        print(f"[Worker {rank}] Init failed: {e}")
        return
    mcts = MCTS(shared_model, device=device, num_simulations=config.MCTS_SIMS_TRAIN)
    total_worker_steps = 0
    start_time = time.time()
    while True:
        # print(f"[Worker {rank}] Resetting game")
        from .utils import lib
        game_seed = random.randint(0, 1000000)
        lib.ai_reset_game(game.ptr, game_seed)
        garbage_lines = random.randint(1, 2)
        game.receive_garbage(garbage_lines)
        game.receive_garbage(garbage_lines)
        game.receive_garbage(garbage_lines)
        game.receive_garbage(garbage_lines)
        print(f"[Worker {rank}] Added {garbage_lines} garbage lines at start.")  # 可选日志
        monitor = StatsMonitor()
        steps = 0
        game_data = []
        board, _ = game.get_state()
        prev_metrics = calculate_heuristics(board)
        while True:
            legal = game.get_legal_moves()
            if len(legal) == 0: # 新: 处理无合法动作（game over）
                res = {'game_over': True} # 模拟 game_over
                break
           
            root = mcts.run(game)
            temp = 1.0 if steps < 20 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
            board, ctx = game.get_state()
            game_data.append([board, ctx, action_probs, None])
           
            # 新: 采样全局 idx
            global_idx = np.random.choice(len(action_probs), p=action_probs)
           
            # 新: 映射回局部 idx
            if root.legal_indices is None: # 不应发生，但防备
                raise ValueError(f"[Worker {rank}] No legal indices at step {steps}")
            local_idx = root.legal_indices.index(global_idx)
           
            move = legal[local_idx]
            res = game.step(move['x'], move['y'], move['rotation'], move['use_hold'])
           
            if rank == 0 and steps % 1 == 0:
                    print(f"[Worker {rank}] Step {steps}, res={res}")
            # 新: 检查 missed clear (occasional)
            if steps % 10 == 0:
                missed = False
                chosen_lines = res['lines_cleared']
                for test_move in legal:
                    clone = game.clone()
                    test_res = clone.step(test_move['x'], test_move['y'], test_move['rotation'], test_move['use_hold'])
                    if test_res['lines_cleared'] > chosen_lines:
                        missed = True
                        break
                    del clone
                if missed:
                    print(f"[DEBUG Worker {rank}] Step {steps}: Missed clear opportunity! Chosen lines={chosen_lines}")
           
            total_worker_steps += 1
            if total_worker_steps % 100 == 0:
                elapsed = time.time() - start_time
                pps = 100 / elapsed
                monitor.update("pps", pps, 'set')
                start_time = time.time()
            next_board, _ = game.get_state()
            cur_metrics = calculate_heuristics(next_board)
            step_reward, force_over = get_reward(res, cur_metrics, prev_metrics, is_training=True)
            print(f"[Worker {rank}] Step {steps}: Reward {step_reward:.2f}, Lines {res['lines_cleared']}, Holes {cur_metrics['holes']}")
            if force_over:
                res['game_over'] = True
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
            prev_metrics = cur_metrics
            steps += 1
            if res['game_over'] or steps > config.MAX_STEPS_TRAIN:
                # if rank == 0: print(f"[Worker {rank}] Game over at step {steps}")
                break
        monitor.update("steps", steps, 'set')
        stats = monitor.get_summary()
        final_val = np.tanh(stats['score'] / 100.0)
        for item in game_data:
            item[3] = final_val
        data_queue.put((game_data, stats))
        # if rank == 0: print(f"[Worker {rank}] Put packet with {len(game_data)} items")
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
    shared_model = TetrisPolicyValue().to(config.DEVICE)
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
                data = pickle.load(f)
                memory.extend(data)
            print(f"[Manager] Memory loaded from {config.MEMORY_FILE}: {len(memory)} items")
        except Exception as e:
            print(f"[Manager] Memory load failed from {config.MEMORY_FILE}: {str(e)}. Trying latest backup...")
            # Fallback: 找 backups/ 最新 mem_*.pkl
            backups = [f for f in os.listdir(backup_dir) if f.startswith('mem_') and f.endswith('.pkl')]
            if backups:
                latest_backup = sorted(backups, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
                backup_path = os.path.join(backup_dir, latest_backup)
                try:
                    with open(backup_path, 'rb') as f:
                        data = pickle.load(f)
                        memory.extend(data)
                    print(f"[Manager] Loaded from backup {backup_path}: {len(memory)} items")
                except Exception as e:
                    print(f"[Manager] Backup load also failed: {str(e)}. Starting fresh.")
            else:
                print("[Manager] No backups found. Starting fresh.")
    else:
        print("[Manager] No memory file found. Starting fresh.")
    game_cnt = start_idx
    stats_keys = ["score", "lines", "steps", "holes_step", "max_height", "reward_normal", "reward_clear", "tetrises"]
    history = {k: deque(maxlen=100) for k in stats_keys}
    print("[Manager] Training started...")
    try:
        while True:
            # 1. 接收数据
            received_packets = 0
            new_data = []
            while not data_queue.empty() and received_packets < 20:
                try:
                    packet = data_queue.get_nowait()
                    new_data.extend(packet[0])
                    stats = packet[1]
                    for k in stats_keys:
                        history[k].append(stats.get(k, 0))
                    game_cnt += 1
                    received_packets += 1
                    print(f"[Game {game_cnt}] Final Reward: {stats['score']:.2f}, Lines: {stats['lines']}")
                except:
                    break
          
            # print(f"[Manager] Received {received_packets} packets, game_cnt={game_cnt}")
            # 2. 只在有新数据时更新内存、训练、保存
            if received_packets > 0:
                memory.extend(new_data)
              
                # 打印日志 (每20局)
                if game_cnt % 20 == 0:
                    avg = {k: np.mean(v) if len(v) > 0 else 0 for k, v in history.items()}
                    print("\n","-" * 75)
                    print(f"[Epoch {game_cnt}] Mem: {len(memory)} | Dev: {config.DEVICE}")
                    print(f" Score : {avg['score']:6.2f} | Steps : {avg['steps']:5.1f} | Lines : {avg['lines']:.2f}")
                    print(f" Holes/Stp: {avg['holes_step']:6.2f} | MaxHt : {avg['max_height']:5.1f} | Tetris: {avg['tetrises']:.2f}")
                    print("-" * 75, "\n")
              
                # 3. 训练网络 (有新数据时才训练)
                if len(memory) >= config.BATCH_SIZE:
                    # ... (Batch采样和训练代码保持不变) ...
                    batch = random.sample(memory, config.BATCH_SIZE)
                    b_board = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                    b_ctx = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                    b_policy = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(config.DEVICE)
                    b_value = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
                  
                    optimizer.zero_grad()
                    p, v = shared_model(b_board, b_ctx)
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
                        if game_cnt % 100 == 0:
                            ckpt_backup_name = f"ckpt_{game_cnt}.pth"
                            ckpt_backup_path = os.path.join(backup_dir, ckpt_backup_name)
                            try:
                                shutil.copy(config.CHECKPOINT_FILE, ckpt_backup_path)
                                print(f"[Backup] Copied checkpoint to {ckpt_backup_path}")
                            except Exception as e:
                                print(f"[Backup Error] Failed to copy checkpoint: {e}")
                        # 3. 创建 Memory 副本 (每2000局)
                        # Memory文件很大，不要太频繁备份，否则会卡顿
                        if game_cnt % 100 == 0:
                            mem_backup_name = f"mem_{game_cnt}.pkl"
                            mem_backup_path = os.path.join(backup_dir, mem_backup_name)
                            try:
                                shutil.copy(config.MEMORY_FILE, mem_backup_path)
                                print(f"[Backup] Copied memory to {mem_backup_path}")
                            except Exception as e:
                                print(f"[Backup Error] Failed to copy memory: {e}")
          
            else:
                time.sleep(0.1) # No new data, sleep to avoid busy loop
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