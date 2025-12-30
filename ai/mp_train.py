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
from .utils import TetrisGame
from .model import TetrisPolicyValue
from .mcts import MCTS
from . import config
from .reward import get_reward, calculate_heuristics
class StatsMonitor:
    def __init__(self):
        self.data = defaultdict(float)
        self.counts = defaultdict(int)
    def update(self, key, value, mode='avg'):
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
        result = {}
        for k, v in self.data.items():
            if k in self.counts and self.counts[k] > 0:
                result[k] = v / self.counts[k]
            else:
                result[k] = v
        return result
def worker_process(rank, shared_model, data_queue, device):
    print(f"[Worker {rank}] Starting on {device}")
    seed = rank + int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    game = TetrisGame()
    # 确保 MCTS 使用 shared_model 且处于 eval 模式
    # 注意：在多进程中直接使用 shared_model 可能会慢，
    # 推荐每个 worker 拥有一个 local_model，并在每局开始前 load_state_dict(shared_model)
    local_model = TetrisPolicyValue().to(device)
    local_model.load_state_dict(shared_model.state_dict())
    local_model.eval()
   
    mcts = MCTS(local_model, device=device, num_simulations=config.MCTS_SIMS_TRAIN)
   
    total_worker_steps = 0
    start_time = time.time()
   
    while True:
        # 每局开始同步模型参数
        if total_worker_steps % 1000 == 0: # 稍微减少同步频率
             local_model.load_state_dict(shared_model.state_dict())
        game_seed = random.randint(0, 1000000)
        game.reset(game_seed)
        garbage_lines = random.randint(1, 2)
        game.receive_garbage(garbage_lines)
        game.receive_garbage(garbage_lines)
        print(f"[Worker {rank}] Added {garbage_lines} garbage lines at start.")
        monitor = StatsMonitor()
        steps = 0
        game_data = []
        board, ctx = game.get_state()
        prev_metrics = calculate_heuristics(board)
       
        while True:
            legal = game.get_legal_moves()
            if len(legal) == 0:
                # 死局模拟结果
                res = {'lines_cleared':0, 'damage_sent':0, 'attack_type':0, 'game_over':True, 'b2b_count':0, 'combo':0}
                break
            root = mcts.run(game)
            temp = 1.0 if steps < 20 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
           
            # 获取状态并 COPY，防止 Tensor 负步长错误
            board, ctx = game.get_state()
            game_data.append([board.copy(), ctx, action_probs, None])
            local_idx = np.random.choice(len(action_probs), p=action_probs)
           
            move = legal[local_idx]
           
            # 修复 2: step 返回字典
            res = game.step(move[0], move[1], move[2], move[4])
            if steps % 10 == 0:
                print(f"[Worker {rank} Step {steps}] Lines: {res[0]}, Game Over: {res[3]}, Combo: {res[4]}")
            total_worker_steps += 1
            if total_worker_steps % 100 == 0:
                elapsed = time.time() - start_time
                pps = 100 / elapsed if elapsed > 0 else 0
                monitor.update("pps", pps, 'set')
                start_time = time.time()
            next_board, _ = game.get_state()
            cur_metrics = calculate_heuristics(next_board)
            step_result = {
                'lines_cleared': res[0],
                'damage_sent': res[1],
                'attack_type': res[2],
                'game_over': res[3],
                'combo': res[4]
            }
            step_reward, force_over = get_reward(step_result, cur_metrics, prev_metrics, steps, is_training=True)
            if force_over:
                step_result['game_over'] = True
            # 减少日志频率
            if steps % 50 == 0:
                print(f"[Worker {rank} Step {steps}] Reward: {step_reward:.2f}, Holes: {cur_metrics['holes']}")
           
            monitor.update("score", step_reward, 'sum')
            monitor.update("lines", step_result['lines_cleared'], 'sum')
            monitor.update("holes_step", cur_metrics['holes'], 'avg')
            monitor.update("max_height", cur_metrics['max_height'], 'max')
            monitor.update("bumpiness", cur_metrics['bumpiness'], 'avg')
            if step_result['lines_cleared'] > 0:
                monitor.update("reward_clear", step_reward, 'avg')
            else:
                monitor.update("reward_normal", step_reward, 'avg')
            if step_result['lines_cleared'] == 4:
                monitor.update("tetrises", 1, 'sum')
            prev_metrics = cur_metrics
            steps += 1
            if step_result['game_over'] or steps > config.MAX_STEPS_TRAIN:
                break
        monitor.update("steps", steps, 'set')
        stats = monitor.get_summary()
        final_val = np.tanh(stats['score'] / 100.0)
        for item in game_data:
            item[3] = final_val
        data_queue.put((game_data, stats))
def save_memory_safe(memory, filename):
    temp_name = filename + ".tmp"
    try:
        data_to_save = list(memory)
        print(f"[System] Saving memory ({len(data_to_save)} items)...")
        with open(temp_name, 'wb') as f:
            pickle.dump(data_to_save, f)
        if os.path.exists(filename):
            os.remove(filename)
        os.rename(temp_name, filename)
        print("[System] Memory saved.")
    except Exception as e:
        print(f"[Error] Save memory failed: {e}")
        if os.path.exists(temp_name):
            os.remove(temp_name)
def train_manager():
    mp.set_start_method('spawn', force=True)
    backup_dir = config.BACKUP_DIR
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    print(f"[Manager] Initializing {config.NUM_WORKERS} workers on {config.DEVICE}...")
    shared_model = TetrisPolicyValue().to(config.DEVICE)
    shared_model.share_memory()
    optimizer = optim.Adam(shared_model.parameters(), lr=config.LR)
    start_idx = 0
    if os.path.exists(config.CHECKPOINT_FILE):
        ckpt = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
        shared_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_idx = ckpt['game_idx']
        print(f"[Manager] Resumed from game {start_idx}")
    data_queue = mp.Queue(maxsize=100)
    processes = []
    for rank in range(config.NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(rank, shared_model, data_queue, config.DEVICE))
        p.start()
        processes.append(p)
    memory = deque(maxlen=config.MEMORY_SIZE)
    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, 'rb') as f:
                memory.extend(pickle.load(f))
            print(f"[Manager] Memory loaded: {len(memory)} items")
        except Exception as e:
            print(f"[Manager] Memory load failed: {e}. Trying backup...")
            backups = [f for f in os.listdir(backup_dir) if f.startswith('mem_') and f.endswith('.pkl')]
            if backups:
                latest = sorted(backups, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)[0]
                backup_path = os.path.join(backup_dir, latest)
                with open(backup_path, 'rb') as f:
                    memory.extend(pickle.load(f))
                print(f"[Manager] Loaded backup {backup_path}: {len(memory)} items")
    game_cnt = start_idx
    stats_keys = ["score", "lines", "steps", "holes_step", "max_height", "reward_normal", "reward_clear", "tetrises"]
    history = {k: deque(maxlen=100) for k in stats_keys}
    print("[Manager] Training started...")
    try:
        while True:
            received_packets = 0
            new_data = []
            while not data_queue.empty() and received_packets < 20:
                packet = data_queue.get_nowait()
                new_data.extend(packet[0])
                stats = packet[1]
                for k in stats_keys:
                    history[k].append(stats.get(k, 0))
                game_cnt += 1
                received_packets += 1
                print(f"[Game {game_cnt}] Score: {stats.get('score', 0):.2f}, Lines: {stats.get('lines', 0):.2f}, Steps: {stats.get('steps', 0):.1f}, Avg Holes/Step: {stats.get('holes_step', 0):.2f}, Max Height: {stats.get('max_height', 0):.1f}, Avg Reward Normal: {stats.get('reward_normal', 0):.2f}, Avg Reward Clear: {stats.get('reward_clear', 0):.2f}, Tetrises: {stats.get('tetrises', 0):.2f}")
            if received_packets > 0:
                memory.extend(new_data)
                if game_cnt % 20 == 0:
                    avg = {k: np.mean(v) if len(v) > 0 else 0 for k, v in history.items()}
                    print("\n" + "-" * 75)
                    print(f"[Epoch {game_cnt}] Mem: {len(memory)} | Dev: {config.DEVICE}")
                    print(f" Score: {avg['score']:6.2f} | Steps: {avg['steps']:5.1f} | Lines: {avg['lines']:.2f}")
                    print(f" Holes/Step: {avg['holes_step']:6.2f} | MaxHt: {avg['max_height']:5.1f} | Tetrises: {avg['tetrises']:.2f}")
                    print("-" * 75 + "\n")
                if len(memory) >= config.BATCH_SIZE:
                    batch = random.sample(memory, config.BATCH_SIZE)
                    b_board = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in batch]).to(config.DEVICE)
                    b_ctx = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in batch]).to(config.DEVICE)
                    b_policy = torch.stack([torch.tensor(x[2], dtype=torch.float32) for x in batch]).to(config.DEVICE)
                    b_value = torch.tensor([x[3] for x in batch], dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
                    optimizer.zero_grad()
                    p, v = shared_model(b_board, b_ctx)
                    loss = -torch.sum(b_policy * F.log_softmax(p, dim=1), dim=1).mean() + F.mse_loss(v, b_value)
                    loss.backward()
                    optimizer.step()
                    print(f"[Manager Train] Loss: {loss.item():.4f}")
                    if game_cnt % 100 == 0:
                        torch.save({
                            'model_state_dict': shared_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'game_idx': game_cnt
                        }, config.CHECKPOINT_FILE)
                        save_memory_safe(memory, config.MEMORY_FILE)
                        if game_cnt % 500 == 0:
                            ckpt_backup = f"ckpt_{game_cnt}.pth"
                            shutil.copy(config.CHECKPOINT_FILE, os.path.join(backup_dir, ckpt_backup))
                            print(f"[Backup] Checkpoint {ckpt_backup}")
                        if game_cnt % 2000 == 0:
                            mem_backup = f"mem_{game_cnt}.pkl"
                            shutil.copy(config.MEMORY_FILE, os.path.join(backup_dir, mem_backup))
                            print(f"[Backup] Memory {mem_backup}")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("[Manager] Stopping...")
        for p in processes:
            p.terminate()
            p.join()
        torch.save({
            'model_state_dict': shared_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'game_idx': game_cnt
        }, config.CHECKPOINT_FILE)
        save_memory_safe(memory, config.MEMORY_FILE)
        print("[Manager] Final save done.")
if __name__ == "__main__":
    train_manager()