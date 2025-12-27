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
import shutil # 用于文件复制备份
from collections import deque

# 引入模块
from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS
from .reward import get_reward
from . import config # 引入统一配置

def worker_process(rank, shared_model, data_queue, device):
    # 设置种子
    torch.manual_seed(rank + int(time.time()))
    np.random.seed(rank + int(time.time()))
    random.seed(rank + int(time.time()))
    
    try:
        game = TetrisGame()
    except Exception as e:
        print(f"[Worker {rank}] Failed to init game: {e}")
        return

    # 使用 config 中的 MCTS_SIMS_TRAIN
    mcts = MCTS(shared_model, device=device, num_simulations=config.MCTS_SIMS_TRAIN)
    
    print(f"[Worker {rank}] Started on {device}")
    
    while True:
        # Reset game
        from .utils import lib
        seed = random.randint(0, 1000000)
        lib.ai_reset_game(game.ptr, seed)
        
        steps = 0
        game_data = []
        total_score = 0
        
        while True:
            root = mcts.run(game)
            
            temp = 1.0 if steps < 20 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
            
            board, ctx, p_type = game.get_state()
            game_data.append([board, ctx, p_type, action_probs, None])
            
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            
            use_hold = 0
            if action_idx >= 40:
                use_hold = 1
                action_idx -= 40
            
            res = game.step(action_idx // 4, action_idx % 4, use_hold)
            
            # 训练模式：开启 force_over
            next_board, _, _ = game.get_state()
            step_reward, force_over = get_reward(res, next_board, steps, is_training=True)
            
            if force_over:
                res['game_over'] = True
            
            total_score += step_reward
            steps += 1
            
            # 使用 config 中的 MAX_STEPS_TRAIN
            if res['game_over'] or steps > config.MAX_STEPS_TRAIN:
                break
        
        final_val = np.tanh(total_score / 50.0)
        for item in game_data:
            item[4] = final_val
            
        data_queue.put(game_data)
        # 减少打印频率
        if rank == 0: 
            print(f"[Worker 0] Game Score: {total_score:.2f}")

def save_checkpoint(net, optimizer, game_cnt, memory=None):
    print(f"\n[Main] Saving checkpoint to {config.CHECKPOINT_FILE}...")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_cnt
    }, config.CHECKPOINT_FILE)
    
    if memory is not None:
        try:
            with open(config.MEMORY_FILE, 'wb') as f:
                pickle.dump(memory, f)
        except Exception as e:
            print(f"[Error] Failed to save memory: {e}")

    # --- 备份逻辑 ---
    # 每 500 局备份一次副本到 backups 文件夹
    if game_cnt % 500 == 0:
        backup_name = f"checkpoint_{game_cnt}.pth"
        backup_path = os.path.join(config.BACKUP_DIR, backup_name)
        print(f"[Backup] Creating backup: {backup_path}")
        try:
            shutil.copy(config.CHECKPOINT_FILE, backup_path)
            
            # 也可以选择性备份 memory (因为文件很大，可能不需要每次都备份)
            if memory is not None and game_cnt % 2000 == 0:
                 mem_backup_name = f"memory_{game_cnt}.pkl"
                 mem_backup_path = os.path.join(config.BACKUP_DIR, mem_backup_name)
                 print(f"[Backup] Creating memory backup: {mem_backup_path}")
                 shutil.copy(config.MEMORY_FILE, mem_backup_path)
        except Exception as e:
            print(f"[Backup] Failed: {e}")
            
    print("[Main] Save Done.")

def train_manager():
    mp.set_start_method('spawn', force=True)
    
    shared_model = TetrisNet().to(config.DEVICE)
    shared_model.share_memory()
    
    optimizer = optim.Adam(shared_model.parameters(), lr=config.LR)
    
    start_idx = 0
    if os.path.exists(config.CHECKPOINT_FILE):
        print(f"[Main] Loading checkpoint {config.CHECKPOINT_FILE}...")
        ckpt = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
        shared_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_idx = ckpt['game_idx']

    data_queue = mp.Queue(maxsize=100)
    processes = []
    
    for rank in range(config.NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(rank, shared_model, data_queue, config.DEVICE))
        p.start()
        processes.append(p)
        
    print(f"[Main] Started {config.NUM_WORKERS} workers.")
    
    memory = deque(maxlen=config.MEMORY_SIZE)
    if os.path.exists(config.MEMORY_FILE):
        print(f"[Main] Loading memory {config.MEMORY_FILE}...")
        try:
            with open(config.MEMORY_FILE, 'rb') as f:
                memory.extend(pickle.load(f))
            print(f"[Main] Loaded {len(memory)} experiences.")
        except:
            print("[Main] Failed to load memory, starting fresh.")
            
    game_cnt = start_idx
    total_samples = 0
    
    try:
        while True:
            while not data_queue.empty():
                try:
                    new_games = data_queue.get_nowait()
                    memory.extend(new_games)
                    game_cnt += 1
                    total_samples += len(new_games)
                    if game_cnt % 50 == 0:
                        print(f"[Main] Game {game_cnt}, Memory {len(memory)}")
                except:
                    break
            
            if len(memory) < config.BATCH_SIZE:
                time.sleep(1)
                continue
            
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
            
            # 定期保存 (使用 config 的保存频率)
            if game_cnt % 100 == 0 and game_cnt > start_idx:
                save_checkpoint(shared_model, optimizer, game_cnt, memory)
                start_idx = game_cnt

    except KeyboardInterrupt:
        print("[Main] Terminating...")
        for p in processes:
            p.terminate()
            p.join()
        save_checkpoint(shared_model, optimizer, game_cnt, memory)

if __name__ == "__main__":
    train_manager()