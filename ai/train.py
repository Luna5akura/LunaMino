# ai/train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import sys
import pickle
import shutil
from collections import deque

from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS
from .reward import get_reward
from . import config # 引入统一配置

def normalize_score(score):
    return np.tanh(score / 50.0)

def battle_simulation(net, mcts_sims, render=False):
    game = TetrisGame()
    if render:
        game.enable_render()

    mcts = MCTS(net, device=config.DEVICE, num_simulations=mcts_sims)
    
    training_data = []
    steps = 0
    total_score = 0
    
    while True:
        if render: game.render()
        
        root = mcts.run(game)
        
        # 观看模式下，温度系数可以设低一点，看它最好的表现
        # 或者保持随机性以收集数据
        temp = 0.5 
        action_probs = mcts.get_action_probs(root, temp=temp)
        
        board, ctx, p_type = game.get_state()
        training_data.append([board, ctx, p_type, action_probs, None])
        
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        use_hold = 0
        if action_idx >= 40:
            use_hold = 1
            action_idx -= 40
            
        res = game.step(action_idx // 4, action_idx % 4, use_hold)
        if render: game.render()
        
        # --- 关键修改 ---
        # 传入 is_training=False，禁用“强制空洞结束”
        # 这样即使 AI 玩崩了，也会继续挣扎，直到真正死亡
        next_board, _, _ = game.get_state()
        step_reward, force_over = get_reward(res, next_board, steps, is_training=False)

        # 在 Eval 模式下，忽略 force_over 标记
        # 只有真正的 game_over 才会结束
        
        total_score += step_reward
        steps += 1
        
        # 使用 EVAL 的最大步数 (50000)，解决提前结束问题
        if res['game_over'] or steps > config.MAX_STEPS_EVAL:
            break
            
    final_value = normalize_score(total_score)
    for item in training_data:
        item[4] = final_value
        
    if render:
        game.close_render()
        
    return training_data, total_score

def save_checkpoint(net, optimizer, memory, game_idx):
    # 复用 mp_train 类似的保存逻辑，但单机版通常不需要备份，
    # 或者如果这是专门用来微调的，也可以加上备份逻辑
    print(f"\n[Saving] Saving checkpoint to {config.CHECKPOINT_FILE}...")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_idx
    }, config.CHECKPOINT_FILE)
    
    # 简单的覆盖保存 memory
    try:
        with open(config.MEMORY_FILE, 'wb') as f:
            pickle.dump(memory, f)
    except Exception as e:
         print(f"[Error] Failed to save memory: {e}")
         
    # 单机版训练较慢，每 100 局备份一次即可
    if game_idx % 100 == 0:
        backup_path = os.path.join(config.BACKUP_DIR, f"single_ckpt_{game_idx}.pth")
        try:
            shutil.copy(config.CHECKPOINT_FILE, backup_path)
            print(f"[Backup] Saved to {backup_path}")
        except: pass

    print("[Saving] Done.")

def load_checkpoint(net, optimizer):
    start_idx = 0
    memory = deque(maxlen=config.MEMORY_SIZE)
    
    if os.path.exists(config.CHECKPOINT_FILE):
        print(f"[Loading] Found checkpoint {config.CHECKPOINT_FILE}...")
        try:
            checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_idx = checkpoint['game_idx']
            print(f"[Loading] Resuming from Game {start_idx}")
        except Exception as e:
            print(f"[Error] Corrupted: {e}")
    else:
        print("[Warning] No checkpoint found. Starting from scratch.")

    if os.path.exists(config.MEMORY_FILE):
        try:
            with open(config.MEMORY_FILE, 'rb') as f:
                memory.extend(pickle.load(f))
            print(f"[Loading] Restored {len(memory)} experiences.")
        except: pass
            
    return start_idx, memory

def train():
    net = TetrisNet().to(config.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=config.LR)
    
    game_idx, memory = load_checkpoint(net, optimizer)
    
    print(f"Starting Single Process Training on {config.DEVICE}...")
    
    try:
        while True:
            # 默认开启渲染，因为单机版主要目的就是为了看
            do_render = True 
            
            # 使用 config.MCTS_SIMS_EVAL (通常比训练时高，比如 50 或 100)
            new_data, score = battle_simulation(net, mcts_sims=config.MCTS_SIMS_EVAL, render=do_render)
            memory.extend(new_data)
            
            game_idx += 1
            print(f"Game {game_idx}: Score = {score:.2f}, Steps = {len(new_data)}")
            
            if len(memory) < config.BATCH_SIZE:
                continue
                
            batch = random.sample(memory, config.BATCH_SIZE)
            
            b_board = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(config.DEVICE)
            b_ctx = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(config.DEVICE)
            b_ptype = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.long).to(config.DEVICE)
            b_policy = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(config.DEVICE)
            b_value = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
            
            optimizer.zero_grad()
            p, v = net(b_board, b_ctx, b_ptype)
            loss = -torch.sum(b_policy * F.log_softmax(p, dim=1), dim=1).mean() + F.mse_loss(v, b_value)
            loss.backward()
            optimizer.step()
            
            # 自动保存
            if game_idx % 20 == 0:
                save_checkpoint(net, optimizer, memory, game_idx)
                
    except KeyboardInterrupt:
        print("\n[Interrupt] Saving...")
        save_checkpoint(net, optimizer, memory, game_idx)

if __name__ == "__main__":
    train()