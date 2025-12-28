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
import time
from collections import deque

from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS
from .reward import get_reward, calculate_heuristics
from . import config # 引入统一配置

def normalize_score(score):
    # 使用 tanh 归一化，分母设为 100 适应目前的分数膨胀
    return np.tanh(score / 100.0)

def battle_simulation(net, mcts_sims, render=False):
    """
    单局模拟函数
    """
    game = TetrisGame()
    if render:
        game.enable_render()

    mcts = MCTS(net, device=config.DEVICE, num_simulations=mcts_sims)
    
    training_data = []
    steps = 0
    total_score = 0
    
    # --- 统计变量 ---
    total_lines = 0
    tetris_count = 0
    reward_sum_clear = 0.0
    count_clear = 0
    reward_sum_normal = 0.0
    count_normal = 0
    hole_sum = 0
    
    # --- 1. 获取初始指标 (用于 Delta 奖励) ---
    board, ctx, p_type = game.get_state()
    prev_metrics = calculate_heuristics(board)
    
    while True:
        if render: game.render()
        
        # MCTS 运行
        root = mcts.run(game)
        
        # 训练模式下保持一定的探索性
        temp = 1.0 if steps < 30 else 0.5
        action_probs = mcts.get_action_probs(root, temp=temp)
        
        # 记录数据
        # 注意：这里需要重新获取 state，或者直接用上面的 board, ctx, p_type
        # 为了代码清晰，重新获取一次 current state 存入 buffer
        s_board, s_ctx, s_ptype = game.get_state()
        training_data.append([s_board, s_ctx, s_ptype, action_probs, None])
        
        # 选择动作
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        use_hold = 1 if action_idx >= 40 else 0
        if use_hold: action_idx -= 40
            
        # 执行步骤
        res = game.step(action_idx // 4, action_idx % 4, use_hold)
        if render: game.render()
        
        # --- 2. 获取新状态 & 计算指标 ---
        next_board, _, _ = game.get_state()
        cur_metrics = calculate_heuristics(next_board)
        
        # --- 3. 计算奖励 (传入 prev 和 cur) ---
        # 单机训练也开启 force_over (is_training=True)
        step_reward, force_over = get_reward(res, cur_metrics, prev_metrics, is_training=True)

        if force_over:
            res['game_over'] = True
            
        # --- 统计更新 ---
        total_score += step_reward
        steps += 1
        hole_sum += cur_metrics['holes']
        
        if res['lines_cleared'] > 0:
            total_lines += res['lines_cleared']
            reward_sum_clear += step_reward
            count_clear += 1
            if res['lines_cleared'] == 4:
                tetris_count += 1
        else:
            reward_sum_normal += step_reward
            count_normal += 1
            
        # 更新上一帧指标
        prev_metrics = cur_metrics
        
        # 这里的 max steps 可以用 config.MAX_STEPS_TRAIN
        if res['game_over'] or steps > config.MAX_STEPS_TRAIN:
            break
            
    final_value = normalize_score(total_score)
    for item in training_data:
        item[4] = final_value
        
    if render:
        game.close_render()
    
    # 计算本局统计摘要
    stats = {
        "score": total_score,
        "steps": steps,
        "lines": total_lines,
        "tetrises": tetris_count,
        "avg_holes": hole_sum / steps if steps > 0 else 0,
        "max_height": cur_metrics['max_height'],
        "avg_r_normal": reward_sum_normal / count_normal if count_normal > 0 else 0,
        "avg_r_clear": reward_sum_clear / count_clear if count_clear > 0 else 0
    }
        
    return training_data, stats

def save_checkpoint(net, optimizer, memory, game_idx):
    print(f"\n[Saving] Saving checkpoint to {config.CHECKPOINT_FILE}...")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_idx
    }, config.CHECKPOINT_FILE)
    
    try:
        with open(config.MEMORY_FILE, 'wb') as f:
            pickle.dump(memory, f)
    except Exception as e:
         print(f"[Error] Failed to save memory: {e}")

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
    print("Tip: Use 'mp_train.py' for faster training.")
    
    try:
        while True:
            # 单机版通常为了调试或观看，默认开启渲染
            # 如果你想跑得快一点，把这里改成 False
            do_render = True 
            
            # 使用 config 中定义的模拟次数
            # 如果是为了观看效果，建议用 MCTS_SIMS_EVAL (比如 50)
            # 如果是为了单机训练，建议用 MCTS_SIMS_TRAIN (比如 30)
            sims = config.MCTS_SIMS_TRAIN if not do_render else config.MCTS_SIMS_EVAL
            
            new_data, stats = battle_simulation(net, mcts_sims=sims, render=do_render)
            memory.extend(new_data)
            
            game_idx += 1
            
            # 打印像 mp_train 那样的详细日志
            print(f"[Game {game_idx}] Score: {stats['score']:.2f} | Steps: {stats['steps']} | Lines: {stats['lines']}")
            print(f"   Holes/Stp: {stats['avg_holes']:.2f} | MaxHt: {stats['max_height']}")
            print(f"   Rwrd: Norm {stats['avg_r_normal']:.2f} vs Clear {stats['avg_r_clear']:.2f}")
            print("-" * 40)
            
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