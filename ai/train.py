import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import pickle  # 用于保存经验池
from collections import deque
from .utils import TetrisGame
from .model import TetrisNet
from .mcts import MCTS

# Config
BATCH_SIZE = 64
LR = 0.001
MEMORY_SIZE = 10000
MCTS_SIMS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_FILE = "tetris_checkpoint.pth"
MEMORY_FILE = "tetris_memory.pkl"


def battle_simulation(net, mcts_sims=100, render=False): # 增加sims，开启渲染选项
    game = TetrisGame()
    if render:
        game.enable_render()

    mcts = MCTS(net, device=DEVICE, num_simulations=mcts_sims)
    
    training_data = []
    steps = 0
    total_score = 0
    
    while True:
        if render: game.render()
        
        # MCTS
        root = mcts.run(game)
        
        # 温度系数控制：前30步探索，后面利用
        temp = 1.0 if steps < 30 else 0.5 
        action_probs = mcts.get_action_probs(root, temp=temp)
        
        # 记录数据
        board, ctx, p_type = game.get_state()
        training_data.append([board, ctx, p_type, action_probs, None])
        
        action_idx = np.random.choice(len(action_probs), p=action_probs)
        
        # 解码
        use_hold = 0
        if action_idx >= 40:
            use_hold = 1
            action_idx -= 40 # 变回 0-39
            
        x = action_idx // 4
        rot = action_idx % 4
        
        # 执行
        res = game.step(x, rot, use_hold)
        if render: game.render()
        
        # --- 奖励函数优化 ---
        step_reward = 0.01 # 生存奖励
        if res['lines_cleared'] > 0:
            step_reward += res['lines_cleared'] * 0.2
        if res['damage_sent'] > 0:
            step_reward += res['damage_sent'] * 1.0
        if res['game_over']:
            step_reward -= 1.0
        
        total_score += step_reward
        steps += 1
        
        if res['game_over'] or steps > 500:
            break
            
    # Value Target 计算
    final_value = total_score
    # 简单的归一化 (-1 到 1)
    if final_value > 10: final_value = 1.0
    elif final_value < -1: final_value = -1.0
    else: final_value = final_value / 10.0
    
    processed_data = []
    for item in training_data:
        item[4] = final_value
        processed_data.append(item)
        
    if render:
        game.close_render() # 记得关闭窗口
        
    return processed_data, total_score

def save_checkpoint(net, optimizer, memory, game_idx):
    print(f"\n[Saving] Saving checkpoint to {CHECKPOINT_FILE}...")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_idx
    }, CHECKPOINT_FILE)
    
    print(f"[Saving] Saving memory ({len(memory)} items) to {MEMORY_FILE}...")
    with open(MEMORY_FILE, 'wb') as f:
        pickle.dump(memory, f)
    print("[Saving] Done.")

def load_checkpoint(net, optimizer):
    start_idx = 0
    memory = deque(maxlen=MEMORY_SIZE)
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"[Loading] Found checkpoint {CHECKPOINT_FILE}, loading...")
        checkpoint = torch.load(CHECKPOINT_FILE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_idx = checkpoint['game_idx']
        print(f"[Loading] Resuming from Game {start_idx}")
    else:
        print("[Loading] No checkpoint found, starting from scratch.")

    if os.path.exists(MEMORY_FILE):
        print(f"[Loading] Found memory file {MEMORY_FILE}, loading...")
        try:
            with open(MEMORY_FILE, 'rb') as f:
                loaded_memory = pickle.load(f)
                memory.extend(loaded_memory)
            print(f"[Loading] Restored {len(memory)} experiences.")
        except Exception as e:
            print(f"[Loading] Failed to load memory: {e}")
            
    return start_idx, memory

def train():
    net = TetrisNet().to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    
    # --- 断点续训逻辑 ---
    game_idx, memory = load_checkpoint(net, optimizer)
    
    print(f"Starting training on {DEVICE}...")
    
    try:
        while True:
            # 渲染策略：前5局渲染，之后每100局渲染一次
            do_render = (game_idx < 5) or (game_idx % 100 == 0)

            do_render = True
            
            # 运行游戏
            new_data, score = battle_simulation(net, mcts_sims=MCTS_SIMS, render=do_render)
            memory.extend(new_data)
            
            game_idx += 1
            print(f"Game {game_idx}: Score = {score:.2f}, Memory = {len(memory)}")
            
            if len(memory) < BATCH_SIZE:
                continue
                
            # 训练步
            batch = random.sample(memory, BATCH_SIZE)
            
            b_board = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_ctx = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_ptype = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.long).to(DEVICE)
            b_policy = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(DEVICE)
            b_value = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            p_logits, v_pred = net(b_board, b_ctx, b_ptype)
            
            log_probs = F.log_softmax(p_logits, dim=1)
            policy_loss = -torch.sum(b_policy * log_probs, dim=1).mean()
            value_loss = F.mse_loss(v_pred, b_value)
            
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            if game_idx % 10 == 0:
                print(f"  Loss: {loss.item():.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f})")
                
            # 自动保存 (每50局存一次)
            if game_idx % 50 == 0:
                save_checkpoint(net, optimizer, memory, game_idx)
                
    except KeyboardInterrupt:
        print("\n\n[Interrupt] Ctrl+C detected! Saving progress before exit...")
        save_checkpoint(net, optimizer, memory, game_idx)
        print("[Interrupt] Safe exit completed.")

if __name__ == "__main__":
    print(f'{DEVICE=}')
    train()