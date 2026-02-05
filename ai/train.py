# ai/train.py

import os
import time
import numpy as np
from .agent import DQNAgent
from .config import *
from .utils import TetrisGame

# 保存路径设置
CHECKPOINT_DIR = "checkpoints"
BUFFER_FILE = os.path.join(CHECKPOINT_DIR, "replay_buffer.pkl")
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "tetris_ai.pth")
BEST_MODEL_FILE = os.path.join(CHECKPOINT_DIR, "best_tetris_ai.pth")

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

def train():
    agent = DQNAgent()
    
    # === 尝试读取旧的训练进度 ===
    agent.load_checkpoint(CHECKPOINT_FILE)
    agent.replay_buffer.load(BUFFER_FILE)

    print("Starting training...")
    
    for episode in range(MAX_EPISODES):
        # 这里的 seed 逻辑可以保留，但建议随着 steps 变化，或者完全随机
        current_seed = (episode + int(time.time())) % 10000 
        game = TetrisGame(seed=current_seed)
        game.reset(seed=current_seed)
        
        total_reward = 0
        total_loss = 0
        optimize_steps = 0
        total_lines = 0  # <--- 新增变量
        
        for step_num in range(MAX_STEPS_PER_EPISODE):
            prev_board, prev_ctx = game.get_state()
            legal_moves, ids = game.get_legal_moves()
            
            if len(legal_moves) == 0:
                break
                
            action = agent.select_action(game, legal_moves, ids)
            x, y, rot, land, hold = action 
            
            # 执行动作
            step_res = game.step(x, y, rot, hold)

            lines_cleared, _, _, done, _, _ = step_res
            total_lines += lines_cleared  # <--- 累加消行数
            
            post_board, post_ctx = game.get_state()
            reward = agent.compute_reward(step_res, prev_board, post_board, land)
            done = step_res[3]
            
            # 获取下一步的 legal moves 用于 Target Q 计算
            next_legal_moves, next_ids = game.get_legal_moves() if not done else ([], [])
            
            # 存入经验池
            # 注意：legal_moves 是 numpy array，我们存切片后的数据
            # action 是 numpy array, action[:4] 是用于网络的输入
            agent.replay_buffer.push(
                prev_board, prev_ctx, action[:4], reward, 
                post_board, post_ctx, done, 
                legal_moves[:, :5], 
                next_legal_moves[:, :5] if len(next_legal_moves) > 0 else np.empty((0,5)), 
                land
            )
            
            loss = agent.optimize()
            
            total_reward += reward

            if loss != 0:
                total_loss += loss
                optimize_steps += 1
                
            if done:
                break
        
        game.close()
        
        # === 日志打印 ===
        avg_loss = total_loss / optimize_steps if optimize_steps > 0 else 0
        print(f"Episode {episode + 1}/{MAX_EPISODES} | "
              f"Reward: {total_reward:.2f} | "
              f"Lines: {total_lines} | "  # <--- 这里打印消行数
              f"Avg Loss: {avg_loss:.4f} | "
              f"Epsilon: {agent.epsilon:.4f} | "
              f"Buffer: {len(agent.replay_buffer)}")


        agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        # === 保存最佳模型 ===
        if total_reward > agent.best_reward:
            agent.best_reward = total_reward
            print(f"--> New High Score! Saving best model (Reward: {total_reward:.2f})")
            agent.save_checkpoint(BEST_MODEL_FILE)
            
        # === 定期保存检查点 (每 50 轮) ===
        if (episode + 1) % 200 == 0:
            agent.save_checkpoint(CHECKPOINT_FILE)
            # 经验池通常很大，不需要太频繁保存，或者只在退出时保存
            agent.replay_buffer.save(BUFFER_FILE) 

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving progress...")
        # 可以在这里保存经验池和模型，防止意外退出丢失
        # agent.save_checkpoint(CHECKPOINT_FILE)
        # agent.replay_buffer.save(BUFFER_FILE)
        print("Done.")