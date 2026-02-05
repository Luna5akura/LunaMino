# ai/agent.py

import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from collections import deque
from .model import QNet
from .config import *
# form ai.utils import TetrisGame # 这里的引用如果在包内可能需要调整，保持原样即可

def normalize_action(action):
    # Normalize [x, y, rot, hold] to [0,1]
    x_norm = (action[0] - X_MIN) / (X_MAX - X_MIN)
    y_norm = (action[1] - Y_MIN) / (Y_MAX - Y_MIN)
    rot_norm = action[2] / ROT_MAX
    hold_norm = action[3] / HOLD_MAX
    return np.array([x_norm, y_norm, rot_norm, hold_norm], dtype=np.float32)

def compute_features(board):
    # 保持原样...
    heights = np.sum(board > 0, axis=0)
    agg_height = np.sum(heights)
    holes = 0
    for col in range(BOARD_WIDTH):
        col_data = board[:, col]
        first_occupied = np.where(col_data > 0)[0]
        if len(first_occupied) > 0:
            holes += np.sum(col_data[first_occupied[0]:] == 0)
    bumpiness = np.sum(np.abs(np.diff(heights)))
    return agg_height, holes, bumpiness

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, board, ctx, action, reward, next_board, next_ctx, done, legal_moves, next_legal_moves, landing_height):
        # 为了节省内存，建议存为 tuple 或 list，而不是对象
        self.buffer.append((board, ctx, action, reward, next_board, next_ctx, done, legal_moves, next_legal_moves, landing_height))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
    
    # === 新增：保存和读取经验池 ===
    def save(self, filepath):
        print(f"Saving replay buffer to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
            
    def load(self, filepath):
        if os.path.exists(filepath):
            print(f"Loading replay buffer from {filepath}...")
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
        else:
            print(f"Replay buffer file {filepath} not found, starting fresh.")

class DQNAgent:
    def __init__(self):
        self.q_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.best_reward = -float('inf') # 用于记录最佳表现

    def select_action(self, game, legal_moves, ids):
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(legal_moves))
        else:
            with torch.no_grad():
                # 批量推理
                actions_norm = torch.tensor([normalize_action(m) for m in legal_moves], dtype=torch.float32)
                # 扩展 state 以匹配 legal_moves 的数量
                state_board, state_ctx = game.get_state()
                boards = torch.tensor(np.repeat(state_board[None, ...], len(legal_moves), axis=0), dtype=torch.float32)
                ctxs = torch.tensor(np.repeat(state_ctx[None, ...], len(legal_moves), axis=0), dtype=torch.float32)
                
                q_values = self.q_net(boards, ctxs, actions_norm).squeeze()
                
                # 处理只有一个动作的情况，squeeze可能会把维数压没了
                if q_values.dim() == 0:
                    idx = 0
                else:
                    idx = torch.argmax(q_values).item()
        return legal_moves[idx]

    def compute_reward(self, step_res, prev_board, post_board, landing_height):
        lines_cleared, _, _, is_game_over, _, _ = step_res
        reward = REWARD_LINES * lines_cleared
        if is_game_over:
            reward += REWARD_GAME_OVER
        prev_agg_h, prev_holes, prev_bump = compute_features(prev_board)
        post_agg_h, post_holes, post_bump = compute_features(post_board)
        
        # 简单的 reward clipping 防止数值过大
        reward += REWARD_HEIGHT * (post_agg_h - prev_agg_h)
        reward += REWARD_HOLES * (post_holes - prev_holes)
        reward += REWARD_BUMPINESS * (post_bump - prev_bump)
        reward += REWARD_LANDING * landing_height
        return reward

    def optimize(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0 # Return 0 loss if not training
            
        batch = self.replay_buffer.sample(BATCH_SIZE)
        
        # Unpack batch (需要处理 numpy array 的堆叠)
        boards = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        ctxs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.stack([normalize_action(b[2]) for b in batch]), dtype=torch.float32)
        rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        next_boards = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32)
        next_ctxs = torch.tensor(np.stack([b[5] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[6] for b in batch], dtype=torch.float32)
        
        # Current Q
        current_q = self.q_net(boards, ctxs, actions).squeeze()
        
        # Target Q
        with torch.no_grad():
            target_q = rewards.clone()
            for i in range(BATCH_SIZE):
                if not batch[i][6]:  # Not done
                    next_legal = batch[i][8]  # next_legal_moves
                    if len(next_legal) > 0:
                        # 评估下一个状态所有可能的动作
                        next_actions_norm = torch.tensor([normalize_action(m) for m in next_legal], dtype=torch.float32)
                        
                        # 重复 next_board 和 next_ctx 以匹配 next_legal 的数量
                        nb_rep = next_boards[i:i+1].repeat(len(next_legal), 1, 1, 1).squeeze(0) # Handle dims carefully
                        nc_rep = next_ctxs[i:i+1].repeat(len(next_legal), 1).squeeze(0)
                        
                        # 这里要注意维度: next_boards[i]是(20,10), unsqueeze后是(1,20,10), repeat后是(N,20,10)
                        # model forward 期望 (Batch, 20, 10)
                        nb_input = next_boards[i].unsqueeze(0).repeat(len(next_legal), 1, 1)
                        nc_input = next_ctxs[i].unsqueeze(0).repeat(len(next_legal), 1)

                        next_q_values = self.target_net(nb_input, nc_input, next_actions_norm).squeeze()
                        
                        max_next_q = torch.max(next_q_values) if next_q_values.numel() > 1 else next_q_values
                        if next_q_values.numel() == 0: max_next_q = 0.0 # 防御性编程

                        target_q[i] += GAMMA * max_next_q
        
        # Loss
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        
        return loss.item()

    # === 新增：保存和读取检查点 ===
    def save_checkpoint(self, filename="checkpoint.pth"):
        state = {
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'best_reward': self.best_reward
        }
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename="checkpoint.pth"):
        if os.path.isfile(filename):
            print(f"Loading checkpoint from {filename}...")
            checkpoint = torch.load(filename)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            print(f"Loaded. Steps: {self.steps}, Epsilon: {self.epsilon:.4f}")
        else:
            print(f"Checkpoint {filename} not found.")