# ai/agent.py

import torch
torch.set_float32_matmul_precision('high')  # 开启 RTX 30 系列 Tensor Core 加速
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

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)


    def push(self, chosen_preview, ctx, reward, next_moves, next_previews, next_ctx, done):
        self.buffer.append((chosen_preview, ctx, reward, next_moves, next_previews, next_ctx, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        print(f"Saving buffer to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        if os.path.exists(filepath):
            print(f"Loading buffer from {filepath}...")
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
        else:
            print("Buffer not found.")

# ai/agent.py
class DQNAgent:
    def __init__(self):
        # 新增：自动检测设备 (CUDA代表N卡，MPS代表苹果M系芯片，否则退回CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🤖 Agent is using device: {self.device}")

        # 修改：将模型推送到指定的 device
        self.q_net = QNet().to(self.device)
        self.target_net = QNet().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # 注意：优化器必须在模型推到 GPU 之后再初始化！
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0001, fused=True)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.best_reward = -float('inf')

    @staticmethod
    def approximate_post_ctx(prev_ctx, move, preview_board, prev_board=None):
        use_hold = move[4] > 0
        current = prev_ctx[0]
        hold = prev_ctx[1]
        previews_ctx = prev_ctx[2:7]
        b2b = prev_ctx[7]
        ren = prev_ctx[8]
        can_hold_prev = prev_ctx[9]
        pending = prev_ctx[10]
        has_hold = hold != -1.0

        # Compute lines cleared from preview_board

        if prev_board is not None:
            prev_blocks = np.sum(prev_board > 0)
            post_blocks = np.sum(preview_board > 0)
            # 每消一行少 10 个方块。加入的方块占 4 格。
            lines = max(0, int(round((prev_blocks + 4 - post_blocks) / 10.0)))
        else:
            # 退化方案（注意 > 0）
            lines = np.sum(np.all(preview_board > 0, axis=1))

        # Approximate ren update
        if lines > 0:
            new_ren = ren + 1 if ren >= -1 else 0
        else:
            new_ren = -1

        # Approximate b2b update (conservative: assume no T-spin)
        new_b2b = b2b
        if lines > 0:
            if lines == 4:
                new_b2b = b2b + 1
            else:
                new_b2b = 0

        # Update piece, hold, and previews
        if use_hold:
            if not has_hold:
                new_current = previews_ctx[0]
                new_hold = current
                new_previews = previews_ctx.copy()
            else:
                new_current = hold
                new_hold = current
                new_previews = previews_ctx.copy()
        else:
            new_current = previews_ctx[0]
            new_hold = hold if has_hold else -1.0
            new_previews = previews_ctx[1:].copy()
            new_previews = np.append(new_previews, -1.0)

        # Always reset can_hold to 1.0 after step
        new_can_hold = 1.0

        # Keep pending_attack unchanged
        new_pending = pending

        # Construct new ctx
        new_hold_val = new_hold if new_hold >= 0 else -1.0
        new_ctx = np.concatenate(([new_current, new_hold_val], new_previews, [new_b2b, new_ren, new_can_hold, new_pending]))
        return new_ctx

    def select_action(self, previews, ctx, legal_moves):
        if len(legal_moves) == 0:
            return None, None
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(len(legal_moves))
        else:
            with torch.no_grad():
                # 传入 None 作为占位符，或者根据你当前的逻辑来
                post_ctxs =[self.approximate_post_ctx(ctx, legal_moves[i], previews[i], None) for i in range(len(previews))]
                
                # 【重点修复】：将推理用的 Tensor 送入 GPU
                post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32).to(self.device)
                boards_t = torch.tensor(previews, dtype=torch.float32).to(self.device)
                
                q_values = self.q_net(boards_t, post_ctx_t).squeeze()
                if q_values.dim() == 0:
                    idx = 0
                else:
                    idx = torch.argmax(q_values).item()
        return legal_moves[idx], previews[idx]

    def optimize(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        
        batch = self.replay_buffer.sample(BATCH_SIZE)

        curr_boards_np = np.stack([b[0] for b in batch])
        curr_ctxs_np = np.stack([b[1] for b in batch])
        rewards_np = np.array([b[2] for b in batch], dtype=np.float32)
        dones_np = np.array([b[6] for b in batch], dtype=np.float32)

        curr_boards = torch.as_tensor(curr_boards_np, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        curr_ctxs = torch.as_tensor(curr_ctxs_np, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        rewards = torch.as_tensor(rewards_np, dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
        # 用 bool 类型的掩码张量，方便后续做索引
        dones_t = torch.as_tensor(dones_np, dtype=torch.bool).to(self.device, non_blocking=True)

        # 【新增】：开启自动混合精度 (bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            current_q = self.q_net(curr_boards, curr_ctxs).squeeze(-1)

        with torch.no_grad():
            target_q = rewards.clone()
            
            # 【终极奥义：无论死没死，全部拼成固定尺寸矩阵，迎合 torch.compile！】
            # 因为之前在 Worker 中强制填充了 MAX_EVALS=20
            # 所以这里 next_boards_np 的形状绝对是 (BATCH_SIZE, 20, 20, 10)
            next_boards_np = np.stack([b[3] for b in batch]) 
            next_ctxs_np = np.stack([b[4] for b in batch])   
            
            # 展平前两个维度，直接生成 (BATCH_SIZE * 20, ...) 的固定矩阵
            N = BATCH_SIZE * 20
            next_b_t = torch.as_tensor(next_boards_np.reshape(N, 20, 10), dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            next_c_t = torch.as_tensor(next_ctxs_np.reshape(N, 11), dtype=torch.float32).pin_memory().to(self.device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                next_q_all = self.target_net(next_b_t, next_c_t).squeeze(-1)
            
            # 算完之后，还原回 (BATCH_SIZE, 20) 的形状并求当前状态下20个动作的最大Q值
            next_q_all = next_q_all.view(BATCH_SIZE, 20)
            max_q, _ = torch.max(next_q_all, dim=1)
            
            # 【优雅的掩码】：只给还没死的对局加上 Target Q，不用写任何 for 循环！
            valid_mask = ~dones_t
            target_q[valid_mask] += GAMMA * max_q[valid_mask]

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save_checkpoint(self, filename):
        # 使用 replace('_orig_mod.', '') 剥离编译前缀，保证保存的是干净的权重
        state = {
            'q_net': {k.replace('_orig_mod.', ''): v for k, v in self.q_net.state_dict().items()},
            'target_net': {k.replace('_orig_mod.', ''): v for k, v in self.target_net.state_dict().items()},
            'optim': self.optimizer.state_dict(),
            'eps': self.epsilon,
            'steps': self.steps,
            'best': self.best_reward
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        if os.path.exists(filename):
            ckpt = torch.load(filename, map_location=self.device, weights_only=False)
            self.q_net.load_state_dict(ckpt['q_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optim'])
            self.epsilon = ckpt['eps']
            self.steps = ckpt['steps']
            self.best_reward = ckpt.get('best', -float('inf'))
            print(f"Loaded checkpoint. Steps: {self.steps}")