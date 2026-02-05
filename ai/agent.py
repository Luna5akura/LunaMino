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

class DQNAgent:
    def __init__(self):
        self.q_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.best_reward = -float('inf')

    @staticmethod
    def approximate_post_ctx(prev_ctx, move, preview_board):
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
        lines = np.sum(np.all(preview_board == 1, axis=1))

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
                # Approximate post_ctx for each preview
                post_ctxs = [self.approximate_post_ctx(ctx, legal_moves[i], previews[i]) for i in range(len(previews))]
                post_ctx_t = torch.tensor(np.stack(post_ctxs), dtype=torch.float32)
                boards_t = torch.tensor(previews, dtype=torch.float32)
                q_values = self.q_net(boards_t, post_ctx_t).squeeze()
                if q_values.dim() == 0:
                    idx = 0
                else:
                    idx = torch.argmax(q_values).item()
        return legal_moves[idx], previews[idx]

    def compute_reward(self, step_res, prev_board, post_board, landing_height):
        lines_cleared, _, _, is_game_over, _, _ = step_res
        reward = REWARD_LINES * lines_cleared
        if is_game_over:
            reward += REWARD_GAME_OVER

        def get_holes(board):
            holes = 0
            for col in range(BOARD_WIDTH):
                col_data = board[:, col]
                first_occupied = np.where(col_data > 0)[0]
                if len(first_occupied) > 0:
                    holes += np.sum(col_data[first_occupied[0]:] == 0)
            return holes

        def get_height(board):
            return np.sum(board > 0)

        prev_holes = get_holes(prev_board)
        post_holes = get_holes(post_board)
        prev_h = get_height(prev_board)
        post_h = get_height(post_board)

        reward += REWARD_HOLES * (post_holes - prev_holes)
        reward += REWARD_HEIGHT * (post_h - prev_h)
        return reward

    def optimize(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        batch = self.replay_buffer.sample(BATCH_SIZE)

        curr_boards = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        curr_ctxs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b[6] for b in batch], dtype=torch.float32)

        current_q = self.q_net(curr_boards, curr_ctxs).squeeze()

        with torch.no_grad():
            target_q = rewards.clone()
            valid_indices = [i for i in range(BATCH_SIZE) if not batch[i][6]]
            if valid_indices:
                all_next_boards = []
                all_next_ctxs = []
                split_sizes = []
                for i in valid_indices:
                    next_moves = batch[i][3]
                    next_previews = batch[i][4]
                    next_prev_ctx = batch[i][5]
                    count = len(next_previews)
                    if count > 0:
                        post_ctxs = [DQNAgent.approximate_post_ctx(next_prev_ctx, next_moves[j], next_previews[j]) for j in range(count)]
                        all_next_boards.append(next_previews)
                        all_next_ctxs.append(np.stack(post_ctxs))
                        split_sizes.append(count)
                    else:
                        split_sizes.append(0)
                if len(all_next_boards) > 0:
                    big_board_batch = torch.tensor(np.concatenate(all_next_boards), dtype=torch.float32)
                    big_ctx_batch = torch.tensor(np.concatenate(all_next_ctxs), dtype=torch.float32)
                    all_next_q_values = self.target_net(big_board_batch, big_ctx_batch).squeeze()
                    if all_next_q_values.dim() == 0:
                        all_next_q_values = all_next_q_values.unsqueeze(0)
                    cursor = 0
                    for idx, count in enumerate(split_sizes):
                        i = valid_indices[idx]
                        if count > 0:
                            sample_qs = all_next_q_values[cursor:cursor + count]
                            if sample_qs.numel() > 0:
                                max_q = torch.max(sample_qs)
                                target_q[i] += GAMMA * max_q
                            cursor += count

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save_checkpoint(self, filename):
        state = {
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'eps': self.epsilon,
            'steps': self.steps,
            'best': self.best_reward
        }
        torch.save(state, filename)

    def load_checkpoint(self, filename):
        if os.path.exists(filename):
            ckpt = torch.load(filename)
            self.q_net.load_state_dict(ckpt['q_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optim'])
            self.epsilon = ckpt['eps']
            self.steps = ckpt['steps']
            self.best_reward = ckpt.get('best', -float('inf'))
            print(f"Loaded checkpoint. Steps: {self.steps}")