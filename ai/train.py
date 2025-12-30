# ai/train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import pickle
import time
from collections import deque
from .utils import TetrisGame
from .model import TetrisPolicyValue
from .mcts import MCTS
from .reward import get_reward, calculate_heuristics
from . import config

def normalize_score(score):
    return np.tanh(score / 100.0)

def battle_simulation(net, mcts_sims, render=False):
    print("[Simulation] Starting simulation, render=", render)
    with TetrisGame() as game:
        garbage_lines = random.randint(1, 2)
        game.receive_garbage(garbage_lines)
        game.receive_garbage(garbage_lines)
        print(f"[Simulation] Added {garbage_lines} garbage lines at start.")
        if render:
            game.enable_render()
        mcts = MCTS(net, device=config.DEVICE, num_simulations=mcts_sims)

        training_data = []
        steps = 0
        total_score = 0.0
        total_lines = 0
        tetris_count = 0
        reward_sum_clear = 0.0
        count_clear = 0
        reward_sum_normal = 0.0
        count_normal = 0
        hole_sum = 0

        board, ctx = game.get_state()
        # 修复 1: 确保 board 内存连续，防止后续 Tensor 转换报错
        board = board.copy()
        prev_metrics = calculate_heuristics(board)

        while True:
            if render:
                game.render()

            root = mcts.run(game)
            temp = 1.0 if steps < 30 else 0.5
            action_probs = mcts.get_action_probs(root, temp=temp)
            # 获取当前状态用于存储
            s_board, s_ctx = game.get_state()
            training_data.append([s_board.copy(), s_ctx, action_probs, None])
            legal = game.get_legal_moves()
            if len(legal) == 0:
                break  # 死局
            local_idx = np.random.choice(len(action_probs), p=action_probs)  # 直接 choice len(legal_count)
            move = legal[local_idx]  # 直接用 local_idx

            # 修复 2: game.step 返回的是字典
            res = game.step(move[0], move[1], move[2], move[4])
            
            if render:
                game.render()

            next_board, _ = game.get_state()
            cur_metrics = calculate_heuristics(next_board)

            # 修复 2: 适配字典返回值
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

            total_score += step_reward
            steps += 1
            hole_sum += cur_metrics['holes']

            if step_result['lines_cleared'] > 0:
                total_lines += step_result['lines_cleared']
                reward_sum_clear += step_reward
                count_clear += 1
                if step_result['lines_cleared'] == 4:
                    tetris_count += 1
            else:
                reward_sum_normal += step_reward
                count_normal += 1

            if steps % 10 == 0:
                print(f"[Step {steps}] Reward: {step_reward:.2f}, Lines: {step_result['lines_cleared']}, Holes: {cur_metrics['holes']}, Max Height: {cur_metrics['max_height']}")

            prev_metrics = cur_metrics

            if step_result['game_over'] or steps > config.MAX_STEPS_TRAIN:
                break

        final_value = normalize_score(total_score)
        for item in training_data:
            item[3] = final_value

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

        print(f"[Simulation End] Score: {stats['score']:.2f}, Steps: {stats['steps']}, Lines: {stats['lines']}, Tetrises: {stats['tetrises']}")
        print(f" Avg Holes: {stats['avg_holes']:.2f}, Max Height: {stats['max_height']}, Avg Reward Normal: {stats['avg_r_normal']:.2f}, Clear: {stats['avg_r_clear']:.2f}")

        return training_data, stats



def save_checkpoint(net, optimizer, memory, game_idx):
    print(f"[Saving] Checkpoint at game {game_idx}")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'game_idx': game_idx
    }, config.CHECKPOINT_FILE)
    with open(config.MEMORY_FILE, 'wb') as f:
        pickle.dump(list(memory), f)
    print("[Saving] Done.")

def load_checkpoint(net, optimizer):
    start_idx = 0
    memory = deque(maxlen=config.MEMORY_SIZE)
    if os.path.exists(config.CHECKPOINT_FILE):
        print("[Loading] Checkpoint found.")
        checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_idx = checkpoint['game_idx']
    if os.path.exists(config.MEMORY_FILE):
        with open(config.MEMORY_FILE, 'rb') as f:
            memory.extend(pickle.load(f))
        print(f"[Loading] Memory loaded: {len(memory)} items.")
    return start_idx, memory

def train():
    net = TetrisPolicyValue().to(config.DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=config.LR)
    game_idx, memory = load_checkpoint(net, optimizer)

    print(f"[Train] Starting single-process training on {config.DEVICE}. Render enabled.")
    render = True

    try:
        while True:
            sims = config.MCTS_SIMS_TRAIN if not render else config.MCTS_SIMS_EVAL
            new_data, stats = battle_simulation(net, mcts_sims=sims, render=render)

            memory.extend(new_data)
            game_idx += 1

            print(f"[Game {game_idx}] Summary - Score: {stats['score']:.2f} | Steps: {stats['steps']} | Lines: {stats['lines']} | Tetrises: {stats['tetrises']}")
            print(f" Avg Holes: {stats['avg_holes']:.2f} | Max Height: {stats['max_height']} | Avg R Normal: {stats['avg_r_normal']:.2f} | Clear: {stats['avg_r_clear']:.2f}")

            if len(memory) < config.BATCH_SIZE:
                continue

            batch = random.sample(memory, config.BATCH_SIZE)
            b_board = torch.stack([torch.tensor(x[0], dtype=torch.float32) for x in batch]).to(config.DEVICE)
            b_ctx = torch.stack([torch.tensor(x[1], dtype=torch.float32) for x in batch]).to(config.DEVICE)
            b_policy = torch.stack([torch.tensor(x[2], dtype=torch.float32) for x in batch]).to(config.DEVICE)
            b_value = torch.tensor([x[3] for x in batch], dtype=torch.float32).unsqueeze(1).to(config.DEVICE)

            optimizer.zero_grad()
            p, v = net(b_board, b_ctx)
            loss = -torch.sum(b_policy * F.log_softmax(p, dim=1), dim=1).mean() + F.mse_loss(v, b_value)
            loss.backward()
            optimizer.step()

            print(f"[Train] Loss: {loss.item():.4f}")

            if game_idx % 20 == 0:
                save_checkpoint(net, optimizer, memory, game_idx)

    except KeyboardInterrupt:
        print("[Interrupt] Saving final checkpoint.")
        save_checkpoint(net, optimizer, memory, game_idx)

if __name__ == "__main__":
    train()